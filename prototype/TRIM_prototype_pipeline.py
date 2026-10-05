# TRIM pipeline with hierarchical release construction integrated into search.

# Goal: given the original dataset and a downstream model, search for a release
# whose validation loss <= the level-0 validation baseline + tolerance.
# At iteration t, TRIM uses the current min-K and assigns rows
# at the finest K-safe snapshot in S0..St, suppressing only the residual rows.
#
# Model roles:
#   - estimator: performs first-stage LGA ranking and is refreshed after each
#     iteration. The default LGA implementation reads flat linear parameters.
#   - proxy: performs second-stage candidate certification and is retrained for
#     each evaluated release.
#   - downstream: defines the utility baselines and performs final evaluation.
#
# Pipeline outline:
# 0. Split train/val/test; encode at level-0 and max level; freeze a
#    standardizer; fit downstream on full level-0 train to get validation and
#    test baselines; sample a small S0 bootstrap.
# 1. Fit the estimator and proxy on S0.
# 2. Greedy loop: optionally use the fitted proxy as the reference model to
#    filter remaining rows when the current estimator disagrees; otherwise
#    evaluate every remaining retention class; build retention_class and
#    vertical_refinement candidates; rank all of them with LGA on the estimator
#    theta; construct each top-k action's TRIM release and retrain the proxy
#    to obtain its validation-loss delta;
#    score each by utility_gain / math.log(max(current_k - next_k, 2)); apply the
#    best action. After a vertical refinement, run the post-vertical swap to
#    restore min(k). Finish the iteration by refitting the estimator on the
#    selected state. The selected action's validation loss
#    becomes the next search loss and controls stopping.
# 3. Final release: rebuild the selected iteration's release on the test split,
#    retrain downstream once, verify test utility, and persist the mixed-level
#    row assignments and suppression set.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import heapq
import hashlib
import json
from pathlib import Path
import re
import time
from types import MappingProxyType
from typing import Any
from uuid import uuid4
import math

import numpy as np
import torch

try:
    from .gpu_logistic import TorchLogisticRegression
    from .gpu_math import (
        _class_indices_tensor,
        classification_log_loss_tensor,
        logistic_gradient,
        multiclass_logistic_gradient,
        model_theta_tensor,
        to_device_tensor,
    )
    from .greedy_selection import SelectionState, calculate_leakage, _write_action_log
    from .initialization import stratified_sample_row_ids, validate_label_coverage
    from .backend_training import fit_standardizer, PassthroughStandardizer
    from .privacy_metrics import individual_tail_risk_stats
    from .enumeration_horizontal import (
        build_retention_class_groups,
        build_retention_class_position_index,
        select_encoded_attributes,
    )
    from .experiment_hooks import (
        ALL_ROWS_AT_MAX_GENERALIZATION,
        EVICT_LOW_K_POSTPROCESSING,
        FULL_SWAP_POSTPROCESSING,
        RETENTION_CLASS_ACTION,
        S0_INITIALIZATION,
        VERTICAL_REFINEMENT_ACTION,
        AblationActions,
        CandidateScorerRunContext,
        CandidateObservation,
        CandidateShortlistContext,
        CandidateStateScoreContext,
        TRIMIterationObservation,
        FixedRowCandidateBuilderContext,
        FixedRowCandidatePlan,
        RankedCandidate,
        SelectionScoreContext,
        default_selection_score,
    )
except ImportError:  # pragma: no cover - direct script import compatibility
    from gpu_logistic import TorchLogisticRegression
    from gpu_math import (
        _class_indices_tensor,
        classification_log_loss_tensor,
        logistic_gradient,
        multiclass_logistic_gradient,
        model_theta_tensor,
        to_device_tensor,
    )
    from greedy_selection import SelectionState, calculate_leakage, _write_action_log
    from initialization import stratified_sample_row_ids, validate_label_coverage
    from backend_training import fit_standardizer, PassthroughStandardizer
    from privacy_metrics import individual_tail_risk_stats
    from enumeration_horizontal import (
        build_retention_class_groups,
        build_retention_class_position_index,
        select_encoded_attributes,
    )
    from experiment_hooks import (
        ALL_ROWS_AT_MAX_GENERALIZATION,
        EVICT_LOW_K_POSTPROCESSING,
        FULL_SWAP_POSTPROCESSING,
        RETENTION_CLASS_ACTION,
        S0_INITIALIZATION,
        VERTICAL_REFINEMENT_ACTION,
        AblationActions,
        CandidateScorerRunContext,
        CandidateObservation,
        CandidateShortlistContext,
        CandidateStateScoreContext,
        TRIMIterationObservation,
        FixedRowCandidateBuilderContext,
        FixedRowCandidatePlan,
        RankedCandidate,
        SelectionScoreContext,
        default_selection_score,
    )


# TRIM defaults. Experiment runners may override these values.
DEFAULT_RANK_TOP_K = 3
DEFAULT_MAX_ITERATIONS = 30
VERTICAL_SWAP_MAX_ITERATIONS = 3000


def _callable_name(callback):
    if callback is None:
        return None
    module = getattr(callback, "__module__", None)
    qualified_name = getattr(callback, "__qualname__", None)
    if qualified_name is None:
        qualified_name = callback.__class__.__qualname__
    return f"{module}.{qualified_name}" if module else qualified_name


def _loaded_input_sha256(X_raw, y) -> str:
    """Hash the exact loaded rows without persisting any raw record."""
    import pandas as pd

    frame = X_raw if isinstance(X_raw, pd.DataFrame) else pd.DataFrame(X_raw)
    target = pd.Series(y, index=frame.index, name="__target__")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "columns": [str(column) for column in frame.columns],
                "dtypes": [str(dtype) for dtype in frame.dtypes],
                "row_count": len(frame),
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(
        pd.util.hash_pandas_object(frame, index=True, categorize=True)
        .to_numpy(dtype=np.uint64, copy=False)
        .tobytes()
    )
    digest.update(
        pd.util.hash_pandas_object(target, index=True, categorize=True)
        .to_numpy(dtype=np.uint64, copy=False)
        .tobytes()
    )
    return digest.hexdigest()


@dataclass
class TrimPipelineResult:
    task_type: str = "classification"
    loss_metric: str = "log_loss"
    selected_row_ids: list[Any] = field(default_factory=list)
    generalization_level: dict[str, int] = field(default_factory=dict)
    state: SelectionState | None = None
    initial_row_ids: list[Any] = field(default_factory=list)
    selected_retention_class_keys: list[Any] = field(default_factory=list)
    termination_condition: str | None = None
    final_actual_model_loss: float | None = None
    utility_constraint_met: bool | None = None
    validation_utility_constraint_met: bool | None = None
    baseline_val_loss: float | None = None
    loss_threshold_val: float | None = None
    baseline_test_loss: float | None = None
    loss_threshold_test: float | None = None
    original_leak_k: int | None = None
    original_leak_k_p1: float | None = None
    original_leak_k_p2: float | None = None
    original_leak_k_p3: float | None = None
    original_leak_k_p4: float | None = None
    original_leak_k_p5: float | None = None
    final_leak_k: int | None = None
    leak_k_p5: float | None = None
    tail_risk_p99: float | None = None
    final_leak_k_sorted: list[int] = field(default_factory=list)
    final_leak_k_95_percentile: int | None = None
    release_target_k: int | None = None
    release_assignment_counts: dict[int, int] = field(default_factory=dict)
    release_suppressed_row_ids: list[Any] = field(default_factory=list)
    release_snapshot_history: list[dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0
    timings: dict[str, float] = field(default_factory=dict)
    run_dir: str | None = None


@dataclass
class _SwapEncodeCache:
    encoded: Any
    codes: Any
    row_ids: Any
    row_id_to_position: dict[Any, int]
    selected_attributes: tuple
    attribute_columns: dict[str, list]


def _leak_distribution_stats(encode):
    """min-K, 1..5 percentile min-K, 95 percentile, sorted per-record k."""
    group_sizes = [] if encode.empty else encode.value_counts(sort=False).values
    return _leak_distribution_stats_from_group_sizes(group_sizes)


def _per_record_equivalence_class_sizes(encode):
    """Return each encoded row's own equivalence-class size, indexed by row ID."""
    if encode.empty:
        return encode.index.to_series().iloc[0:0].astype(np.intp)
    if encode.shape[1] == 0:
        raise ValueError("Privacy encoding must contain at least one QI column.")
    return encode.groupby(
        list(encode.columns),
        sort=False,
        dropna=False,
        observed=True,
    ).transform("size").astype(np.intp)


def _individual_tail_risk_stats(original_per_record_k, current_encode):
    """Compute absolute individual log risk over every loaded original row."""
    current_per_record_k = _per_record_equivalence_class_sizes(current_encode)
    return individual_tail_risk_stats(original_per_record_k.index, current_per_record_k)


def _empty_leak_distribution_stats():
    return {
        "leak_k": 0,
        "leak_k_p1": 0.0,
        "leak_k_p2": 0.0,
        "leak_k_p3": 0.0,
        "leak_k_p4": 0.0,
        "leak_k_p5": 0.0,
        "leak_k_sorted": [],
        "leak_k_95_percentile": None,
    }


def _leak_distribution_stats_from_group_sizes(group_sizes):
    group_sizes = np.asarray(group_sizes, dtype=np.intp)
    group_sizes = group_sizes[group_sizes > 0]
    if group_sizes.size == 0:
        return _empty_leak_distribution_stats()

    per_record_k = np.repeat(group_sizes, group_sizes)
    leak_k_sorted = sorted(int(v) for v in per_record_k.tolist())
    percentile_index = (95 * len(leak_k_sorted) + 99) // 100 - 1
    return {
        "leak_k": leak_k_sorted[0],
        "leak_k_p1": float(np.percentile(per_record_k, 1)),
        "leak_k_p2": float(np.percentile(per_record_k, 2)),
        "leak_k_p3": float(np.percentile(per_record_k, 3)),
        "leak_k_p4": float(np.percentile(per_record_k, 4)),
        "leak_k_p5": float(np.percentile(per_record_k, 5)),
        "leak_k_sorted": leak_k_sorted,
        "leak_k_95_percentile": leak_k_sorted[percentile_index],
    }


def _published_leak_snapshot(current_encode, selected_row_ids, initial_row_id_set,
                             position_index=None):
    if position_index is not None:
        return _published_leak_snapshot_from_position_index(
            selected_row_ids, position_index, initial_row_id_set)

    published_row_ids = _experiment_published_row_ids(
        current_encode, selected_row_ids, initial_row_id_set)
    if published_row_ids:
        group_sizes = [
            int(v)
            for v in current_encode.loc[published_row_ids]
            .value_counts(sort=False)
            .values
        ]
    else:
        group_sizes = []
    return (
        len(published_row_ids),
        _leak_distribution_stats_from_group_sizes(group_sizes),
        group_sizes,
    )


def _published_leak_snapshot_from_position_index(
    selected_row_ids, position_index, initial_row_id_set,
):
    row_id_to_position = position_index["row_id_to_position"]
    row_class_ids = position_index["row_class_ids"]
    class_count = len(position_index["class_total_sizes"])
    if class_count == 0:
        return 0, _empty_leak_distribution_stats(), []

    initial_row_id_set = set(initial_row_id_set)
    selected_counts = np.zeros(class_count, dtype=np.intp)
    seen_selected = set()
    for row_id in selected_row_ids:
        if row_id in initial_row_id_set or row_id in seen_selected:
            continue
        seen_selected.add(row_id)
        selected_counts[row_class_ids[row_id_to_position[row_id]]] += 1

    published_class_mask = selected_counts > 0
    if not np.any(published_class_mask):
        return 0, _empty_leak_distribution_stats(), []

    initial_counts = np.zeros(class_count, dtype=np.intp)
    for row_id in initial_row_id_set:
        position = row_id_to_position.get(row_id)
        if position is None:
            continue
        class_id = row_class_ids[position]
        if published_class_mask[class_id]:
            initial_counts[class_id] += 1

    group_sizes = selected_counts + initial_counts
    positive_group_sizes = group_sizes[group_sizes > 0]
    group_size_values = [int(v) for v in positive_group_sizes.tolist()]
    return (
        int(positive_group_sizes.sum()),
        _leak_distribution_stats_from_group_sizes(positive_group_sizes),
        group_size_values,
    )


def _open_run_dir(results_dir, run_tag):
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_uuid = uuid4().hex[:8]
    safe_tag = ""
    if run_tag:
        safe_tag = "_" + re.sub(r"[^A-Za-z0-9_-]+", "_", str(run_tag)).strip("_")
    run_dir_name = f"run_{run_timestamp}_{run_uuid}{safe_tag}"
    run_dir = Path(results_dir) / run_dir_name if results_dir is not None else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
    action_log_path = None
    if run_dir is not None:
        action_log_path = run_dir / (
            f"pipeline_selection_actions_{run_timestamp}_{run_uuid}.jsonl"
        )
    return run_dir, action_log_path


def _dump_json(obj, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, default=str, sort_keys=True, indent=2)


def _concat_release_model_parts(parts, ordered_row_ids):
    """Combine per-snapshot model inputs and restore dataset row order."""
    if not parts:
        raise ValueError("TRIM release model input requires at least one row part.")

    first_encode = parts[0][1]
    try:
        from scipy import sparse
    except ImportError:
        sparse = None
    if sparse is not None and sparse.issparse(first_encode):
        expected_feature_count = int(first_encode.shape[1])
        matrices = []
        for _row_ids, encode in parts:
            if (
                not sparse.issparse(encode)
                or int(encode.shape[1]) != expected_feature_count
            ):
                raise ValueError(
                    "TRIM release snapshots produced incompatible model-input columns."
                )
            matrices.append(encode)
        combined = sparse.vstack(matrices, format="csr")
        source_row_ids = [
            row_id
            for row_ids, _encode in parts
            for row_id in row_ids
        ]
        source_position = {
            row_id: position
            for position, row_id in enumerate(source_row_ids)
        }
        return combined[[source_position[row_id] for row_id in ordered_row_ids]]

    if hasattr(first_encode, "columns"):
        import pandas as pd

        expected_columns = list(first_encode.columns)
        for _row_ids, encode in parts[1:]:
            if list(encode.columns) != expected_columns:
                raise ValueError(
                    "TRIM release snapshots produced incompatible model-input columns."
                )
        return pd.concat([encode for _row_ids, encode in parts], axis=0).loc[
            ordered_row_ids
        ]

    if isinstance(first_encode, np.ndarray):
        expected_feature_count = int(first_encode.shape[1])
        matrices = []
        source_row_ids = []
        for row_ids, encode in parts:
            if (
                not isinstance(encode, np.ndarray)
                or encode.ndim != 2
                or int(encode.shape[1]) != expected_feature_count
            ):
                raise ValueError(
                    "TRIM release snapshots produced incompatible model-input columns."
                )
            source_row_ids.extend(row_ids)
            matrices.append(encode)
        combined = np.concatenate(matrices, axis=0)
        source_position = {
            row_id: position
            for position, row_id in enumerate(source_row_ids)
        }
        return np.ascontiguousarray(
            combined[[source_position[row_id] for row_id in ordered_row_ids]]
        )

    source_row_ids = []
    tensors = []
    for row_ids, encode in parts:
        source_row_ids.extend(row_ids)
        tensors.append(encode)
    combined = torch.cat(tensors, dim=0)
    source_position = {
        row_id: position for position, row_id in enumerate(source_row_ids)
    }
    ordered_positions = torch.as_tensor(
        [source_position[row_id] for row_id in ordered_row_ids],
        device=combined.device,
        dtype=torch.long,
    )
    return combined.index_select(0, ordered_positions)


def run_trim_pipeline(
    data_loader,
    generalization_tree_path,
    *,
    tolerance=0.1,
    nrows=None,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    max_iterations=DEFAULT_MAX_ITERATIONS,
    model_max_iter=50,
    initial_sample_size=None,
    initial_sample_fraction=None,
    initial_row_ids=None,
    rank_top_k=DEFAULT_RANK_TOP_K,
    model_config=None,
    model_factory=None,
    proxy_model_factory=None,
    estimator_model_factory=None,
    selected_attributes=None,
    standardizer="auto",
    results_dir=None,
    run_tag=None,
    device="cuda",
    dtype="float64",
    release_target_k=None,
    reference_model_filtering=False,
    ablation_actions=None,
    fixed_row_candidate_builder=None,
    candidate_shortlist=None,
    candidate_state_scorer_factory=None,
    selection_score=None,
    candidate_observer=None,
    stop_on_utility=True,
    record_iteration_test_metrics=False,
    trim_iteration_observer=None,
):
    """Run TRIM with release utility integrated into search and publication.

    ``release_target_k=None`` derives the release target from each current or
    candidate state's published min-K.
    ``None`` hooks and ``AblationActions()`` preserve the TRIM algorithm.
    ``reference_model_filtering=False`` evaluates every horizontal candidate.
    ``record_iteration_test_metrics`` is an experiment-only observation and
    its downstream retraining time is kept outside algorithm timings.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle as shuffle_rows

    if not isinstance(reference_model_filtering, (bool, np.bool_)):
        raise TypeError("reference_model_filtering must be a boolean.")
    reference_model_filtering = bool(reference_model_filtering)

    X_raw, y = data_loader.load(nrows=nrows)
    input_data_sha256 = _loaded_input_sha256(X_raw, y)
    tree_artifact_path = Path(generalization_tree_path).expanduser().resolve()
    generalization_tree_sha256 = hashlib.sha256(
        tree_artifact_path.read_bytes()
    ).hexdigest()
    X_raw, y = shuffle_rows(X_raw, y, random_state=random_state)
    pipeline_start_time = time.perf_counter()
    if getattr(data_loader, "task_type", "classification") != "classification":
        raise ValueError("TRIM supports classification datasets only.")
    if release_target_k is not None and int(release_target_k) < 1:
        raise ValueError("release_target_k must be a positive integer or None.")
    if initial_sample_size is not None and initial_sample_fraction is not None:
        raise ValueError(
            "initial_sample_size and initial_sample_fraction are mutually exclusive."
        )
    if initial_sample_fraction is not None:
        initial_sample_fraction = float(initial_sample_fraction)
        if not 0.0 < initial_sample_fraction <= 1.0:
            raise ValueError("initial_sample_fraction must be in (0, 1].")
    if initial_row_ids is not None and (
        initial_sample_size is not None or initial_sample_fraction is not None
    ):
        raise ValueError(
            "Explicit initial_row_ids cannot be combined with an initial sample "
            "size or fraction."
        )
    if fixed_row_candidate_builder is not None and (
        initial_row_ids is not None
        or initial_sample_size is not None
        or initial_sample_fraction is not None
    ):
        raise ValueError(
            "fixed_row_candidate_builder supplies its own initialization and "
            "cannot be combined with initial_row_ids, initial_sample_size, or "
            "initial_sample_fraction."
        )
    if ablation_actions is None:
        ablation_actions = AblationActions()
    elif not isinstance(ablation_actions, AblationActions):
        raise TypeError("ablation_actions must be an AblationActions instance or None.")
    if not isinstance(stop_on_utility, bool):
        raise TypeError("stop_on_utility must be bool.")
    if not isinstance(record_iteration_test_metrics, bool):
        raise TypeError("record_iteration_test_metrics must be bool.")
    for callback_name, callback in (
        ("fixed_row_candidate_builder", fixed_row_candidate_builder),
        ("candidate_shortlist", candidate_shortlist),
        ("candidate_state_scorer_factory", candidate_state_scorer_factory),
        ("selection_score", selection_score),
        ("candidate_observer", candidate_observer),
        ("trim_iteration_observer", trim_iteration_observer),
    ):
        if callback is not None and not callable(callback):
            raise TypeError(f"{callback_name} must be callable or None.")
    enabled_action_types = frozenset(ablation_actions.enabled_action_types)
    loss_metric_name = "log_loss"
    classification_classes = np.unique(np.asarray(y))
    class_count = int(len(classification_classes))
    use_multiclass = class_count > 2

    # --- Split: hold out test, then carve val out of the remainder. ----------
    X_trainval_raw, X_test_raw, y_trainval, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state,
        stratify=y,
    )
    val_size_of_trainval = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_trainval_raw, y_trainval, test_size=val_size_of_trainval,
        random_state=random_state,
        stratify=y_trainval,
    )

    # --- Generalization encodes. --------------------------------------------
    try:
        from . import dataloader as dataloader_module
    except ImportError:  # pragma: no cover
        import dataloader as dataloader_module

    original_generalization = dataloader_module.load_generalization_rules_from_file(
        file_path=generalization_tree_path, data_loader=data_loader,
        generalization_level={a: 0 for a in data_loader.feature_columns},
    )
    train_original_encode = original_generalization.encode(X_train_raw)
    val_original_encode = original_generalization.encode(X_val_raw)
    test_original_encode = original_generalization.encode(X_test_raw)

    qi_attributes = getattr(data_loader, "qi_attributes", None)
    if qi_attributes is None:
        qi_attributes = data_loader.feature_columns
    qi_attributes = tuple(qi_attributes)
    if selected_attributes is None:
        selected_attributes = getattr(data_loader, "qi_attributes", None)
    if selected_attributes is None:
        selected_attributes = qi_attributes
    selected_attributes = tuple(selected_attributes)

    max_generalization_level = {}
    try:
        from .greedy_selection import _load_max_generalization_level
    except ImportError:  # pragma: no cover
        from greedy_selection import _load_max_generalization_level
    max_generalization_level = _load_max_generalization_level(
        generalization_tree_path, qi_attributes,
    )
    fixed_row_candidate_plan = None
    fixed_row_candidate_build_time = 0.0
    if fixed_row_candidate_builder is not None:
        fixed_row_candidate_build_started_at = time.perf_counter()
        fixed_row_candidate_plan = fixed_row_candidate_builder(
            FixedRowCandidateBuilderContext(
                train_level0_encode=train_original_encode.copy(),
                selected_attributes=selected_attributes,
                random_state=int(random_state),
                max_generalization_level=dict(max_generalization_level),
            )
        )
        fixed_row_candidate_build_time = (
            time.perf_counter() - fixed_row_candidate_build_started_at
        )
        if not isinstance(fixed_row_candidate_plan, FixedRowCandidatePlan):
            raise TypeError(
                "fixed_row_candidate_builder must return FixedRowCandidatePlan."
            )
        if RETENTION_CLASS_ACTION not in enabled_action_types:
            raise ValueError(
                "A fixed row-candidate plan requires the retention-class "
                "action to be enabled."
            )
        if (
            VERTICAL_REFINEMENT_ACTION in enabled_action_types
            and ablation_actions.vertical_postprocessing
            != NO_VERTICAL_POSTPROCESSING
        ):
            raise ValueError(
                "A fixed row-candidate plan combined with vertical refinement "
                "must disable vertical postprocessing so the fixed partition "
                "remains intact."
            )
        if ablation_actions.initialization != S0_INITIALIZATION:
            raise ValueError(
                "A fixed row-candidate plan supplies its own initialization; "
                "AblationActions.initialization must remain 's0'."
            )
        expected_rows = list(train_original_encode.index)
        planned_rows = [
            row_id
            for _key, row_ids in fixed_row_candidate_plan.candidate_groups
            for row_id in row_ids
        ]
        if len(planned_rows) != len(set(planned_rows)):
            raise ValueError("Fixed row candidate groups must be disjoint.")
        if set(planned_rows) != set(expected_rows):
            missing = set(expected_rows) - set(planned_rows)
            extra = set(planned_rows) - set(expected_rows)
            raise ValueError(
                "Fixed row candidate groups must partition the training rows; "
                f"missing={len(missing)}, extra={len(extra)}."
            )
        if not set(fixed_row_candidate_plan.initial_row_ids).issubset(planned_rows):
            raise ValueError("Fixed-plan initial rows must belong to its groups.")
        if not set(fixed_row_candidate_plan.privacy_exempt_row_ids).issubset(
            fixed_row_candidate_plan.initial_row_ids
        ):
            raise ValueError(
                "Fixed-plan privacy-exempt rows must be initial rows."
            )
        if set(fixed_row_candidate_plan.generalization_level) != set(
            max_generalization_level
        ):
            raise ValueError(
                "Fixed-plan generalization_level must declare every QI attribute."
            )
        initial_generalization_level = dict(
            fixed_row_candidate_plan.generalization_level
        )
    else:
        initial_generalization_level = max_generalization_level
    current_generalization = dataloader_module.load_generalization_rules_from_file(
        file_path=generalization_tree_path, data_loader=data_loader,
        generalization_level=initial_generalization_level,
    )
    swap_encode_cache = _build_swap_encode_cache(
        current_generalization,
        X_train_raw,
        selected_attributes=selected_attributes,
    )
    initial_generalization_train_encode = swap_encode_cache.encoded

    # --- Original dataset leak. ---------------------------------------------
    original_dataset_encode = original_generalization.encode(X_raw)
    original_privacy_encode = select_encoded_attributes(
        original_dataset_encode,
        qi_attributes,
    )
    original_leak_stats = _leak_distribution_stats(original_privacy_encode)
    original_leak_k = original_leak_stats["leak_k"]
    original_leak_k_p1 = original_leak_stats["leak_k_p1"]
    original_leak_k_p2 = original_leak_stats["leak_k_p2"]
    original_leak_k_p3 = original_leak_stats["leak_k_p3"]
    original_leak_k_p4 = original_leak_stats["leak_k_p4"]
    original_leak_k_p5 = original_leak_stats["leak_k_p5"]
    original_per_record_k = _per_record_equivalence_class_sizes(
        original_privacy_encode
    )
    # Optional per-iteration downstream test retrains are experiment
    # observations, not part of the TRIM algorithm timings.
    total_curve_metric_time = 0.0

    # --- Device tensors. ----------------------------------------------------
    train_original_tensor = to_device_tensor(train_original_encode, device=device, dtype=dtype)
    val_original_tensor = to_device_tensor(val_original_encode, device=train_original_tensor.device, dtype=train_original_tensor.dtype)
    test_original_tensor = to_device_tensor(test_original_encode, device=train_original_tensor.device, dtype=train_original_tensor.dtype)
    y_train_tensor = to_device_tensor(y_train, device=train_original_tensor.device, dtype=train_original_tensor.dtype).reshape(-1)
    y_val_tensor = to_device_tensor(y_val, device=train_original_tensor.device, dtype=train_original_tensor.dtype).reshape(-1)
    y_test_tensor = to_device_tensor(y_test, device=train_original_tensor.device, dtype=train_original_tensor.dtype).reshape(-1)

    # --- Standardizer (frozen on level-0 train). ----------------------------
    if standardizer == "auto":
        standardizer = fit_standardizer(train_original_encode, data_loader)
    if standardizer is None:
        standardizer = PassthroughStandardizer()

    # --- Model factories. ---------------------------------------------------
    # Target model for utility baselines and final evaluation.
    if model_factory is None:
        model_factory = lambda: TorchLogisticRegression(
            max_iter=model_max_iter, warm_start=True, device=device, dtype=dtype)
    downstream_factory = model_factory

    # Estimator for first-stage LGA ranking.
    if estimator_model_factory is not None:
        estimator_factory = estimator_model_factory
    else:
        estimator_factory = downstream_factory

    # For second-stage candidate certification. Fall back to the downstream
    # model factory when no proxy is configured.
    proxy_factory = proxy_model_factory if proxy_model_factory is not None else downstream_factory

    def _model_family(factory_or_model):
        family = getattr(factory_or_model, "model_family", None)
        if family is not None:
            return family
        if not callable(factory_or_model) and factory_or_model is not None:
            if factory_or_model.__class__.__name__ == "XGBoostGPUClassifier":
                return "xgboost"
        return None

    # XGBoost proxy and downstream models use leaf-space input encodings.
    proxy_uses_xgboost_input = _model_family(proxy_factory) == "xgboost"
    downstream_uses_xgboost_input = _model_family(downstream_factory) == "xgboost"

    def _prepare_model_for_task(model):
        if hasattr(model, "set_classes"):
            model.set_classes(classification_classes)
        return model

    def _build_model(factory_or_model):
        model = factory_or_model() if callable(factory_or_model) else factory_or_model
        return _prepare_model_for_task(model)

    def _model_role_encode(use_xgboost_input, generalization, X_raw_subset, default_encode):
        if use_xgboost_input:
            return _xgboost_model_input(generalization, X_raw_subset)
        return default_encode

    def _standardized_level_encode(generalization, X_raw_subset):
        encode = generalization.encode(X_raw_subset)
        encode = to_device_tensor(encode, device=train_original_tensor.device, dtype=train_original_tensor.dtype)
        return standardizer.transform(encode)

    def _model_eval_encode(use_xgboost_input, generalization, X_raw_subset):
        if use_xgboost_input:
            return _xgboost_model_input(generalization, X_raw_subset)
        return _standardized_level_encode(generalization, X_raw_subset)

    def _model_loss(model, y_true_tensor, encode_for_model):
        return float(classification_log_loss_tensor(
            y_true_tensor, model.predict_proba_tensor(encode_for_model),
            classes=classification_classes, n_classes=class_count).detach().cpu())

    def _materialize_trim_release(
        snapshot_history,
        target_k,
        *,
        X_eval_raw,
        use_xgboost_input,
    ):
        """Build one mixed-level release without fitting an evaluation model."""
        import pandas as pd

        started_at = time.perf_counter()
        target_k = int(target_k)
        if target_k < 1:
            return {
                "status": "invalid_target_k",
                "target_k": target_k,
                "row_count": 0,
                "suppressed_row_count": int(len(X_train_raw)),
                "coverage_fraction": 0.0,
                "materialization_seconds": float(time.perf_counter() - started_at),
            }

        ordered_snapshots = sorted(
            snapshot_history,
            key=lambda snapshot: int(snapshot["iteration"]),
            reverse=True,
        )
        if not ordered_snapshots:
            raise ValueError("A TRIM release requires at least one generalization snapshot.")

        resolved_snapshots = []
        for snapshot in ordered_snapshots:
            generalization = snapshot.get("_generalization")
            if generalization is None:
                generalization = dataloader_module.load_generalization_rules_from_file(
                    file_path=generalization_tree_path,
                    data_loader=data_loader,
                    generalization_level=snapshot["generalization_level"],
                )
            resolved_snapshots.append((snapshot, generalization))

        remaining_ids = list(X_train_raw.index)
        privacy_parts = []
        train_model_parts = []
        accepted_groups = {}
        assignment_counts = {}
        assigned_snapshot_by_row_id = {}
        for snapshot, generalization in resolved_snapshots:
            iteration = int(snapshot["iteration"])
            if not remaining_ids:
                assignment_counts[iteration] = 0
                continue
            remaining_model_encode = generalization.encode(
                X_train_raw.loc[remaining_ids]
            )
            remaining_privacy_encode = select_encoded_attributes(
                remaining_model_encode,
                qi_attributes,
            )
            group_sizes = remaining_privacy_encode.groupby(
                list(remaining_privacy_encode.columns),
                sort=False,
                dropna=False,
                observed=True,
            ).transform("size")
            safe_ids = remaining_privacy_encode.index[
                group_sizes >= target_k
            ].tolist()
            assignment_counts[iteration] = len(safe_ids)
            if not safe_ids:
                continue

            safe_encode = remaining_privacy_encode.loc[safe_ids]
            privacy_parts.append(safe_encode)
            accepted_groups[iteration] = safe_encode.drop_duplicates().reset_index(
                drop=True
            )
            train_model_parts.append((
                safe_ids,
                _model_eval_encode(
                    use_xgboost_input,
                    generalization,
                    X_train_raw.loc[safe_ids],
                ),
            ))
            for row_id in safe_ids:
                assigned_snapshot_by_row_id[row_id] = iteration
            safe_id_set = set(safe_ids)
            remaining_ids = [
                row_id for row_id in remaining_ids if row_id not in safe_id_set
            ]

        if not privacy_parts:
            return {
                "status": "no_safe_groups",
                "target_k": target_k,
                "row_count": 0,
                "unique_row_count": 0,
                "suppressed_row_count": int(len(X_train_raw)),
                "coverage_fraction": 0.0,
                "full_release": False,
                "assignment_counts": assignment_counts,
                "assigned_row_ids": [],
                "suppressed_row_ids": [str(row_id) for row_id in remaining_ids],
                "materialization_seconds": float(time.perf_counter() - started_at),
            }

        assigned_id_set = set(X_train_raw.index) - set(remaining_ids)
        assigned_order = [
            row_id for row_id in X_train_raw.index if row_id in assigned_id_set
        ]
        release_privacy_encode = pd.concat(privacy_parts, axis=0).loc[
            assigned_order
        ]
        release_group_sizes = [
            int(value)
            for value in release_privacy_encode.value_counts(sort=False).values
        ]
        release_leak_stats = _leak_distribution_stats(release_privacy_encode)
        tail_risk_stats = _individual_tail_risk_stats(
            original_per_record_k,
            release_privacy_encode,
        )
        release_train_encode = _concat_release_model_parts(
            train_model_parts,
            assigned_order,
        )
        release_train_positions = [
            row_id_to_position[row_id] for row_id in assigned_order
        ]
        release_train_y = y_train_tensor[release_train_positions]

        remaining_eval_ids = list(X_eval_raw.index)
        eval_model_parts = []
        eval_assignment_counts = {}
        for snapshot, generalization in resolved_snapshots:
            iteration = int(snapshot["iteration"])
            groups = accepted_groups.get(iteration)
            if not remaining_eval_ids or groups is None or groups.empty:
                eval_assignment_counts[iteration] = 0
                continue
            candidate_model_encode = generalization.encode(
                X_eval_raw.loc[remaining_eval_ids]
            )
            candidate_privacy_encode = select_encoded_attributes(
                candidate_model_encode,
                qi_attributes,
            )
            candidate_keys = pd.MultiIndex.from_frame(candidate_privacy_encode)
            accepted_keys = pd.MultiIndex.from_frame(groups)
            matched_ids = candidate_privacy_encode.index[
                candidate_keys.isin(accepted_keys)
            ].tolist()
            eval_assignment_counts[iteration] = len(matched_ids)
            if not matched_ids:
                continue
            eval_model_parts.append((
                matched_ids,
                _model_eval_encode(
                    use_xgboost_input,
                    generalization,
                    X_eval_raw.loc[matched_ids],
                ),
            ))
            matched_id_set = set(matched_ids)
            remaining_eval_ids = [
                row_id
                for row_id in remaining_eval_ids
                if row_id not in matched_id_set
            ]

        coarsest_eval_row_count = len(remaining_eval_ids)
        if remaining_eval_ids:
            _snapshot, coarsest_generalization = resolved_snapshots[-1]
            eval_model_parts.append((
                list(remaining_eval_ids),
                _model_eval_encode(
                    use_xgboost_input,
                    coarsest_generalization,
                    X_eval_raw.loc[remaining_eval_ids],
                ),
            ))

        release_eval_encode = _concat_release_model_parts(
            eval_model_parts,
            list(X_eval_raw.index),
        )
        return {
            "status": "ok",
            "target_k": target_k,
            "min_k": int(release_leak_stats["leak_k"]),
            "leak_k_p1": release_leak_stats["leak_k_p1"],
            "leak_k_p2": release_leak_stats["leak_k_p2"],
            "leak_k_p3": release_leak_stats["leak_k_p3"],
            "leak_k_p4": release_leak_stats["leak_k_p4"],
            "leak_k_p5": release_leak_stats["leak_k_p5"],
            "leak_k_sorted": release_leak_stats["leak_k_sorted"],
            "leak_k_95_percentile": release_leak_stats[
                "leak_k_95_percentile"
            ],
            **tail_risk_stats,
            "privacy_constraint_met": (
                int(release_leak_stats["leak_k"]) >= target_k
            ),
            "row_count": int(len(assigned_order)),
            "unique_row_count": int(len(assigned_order)),
            "suppressed_row_count": int(len(remaining_ids)),
            "coverage_fraction": float(len(assigned_order) / len(X_train_raw)),
            "full_release": not remaining_ids,
            "assignment_counts": assignment_counts,
            "eval_assignment_counts": eval_assignment_counts,
            "eval_coarsest_release_row_count": int(coarsest_eval_row_count),
            "assigned_row_ids": [str(row_id) for row_id in assigned_order],
            "suppressed_row_ids": [str(row_id) for row_id in remaining_ids],
            "row_snapshot_assignments": {
                str(row_id): int(assigned_snapshot_by_row_id[row_id])
                for row_id in assigned_order
            },
            "materialization_seconds": float(time.perf_counter() - started_at),
            "_assigned_row_ids": assigned_order,
            "_suppressed_row_ids": list(remaining_ids),
            "_train_encode": release_train_encode,
            "_train_y": release_train_y,
            "_eval_encode": release_eval_encode,
            "_group_sizes": release_group_sizes,
        }

    def _evaluate_trim_release(
        snapshot_history,
        target_k,
        *,
        model_factory_for_release,
        use_xgboost_input,
        X_eval_raw,
        y_eval_tensor,
        eval_split,
        baseline_loss,
        loss_threshold,
    ):
        started_at = time.perf_counter()
        result = _materialize_trim_release(
            snapshot_history,
            target_k,
            X_eval_raw=X_eval_raw,
            use_xgboost_input=use_xgboost_input,
        )
        if result["status"] != "ok":
            result["evaluation_seconds"] = float(
                time.perf_counter() - started_at
            )
            return result

        release_model = _build_model(model_factory_for_release)
        release_model.fit(result["_train_encode"], result["_train_y"])
        release_loss = _model_loss(
            release_model,
            y_eval_tensor,
            result["_eval_encode"],
        )
        # Retain the fitted proxy for optional reference-model filtering in the
        # next iteration. Internal fields are excluded from persisted release
        # records by _public_release_result.
        result["_model"] = release_model
        result[f"{eval_split}_loss"] = float(release_loss)
        result["delta_u"] = float(release_loss - baseline_loss)
        result["delta_u_semantics"] = (
            f"{loss_metric_name}_minus_full_level0_{eval_split}_baseline"
        )
        result["utility_constraint_met"] = bool(
            release_loss <= loss_threshold
        )
        result["evaluation_seconds"] = float(time.perf_counter() - started_at)
        return result

    def _public_release_result(result):
        return {
            key: value
            for key, value in result.items()
            if not key.startswith("_")
        }

    # --- Baselines: downstream on full level-0 train. -----------------------
    threshold_model = _build_model(downstream_factory)
    threshold_train_encode = _model_role_encode(
        downstream_uses_xgboost_input, original_generalization, X_train_raw,
        standardizer.transform(train_original_tensor))
    threshold_val_encode = _model_role_encode(
        downstream_uses_xgboost_input, original_generalization, X_val_raw,
        standardizer.transform(val_original_tensor))
    threshold_test_encode = _model_role_encode(
        downstream_uses_xgboost_input, original_generalization, X_test_raw,
        standardizer.transform(test_original_tensor))
    threshold_model.fit(threshold_train_encode, y_train_tensor)
    baseline_val_loss = _model_loss(
        threshold_model, y_val_tensor, threshold_val_encode)
    baseline_test_loss = _model_loss(threshold_model, y_test_tensor, threshold_test_encode)
    loss_threshold_val = baseline_val_loss + tolerance
    loss_threshold_test = baseline_test_loss + tolerance

    candidate_state_scorer = None
    candidate_state_scorer_setup_seconds = 0.0
    if candidate_state_scorer_factory is not None:
        scorer_setup_started_at = time.perf_counter()
        candidate_state_scorer = candidate_state_scorer_factory(
            CandidateScorerRunContext(
                train_original_encode=standardizer.transform(
                    train_original_tensor
                ),
                train_y=y_train_tensor,
            )
        )
        candidate_state_scorer_setup_seconds = float(
            time.perf_counter() - scorer_setup_started_at
        )
        if not callable(candidate_state_scorer):
            raise TypeError(
                "candidate_state_scorer_factory must return a callable scorer."
            )

    # --- S0 or fixed-plan bootstrap. ---------------------------------------
    train_row_ids = list(train_original_encode.index)
    if fixed_row_candidate_plan is not None:
        initial_row_ids = list(fixed_row_candidate_plan.initial_row_ids)
        initial_sample_size = len(initial_row_ids)
    elif initial_row_ids is None:
        if initial_sample_fraction is not None:
            initial_sample_size = max(
                1,
                int(len(train_row_ids) * initial_sample_fraction),
            )
        elif initial_sample_size is None:
            initial_sample_size = 1
        else:
            initial_sample_size = int(initial_sample_size)
        if initial_sample_size < 1:
            raise ValueError("initial_sample_size must be at least 1.")
        if initial_sample_size > len(train_row_ids):
            raise ValueError("initial_sample_size cannot exceed the training row count.")
        initial_row_ids = stratified_sample_row_ids(
            train_row_ids,
            y_train,
            initial_sample_size,
            random_state,
        )
    else:
        missing = [rid for rid in initial_row_ids if rid not in train_original_encode.index]
        if missing:
            raise KeyError(f"initial_row_ids are not in the training rows: {missing}")
        initial_row_ids = list(initial_row_ids)
        initial_sample_size = len(initial_row_ids)
        validate_label_coverage(train_row_ids, y_train, initial_row_ids)
    if not initial_row_ids:
        raise ValueError("initial_row_ids must contain at least one row.")
    initial_row_id_set = (
        set(fixed_row_candidate_plan.privacy_exempt_row_ids)
        if fixed_row_candidate_plan is not None
        else set(initial_row_ids)
    )

    current_train_encode = initial_generalization_train_encode.copy()
    current_train_tensor = to_device_tensor(
        current_train_encode, device=train_original_tensor.device, dtype=train_original_tensor.dtype)
    row_id_to_position = {rid: pos for pos, rid in enumerate(current_train_encode.index)}

    # Standard initialization treats S0 as privacy-exempt. The all-rows ablation
    # keeps that exemption set while initializing the selected state with every
    # row at the same maximum-generalization snapshot.
    initialization_row_ids = (
        list(fixed_row_candidate_plan.initial_row_ids)
        if fixed_row_candidate_plan is not None
        else (
            list(train_row_ids)
            if ablation_actions.initialization == ALL_ROWS_AT_MAX_GENERALIZATION
            else list(initial_row_ids)
        )
    )
    initialization_row_id_set = set(initialization_row_ids)

    # Fit the estimator and proxy on the initial sample. Candidate certification
    # then updates the validation-loss signal after each selected release; the
    # downstream model is reserved for final utility evaluation.
    initial_positions = [
        row_id_to_position[rid] for rid in initialization_row_ids
    ]
    initial_fit_encode = standardizer.transform(current_train_tensor[initial_positions])
    initial_fit_y = y_train_tensor[initial_positions]

    backend_model = _build_model(estimator_factory)
    if hasattr(backend_model, "warm_start"):
        backend_model.warm_start = True
    backend_model.fit(initial_fit_encode, initial_fit_y)

    proxy_model = _build_model(proxy_factory)
    initial_proxy_fit_encode = _model_role_encode(
        proxy_uses_xgboost_input, current_generalization,
        X_train_raw.loc[initialization_row_ids], initial_fit_encode)
    proxy_val_encode = _model_eval_encode(
        proxy_uses_xgboost_input, current_generalization, X_val_raw)
    proxy_model.fit(initial_proxy_fit_encode, initial_fit_y)
    proxy_model_loss = _model_loss(proxy_model, y_val_tensor, proxy_val_encode)

    # --- Running timing totals. ---------------------------------------------
    total_enumeration_time = 0.0
    total_selection_time = 0.0
    total_privacy_accounting_time = 0.0
    total_retraining_time = 0.0
    total_candidate_observer_time = 0.0
    total_trim_iteration_observer_time = 0.0

    # --- Initial retention classes (max level, minus selected rows). --------
    initial_enumeration_started_at = time.perf_counter()
    if RETENTION_CLASS_ACTION not in enabled_action_types:
        remaining_retention_classes = []
    elif fixed_row_candidate_plan is not None:
        remaining_retention_classes = []
        selected_initial_set = set(initialization_row_ids)
        for group_key, group_row_ids in fixed_row_candidate_plan.candidate_groups:
            remaining_index = [
                row_id
                for row_id in group_row_ids
                if row_id not in selected_initial_set
            ]
            if remaining_index:
                remaining_retention_classes.append(
                    (group_key, current_train_encode.loc[remaining_index].copy())
                )
    elif _can_use_position_index_leakage(swap_encode_cache):
        initial_position_index = _swap_position_index_from_cache(swap_encode_cache)
        remaining_retention_classes = (
            _retention_class_index_groups_from_position_index(
                initial_position_index,
                excluded_row_ids=initialization_row_id_set,
            )
        )
    else:
        retention_classes = build_retention_class_groups(
            initial_generalization_train_encode,
            selected_attributes=selected_attributes)
        remaining_retention_classes = []
        for rc_key, rc in retention_classes.items():
            remaining_index = [
                rid for rid in rc.index
                if rid not in initialization_row_id_set
            ]
            if remaining_index:
                remaining_retention_classes.append(
                    (rc_key, rc.loc[remaining_index].copy()))
    initial_retention_class_build_time = (
        fixed_row_candidate_build_time
        + time.perf_counter()
        - initial_enumeration_started_at
    )
    next_iteration_enumeration_time = 0.0

    # --- Run directory + action log. ---------------------------------------
    run_dir, action_log_path = _open_run_dir(results_dir, run_tag)
    if action_log_path is not None:
        with Path(action_log_path).open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps({
                "event": "pipeline_original_dataset",
                "original_row_count": len(original_dataset_encode),
                "original_leak_k": original_leak_k,
                "original_leak_k_p1": original_leak_k_p1,
                "original_leak_k_p2": original_leak_k_p2,
                "original_leak_k_p3": original_leak_k_p3,
                "original_leak_k_p4": original_leak_k_p4,
                "original_leak_k_p5": original_leak_k_p5,
            }, default=str, sort_keys=True))
            log_file.write("\n")

    state = SelectionState(
        predicted_model_loss=proxy_model_loss,
        actual_model_loss=proxy_model_loss,
        is_actual_model_loss_trusted=True,
        current_generalization=current_generalization,
        selected_row_ids=list(initialization_row_ids),
        current_train_encode=current_train_encode,
        current_train_tensor=current_train_tensor,
        action_log_path=str(action_log_path) if action_log_path is not None else None,
    )

    selected_by_action_row_ids = {
        row_id
        for row_id in initialization_row_ids
        if row_id not in initial_row_id_set
    }
    selected_retention_class_keys = []
    iteration_count = 0
    termination_condition = None
    release_snapshot_history = [{
        "iteration": -1,
        "generalization_level": dict(
            current_generalization.generalization_level
        ),
        "_generalization": current_generalization,
    }]
    trim_iteration_results = []

    def _evaluate_candidate_release(candidate_generalization, candidate_min_k):
        target_k = (
            int(release_target_k)
            if release_target_k is not None
            else int(candidate_min_k)
        )
        candidate_history = [
            *release_snapshot_history,
            {
                "iteration": int(iteration_count),
                "generalization_level": dict(
                    candidate_generalization.generalization_level
                ),
                "_generalization": candidate_generalization,
            },
        ]
        return _evaluate_trim_release(
            candidate_history,
            target_k,
            model_factory_for_release=proxy_factory,
            use_xgboost_input=proxy_uses_xgboost_input,
            X_eval_raw=X_val_raw,
            y_eval_tensor=y_val_tensor,
            eval_split="validation",
            baseline_loss=baseline_val_loss,
            loss_threshold=loss_threshold_val,
        )

    def pipeline_leakage(encode):
        # S0 is privacy-exempt by default. A fixed plan may declare no exempt
        # rows in its published initialization.
        nonlocal total_privacy_accounting_time
        t0 = time.perf_counter()
        try:
            privacy_encode = encode.drop(
                index=list(initial_row_id_set),
                errors="ignore",
            )
            privacy_encode = select_encoded_attributes(
                privacy_encode,
                qi_attributes,
            )
            if privacy_encode.empty:
                return 0
            return calculate_leakage(privacy_encode)
        finally:
            total_privacy_accounting_time += time.perf_counter() - t0

    # =========================================================================
    # Greedy loop.
    # =========================================================================
    while (
        not trim_iteration_results
        or not stop_on_utility
        or state.actual_model_loss > loss_threshold_val
    ):
        if max_iterations is not None and iteration_count >= max_iterations:
            termination_condition = "max_iterations_reached"
            break

        iter_enumeration_time = next_iteration_enumeration_time
        next_iteration_enumeration_time = 0.0
        iter_selection_time_start = total_selection_time
        iter_privacy_time_start = total_privacy_accounting_time
        iter_retraining_time_start = total_retraining_time
        privacy_before_selection = total_privacy_accounting_time

        # Mutable accumulators shared with _greedy_select so its LGA / proxy /
        # leakage / ranking timings fold into the running totals in place.
        selection_timing = {
            # Selection wall time excludes candidate-observer callbacks. The
            # caller separately removes privacy-accounting time below.
            "selection": 0.0,
            "reference_filter": 0.0,
            "lga": 0.0, "proxy_retrain": 0.0, "leakage": 0.0, "ranking": 0.0,
            "candidate_observer": 0.0,
        }
        selected_action = _greedy_select(
            state,
            remaining_retention_classes=remaining_retention_classes,
            X_train_raw=X_train_raw, X_val_raw=X_val_raw,
            train_original_encode=train_original_encode,
            backend_model=backend_model,
            reference_proxy_model=proxy_model,
            reference_model_filtering=reference_model_filtering,
            proxy_factory=proxy_factory,
            proxy_uses_xgboost_input=proxy_uses_xgboost_input,
            leakage_func=pipeline_leakage,
            rank_top_k=rank_top_k,
            standardizer=standardizer,
            row_id_to_position=row_id_to_position,
            train_original_tensor=train_original_tensor,
            val_original_tensor=val_original_tensor,
            train_y_tensor=y_train_tensor,
            val_y_tensor=y_val_tensor,
            classification_classes=classification_classes,
            class_count=class_count,
            use_multiclass=use_multiclass,
            selection_timing=selection_timing,
            action_log_path=action_log_path,
            iteration_count=iteration_count,
            current_train_level_encode=swap_encode_cache.encoded,
            swap_encode_cache=swap_encode_cache,
            initial_row_id_set=initial_row_id_set,
            candidate_release_evaluator=_evaluate_candidate_release,
            enabled_action_types=enabled_action_types,
            vertical_postprocessing=ablation_actions.vertical_postprocessing,
            move_out_threshold=ablation_actions.move_out_threshold,
            whole_retention_group_admission=bool(
                fixed_row_candidate_plan is not None
                and fixed_row_candidate_plan.whole_group_admission
            ),
            candidate_shortlist=candidate_shortlist,
            candidate_state_scorer=candidate_state_scorer,
            selection_score=selection_score,
            candidate_observer=candidate_observer,
        )
        # pipeline_leakage and the optimized class-id leakage path both credit
        # privacy time during selection; selection-exclusive time is the rest of
        # the selection wall clock.
        total_privacy_accounting_time += selection_timing.get("optimized_privacy", 0.0)
        privacy_inside_selection = total_privacy_accounting_time - privacy_before_selection
        total_selection_time += max(0.0, selection_timing["selection"] - privacy_inside_selection)
        total_candidate_observer_time += selection_timing.get(
            "candidate_observer", 0.0
        )

        if selected_action["action"] is None:
            termination_condition = "no_action_available"
            break

        candidates = selected_action["candidates"]
        _write_action_log(
            action_log_path=action_log_path, iteration=iteration_count,
            current_state=state, candidates=candidates,
            best_candidate=candidates[selected_action["selected_candidate_position"]])

        previous_selected = set(state.selected_row_ids)
        state = selected_action["state"]
        state.action_log_path = str(action_log_path) if action_log_path is not None else None
        current_level_encode_for_reenum = None
        snapshot_position_index = selected_action.get("position_index")

        if selected_action["action"] == "retention_class":
            rc_key = selected_action["retention_class_key"]
            if rc_key not in selected_retention_class_keys:
                selected_retention_class_keys.append(rc_key)
            admitted = set(state.selected_row_ids) - previous_selected
            selected_by_action_row_ids.update(
                rid for rid in admitted if rid not in initial_row_id_set)
        elif selected_action["action"] == "vertical_refinement":
            swap_stats = selected_action.get("swap_stats")
            current_selected = set(state.selected_row_ids)
            selected_by_action_row_ids.difference_update(
                previous_selected - current_selected)
            admitted = current_selected - previous_selected
            selected_by_action_row_ids.update(
                rid for rid in admitted if rid not in initial_row_id_set)
            swap_level_encode = selected_action.get("swap_level_encode")
            swap_codes = selected_action.get("swap_codes")
            if (
                swap_encode_cache is not None
                and swap_level_encode is not None
                and swap_codes is not None
            ):
                swap_encode_cache.encoded = swap_level_encode
                swap_encode_cache.codes = swap_codes
                current_level_encode_for_reenum = swap_encode_cache.encoded
            else:
                current_level_encode_for_reenum = _update_swap_encode_cache(
                    swap_encode_cache, state.current_generalization,
                    X_train_raw, selected_action.get("attribute"))
            if action_log_path is not None and swap_stats is not None:
                with Path(action_log_path).open("a", encoding="utf-8") as log_file:
                    log_file.write(json.dumps({
                        "event": "swap_timing", "iteration": iteration_count,
                        "action": selected_action["action"],
                        "attribute": selected_action.get("attribute"),
                        "swap_iterations": swap_stats["swap_iterations"],
                        "elapsed_seconds": swap_stats["elapsed_seconds"],
                        "encode_time_seconds": swap_stats["encode_time_seconds"],
                        "position_index_time_seconds": swap_stats["position_index_time_seconds"],
                        "loop_time_seconds": swap_stats["loop_time_seconds"],
                        "sync_time_seconds": swap_stats["sync_time_seconds"],
                        "break_reason": swap_stats["break_reason"],
                        "selected_row_count": swap_stats["selected_row_count"],
                        "candidate_row_count": swap_stats["candidate_row_count"],
                        "postprocessing_policy": swap_stats.get(
                            "postprocessing_policy"
                        ),
                        "move_out_threshold": swap_stats.get(
                            "move_out_threshold"
                        ),
                        "move_out_row_count": swap_stats.get(
                            "move_out_row_count", 0
                        ),
                        "move_in_row_count": swap_stats.get(
                            "move_in_row_count", 0
                        ),
                        "extra_swap_in_count": swap_stats.get(
                            "extra_swap_in_count", 0
                        ),
                    }, default=str, sort_keys=True))
                    log_file.write("\n")

        iteration_count += 1

        # --- Refit the estimator for the next iteration's LGA ranking. ------
        selected_positions = [row_id_to_position[rid] for rid in state.selected_row_ids]
        t0 = time.perf_counter()
        selected_fit_encode = standardizer.transform(
            state.current_train_tensor[selected_positions])
        selected_fit_y = y_train_tensor[selected_positions]

        backend_model = _build_model(estimator_factory)
        if hasattr(backend_model, "warm_start"):
            backend_model.warm_start = True
        backend_model.fit(selected_fit_encode, selected_fit_y)
        total_retraining_time += time.perf_counter() - t0

        # --- Per-iteration action log + leak snapshot. ----------------------
        iter_selection_time = total_selection_time - iter_selection_time_start
        iter_privacy_time = total_privacy_accounting_time - iter_privacy_time_start
        iter_retraining_time = total_retraining_time - iter_retraining_time_start

        if (
            snapshot_position_index is None
            and _can_use_position_index_leakage(swap_encode_cache)
        ):
            snapshot_position_index = _swap_position_index_from_cache(swap_encode_cache)
        published_row_count, leak_stats, _ = _published_leak_snapshot(
            select_encoded_attributes(
                state.current_train_encode,
                qi_attributes,
            ),
            state.selected_row_ids,
            initial_row_id_set,
            position_index=snapshot_position_index,
        )
        release_result = selected_action.get("release_result")
        if release_result is None or release_result.get("status") != "ok":
            raise RuntimeError(
                "The selected action does not have a valid TRIM release."
            )
        proxy_model = release_result.get("_model")
        if proxy_model is None:
            raise RuntimeError(
                "The selected TRIM release does not retain its fitted proxy model."
            )
        expected_release_target_k = (
            int(release_target_k)
            if release_target_k is not None
            else int(leak_stats["leak_k"])
        )
        if int(release_result["target_k"]) != expected_release_target_k:
            raise RuntimeError(
                "The selected release target disagrees with the published min-K."
            )
        snapshot_iteration = iteration_count - 1
        release_snapshot_history.append({
            "iteration": int(snapshot_iteration),
            "generalization_level": dict(
                state.current_generalization.generalization_level
            ),
            "_generalization": state.current_generalization,
        })
        proxy_model_loss = float(release_result["validation_loss"])
        state.predicted_model_loss = proxy_model_loss
        state.actual_model_loss = proxy_model_loss
        state.is_actual_model_loss_trusted = True
        trim_iteration_record = {
            "iteration": int(snapshot_iteration),
            "action": selected_action["action"],
            "attribute": selected_action.get("attribute"),
            "original_leak_k": original_leak_k,
            "original_leak_k_p1": original_leak_k_p1,
            "original_leak_k_p2": original_leak_k_p2,
            "original_leak_k_p3": original_leak_k_p3,
            "original_leak_k_p4": original_leak_k_p4,
            "original_leak_k_p5": original_leak_k_p5,
            "ordinary_min_k": int(leak_stats["leak_k"]),
            "ordinary_published_row_count": int(published_row_count),
            "iter_enumeration_time": float(iter_enumeration_time),
            "iter_selection_time": float(iter_selection_time),
            "iter_privacy_time": float(iter_privacy_time),
            "iter_retraining_time": float(iter_retraining_time),
            "snapshot_history": [
                {
                    "iteration": int(snapshot["iteration"]),
                    "generalization_level": dict(
                        snapshot["generalization_level"]
                    ),
                }
                for snapshot in release_snapshot_history
            ],
            **_public_release_result(release_result),
        }
        if record_iteration_test_metrics:
            observation_started_at = time.perf_counter()
            try:
                iteration_test_result = _evaluate_trim_release(
                    release_snapshot_history,
                    int(release_result["target_k"]),
                    model_factory_for_release=downstream_factory,
                    use_xgboost_input=downstream_uses_xgboost_input,
                    X_eval_raw=X_test_raw,
                    y_eval_tensor=y_test_tensor,
                    eval_split="test",
                    baseline_loss=baseline_test_loss,
                    loss_threshold=loss_threshold_test,
                )
            finally:
                observation_seconds = (
                    time.perf_counter() - observation_started_at
                )
                total_curve_metric_time += observation_seconds
            trim_iteration_record.update({
                "test_status": iteration_test_result.get("status"),
                "test_loss": (
                    float(iteration_test_result["test_loss"])
                    if iteration_test_result.get("status") == "ok"
                    else None
                ),
                "test_delta_u": (
                    float(iteration_test_result["delta_u"])
                    if iteration_test_result.get("status") == "ok"
                    else None
                ),
                "test_utility_constraint_met": (
                    bool(iteration_test_result["utility_constraint_met"])
                    if iteration_test_result.get("status") == "ok"
                    else None
                ),
                "test_observation_seconds": float(observation_seconds),
            })
        else:
            trim_iteration_record.update({
                "test_status": "not_recorded",
                "test_loss": None,
                "test_delta_u": None,
                "test_utility_constraint_met": None,
                "test_observation_seconds": 0.0,
            })
        trim_iteration_results.append(trim_iteration_record)

        if trim_iteration_observer is not None:
            observer_started_at = time.perf_counter()
            try:
                trim_iteration_observer(TRIMIterationObservation(
                    iteration=int(snapshot_iteration),
                    action=selected_action["action"],
                    attribute=selected_action.get("attribute"),
                    selected_row_count=len(state.selected_row_ids),
                    ordinary_min_k=int(leak_stats["leak_k"]),
                    ordinary_published_row_count=int(published_row_count),
                    generalization_level=MappingProxyType(dict(
                        state.current_generalization.generalization_level
                    )),
                    release_target_k=int(release_result["target_k"]),
                    release_validation_loss=proxy_model_loss,
                    release_row_count=int(release_result["row_count"]),
                    release_suppressed_row_count=int(
                        release_result["suppressed_row_count"]
                    ),
                    release_coverage_fraction=float(
                        release_result["coverage_fraction"]
                    ),
                    release_utility_constraint_met=bool(
                        release_result["utility_constraint_met"]
                    ),
                ))
            finally:
                total_trim_iteration_observer_time += (
                    time.perf_counter() - observer_started_at
                )

        if action_log_path is not None:
            with Path(action_log_path).open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps({
                    "event": "pipeline_iteration_state",
                    "iteration": iteration_count - 1,
                    "action": selected_action["action"],
                    "attribute": selected_action.get("attribute"),
                    "selected_row_count": len(state.selected_row_ids),
                    "published_row_count": published_row_count,
                    "generalization_level": dict(state.current_generalization.generalization_level),
                    "leak_k": leak_stats["leak_k"],
                    "leak_k_p1": leak_stats["leak_k_p1"],
                    "leak_k_p2": leak_stats["leak_k_p2"],
                    "leak_k_p3": leak_stats["leak_k_p3"],
                    "leak_k_p4": leak_stats["leak_k_p4"],
                    "leak_k_p5": leak_stats["leak_k_p5"],
                    "loss_metric": loss_metric_name,
                    "proxy_val_loss": proxy_model_loss,
                    "release_validation_loss": proxy_model_loss,
                    "release_target_k": release_result["target_k"],
                    "release_row_count": release_result["row_count"],
                    "release_suppressed_row_count": release_result[
                        "suppressed_row_count"
                    ],
                    "release_coverage_fraction": release_result[
                        "coverage_fraction"
                    ],
                    "release_utility_constraint_met": release_result[
                        "utility_constraint_met"
                    ],
                    "iter_selection_time": iter_selection_time,
                    "iter_enumeration_time": iter_enumeration_time,
                    "iter_privacy_time": iter_privacy_time,
                    "iter_retraining_time": iter_retraining_time,
                }, default=str, sort_keys=True))
                log_file.write("\n")

        # --- Re-enumerate retention classes for the next iteration. ---------
        selected_set = set(state.selected_row_ids)
        remaining_row_ids = [rid for rid in train_original_encode.index if rid not in selected_set]
        if RETENTION_CLASS_ACTION not in enabled_action_types:
            remaining_retention_classes = []
        elif fixed_row_candidate_plan is not None:
            t0 = time.perf_counter()
            remaining_retention_classes = []
            for group_key, group_row_ids in fixed_row_candidate_plan.candidate_groups:
                remaining_index = [
                    row_id
                    for row_id in group_row_ids
                    if row_id not in selected_set
                ]
                if remaining_index:
                    remaining_retention_classes.append(
                        (
                            group_key,
                            state.current_train_encode.loc[
                                remaining_index
                            ].copy(),
                        )
                    )
            next_iteration_enumeration_time = time.perf_counter() - t0
            total_enumeration_time += next_iteration_enumeration_time
        elif remaining_row_ids:
            t0 = time.perf_counter()
            if _can_use_position_index_leakage(swap_encode_cache):
                reenum_position_index = snapshot_position_index
                if reenum_position_index is None:
                    reenum_position_index = _swap_position_index_from_cache(
                        swap_encode_cache)
                remaining_retention_classes = (
                    _retention_class_index_groups_from_position_index(
                        reenum_position_index,
                        excluded_row_ids=selected_set,
                    )
                )
            else:
                if current_level_encode_for_reenum is not None:
                    current_level_encode = current_level_encode_for_reenum
                elif swap_encode_cache is not None:
                    current_level_encode = swap_encode_cache.encoded
                else:
                    current_level_encode = state.current_generalization.encode(
                        X_train_raw)
                retention_classes = build_retention_class_groups(
                    current_level_encode.loc[remaining_row_ids],
                    selected_attributes=selected_attributes)
                remaining_retention_classes = list(retention_classes.items())
            next_iteration_enumeration_time = time.perf_counter() - t0
            total_enumeration_time += next_iteration_enumeration_time
        else:
            remaining_retention_classes = []

    if termination_condition is None:
        termination_condition = "loss_threshold_met"

    # =========================================================================
    # Final closure + verification.
    # =========================================================================
    final_level_encode = swap_encode_cache.encoded
    if (
        fixed_row_candidate_plan is not None
        and not fixed_row_candidate_plan.close_remaining_groups
    ):
        final_retention_class_rows = []
    elif _can_use_position_index_leakage(swap_encode_cache):
        final_position_index = _swap_position_index_from_cache(swap_encode_cache)
        final_retention_class_rows = [
            row_ids
            for _, row_ids in _retention_class_index_groups_from_position_index(
                final_position_index,
            )
        ]
    else:
        final_retention_classes = build_retention_class_groups(
            final_level_encode, selected_attributes=selected_attributes)
        final_retention_class_rows = [
            list(final_rc.index)
            for final_rc in final_retention_classes.values()
        ]

    # Admit cap-leftover rows of every live final-level retention class.
    live_published = set(selected_by_action_row_ids)
    current_selected_set = set(state.selected_row_ids)
    newly_admitted = []
    for rc_row_ids in final_retention_class_rows:
        if not any(rid in live_published for rid in rc_row_ids):
            continue
        for rid in rc_row_ids:
            if rid not in current_selected_set:
                newly_admitted.append(rid)
                current_selected_set.add(rid)
    if newly_admitted:
        state.selected_row_ids = list(state.selected_row_ids) + newly_admitted
        selected_by_action_row_ids.update(
            rid for rid in newly_admitted if rid not in initial_row_id_set)

    # S0-retention merge: keep S0 rows whose final class is fully published.
    retained_initial_row_ids = set()
    for rc_row_ids in final_retention_class_rows:
        rc_initial = [rid for rid in rc_row_ids if rid in initial_row_id_set]
        if not rc_initial:
            continue
        rc_non_initial = [rid for rid in rc_row_ids if rid not in initial_row_id_set]
        if rc_non_initial and all(rid in selected_by_action_row_ids for rid in rc_non_initial):
            retained_initial_row_ids.update(rc_initial)

    ordinary_final_selected_row_ids = []
    for rid in state.selected_row_ids:
        if rid in initial_row_id_set and rid not in retained_initial_row_ids:
            continue
        if rid not in ordinary_final_selected_row_ids:
            ordinary_final_selected_row_ids.append(rid)
    state.selected_row_ids = ordinary_final_selected_row_ids

    # Final downstream evaluation uses the materialized mixed-level release.
    final_verification_time = 0.0
    t0 = time.perf_counter()
    if trim_iteration_results:
        final_release_target_k = int(
            trim_iteration_results[-1]["target_k"]
        )
        final_release_result = _evaluate_trim_release(
            release_snapshot_history,
            final_release_target_k,
            model_factory_for_release=downstream_factory,
            use_xgboost_input=downstream_uses_xgboost_input,
            X_eval_raw=X_test_raw,
            y_eval_tensor=y_test_tensor,
            eval_split="test",
            baseline_loss=baseline_test_loss,
            loss_threshold=loss_threshold_test,
        )
    else:
        final_release_target_k = None
        final_release_result = {
            "status": "no_published_iteration",
            "row_count": 0,
            "suppressed_row_count": int(len(X_train_raw)),
            "coverage_fraction": 0.0,
            "assigned_row_ids": [],
            "suppressed_row_ids": [
                str(row_id) for row_id in X_train_raw.index
            ],
        }
    final_verification_time += time.perf_counter() - t0
    if final_release_result.get("status") == "ok":
        final_actual_model_loss = float(final_release_result["test_loss"])
        selected_row_ids = list(final_release_result["_assigned_row_ids"])
        release_suppressed_row_ids = list(
            final_release_result["_suppressed_row_ids"]
        )
        final_group_sizes = list(final_release_result["_group_sizes"])
    else:
        final_actual_model_loss = float("inf")
        selected_row_ids = []
        release_suppressed_row_ids = list(X_train_raw.index)
        final_group_sizes = []
    utility_constraint_met = final_actual_model_loss <= loss_threshold_test
    validation_utility_constraint_met = bool(
        trim_iteration_results
        and state.actual_model_loss <= loss_threshold_val
    )

    # Final privacy metrics describe the materialized release, not the internal
    # selected-state encoding.
    final_leak_k = int(final_release_result.get("min_k", 0))
    leak_k_p1 = float(final_release_result.get("leak_k_p1", 0.0))
    leak_k_p2 = float(final_release_result.get("leak_k_p2", 0.0))
    leak_k_p3 = float(final_release_result.get("leak_k_p3", 0.0))
    leak_k_p4 = float(final_release_result.get("leak_k_p4", 0.0))
    leak_k_p5 = float(final_release_result.get("leak_k_p5", 0.0))
    tail_risk_p99 = final_release_result.get("tail_risk_p99")
    if tail_risk_p99 is not None:
        tail_risk_p99 = float(tail_risk_p99)
    final_leak_k_sorted = list(
        final_release_result.get("leak_k_sorted", [])
    )
    final_leak_k_95_percentile = final_release_result.get(
        "leak_k_95_percentile"
    )
    public_release_snapshot_history = [
        {
            "iteration": int(snapshot["iteration"]),
            "generalization_level": dict(snapshot["generalization_level"]),
        }
        for snapshot in release_snapshot_history
    ]
    final_release_public = _public_release_result(final_release_result)
    final_release_public.update({
        "original_leak_k": original_leak_k,
        "original_leak_k_p1": original_leak_k_p1,
        "original_leak_k_p2": original_leak_k_p2,
        "original_leak_k_p3": original_leak_k_p3,
        "original_leak_k_p4": original_leak_k_p4,
        "original_leak_k_p5": original_leak_k_p5,
        "validation_loss": (
            float(state.actual_model_loss)
            if trim_iteration_results
            else None
        ),
        "validation_utility_constraint_met": (
            validation_utility_constraint_met
        ),
        "snapshot_history": public_release_snapshot_history,
    })

    timings = {
        "enumeration_time_seconds": total_enumeration_time,
        "initial_retention_class_build_time_seconds": (
            initial_retention_class_build_time
        ),
        "candidate_enumeration_total_time_seconds": (
            initial_retention_class_build_time + total_enumeration_time
        ),
        "selection_time_seconds": total_selection_time,
        "privacy_accounting_time_seconds": total_privacy_accounting_time,
        "backend_retraining_certification_time_seconds": total_retraining_time,
        "final_verification_time_seconds": final_verification_time,
        "curve_metric_time_seconds": total_curve_metric_time,
        "experiment_observation_time_seconds": total_curve_metric_time,
        "candidate_observer_time_seconds": total_candidate_observer_time,
        "candidate_state_scorer_setup_time_seconds": (
            candidate_state_scorer_setup_seconds
        ),
        "trim_iteration_observer_time_seconds": (
            total_trim_iteration_observer_time
        ),
        "pipeline_wall_time_seconds": time.perf_counter() - pipeline_start_time,
    }
    if action_log_path is not None:
        with Path(action_log_path).open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps({
                "event": "pipeline_summary",
                "task_type": "classification",
                "loss_metric": loss_metric_name,
                "termination_condition": termination_condition,
                "selected_row_count": len(selected_row_ids),
                "ordinary_selected_row_count": len(
                    ordinary_final_selected_row_ids
                ),
                "initial_row_count": len(initial_row_ids),
                "initialization_row_count": len(initialization_row_ids),
                "ablation_actions": ablation_actions.as_dict(),
                "fixed_row_candidate_plan": (
                    fixed_row_candidate_plan.as_dict()
                    if fixed_row_candidate_plan is not None
                    else None
                ),
                "stop_on_utility": stop_on_utility,
                "reference_model_filtering": reference_model_filtering,
                "record_iteration_test_metrics": (
                    record_iteration_test_metrics
                ),
                "generalization_level": dict(state.current_generalization.generalization_level),
                "baseline_val_loss": baseline_val_loss,
                "loss_threshold_val": loss_threshold_val,
                "baseline_test_loss": baseline_test_loss,
                "loss_threshold_test": loss_threshold_test,
                "final_actual_model_loss": final_actual_model_loss,
                "utility_constraint_met": utility_constraint_met,
                "validation_utility_constraint_met": (
                    validation_utility_constraint_met
                ),
                "release_target_k": final_release_target_k,
                "release_row_count": len(selected_row_ids),
                "release_suppressed_row_count": len(
                    release_suppressed_row_ids
                ),
                "original_leak_k": original_leak_k,
                "original_leak_k_p1": original_leak_k_p1,
                "original_leak_k_p2": original_leak_k_p2,
                "original_leak_k_p3": original_leak_k_p3,
                "original_leak_k_p4": original_leak_k_p4,
                "original_leak_k_p5": original_leak_k_p5,
                "final_leak_k": final_leak_k,
                "final_leak_k_p1": leak_k_p1,
                "final_leak_k_p2": leak_k_p2,
                "final_leak_k_p3": leak_k_p3,
                "final_leak_k_p4": leak_k_p4,
                "final_leak_k_p5": leak_k_p5,
                "tail_risk_p99": tail_risk_p99,
                "tail_risk_semantics": final_release_result.get("tail_risk_semantics"),
                "tail_risk_population": final_release_result.get("tail_risk_population"),
                "tail_risk_population_size": final_release_result.get("tail_risk_population_size"),
                "tail_risk_percentile_method": final_release_result.get("tail_risk_percentile_method"),
                "final_leak_k_95_percentile": final_leak_k_95_percentile,
                "timings": timings,
            }, default=str, sort_keys=True))
            log_file.write("\n")

    if run_dir is not None:
        from collections import Counter
        if final_group_sizes:
            leak_distribution = {
                str(k): c for k, c in sorted(Counter(final_group_sizes).items())
            }
        else:
            leak_distribution = {}
        _dump_json({
            "generalization_tree_path": str(tree_artifact_path),
            "generalization_tree_sha256": generalization_tree_sha256,
            "input_data_sha256": input_data_sha256,
            "effective_loaded_row_count": len(X_raw),
            "preprocessing": getattr(data_loader, "preprocessing_metadata", {}),
            "nrows": nrows, "val_size": val_size, "test_size": test_size,
            "random_state": random_state, "tolerance": tolerance,
            "rank_top_k": rank_top_k,
            "max_iterations": max_iterations, "model_max_iter": model_max_iter,
            "model": model_config, "initial_sample_size": initial_sample_size,
            "initial_sample_fraction": initial_sample_fraction,
            "device": device, "dtype": dtype, "run_tag": run_tag,
            "task_type": "classification", "loss_metric": loss_metric_name,
            "ablation_actions": ablation_actions.as_dict(),
            "fixed_row_candidate_builder": _callable_name(
                fixed_row_candidate_builder
            ),
            "fixed_row_candidate_plan": (
                fixed_row_candidate_plan.as_dict()
                if fixed_row_candidate_plan is not None
                else None
            ),
            "candidate_shortlist": _callable_name(candidate_shortlist),
            "candidate_state_scorer_factory": _callable_name(
                candidate_state_scorer_factory
            ),
            "selection_score": _callable_name(selection_score),
            "candidate_observer": _callable_name(candidate_observer),
            "stop_on_utility": stop_on_utility,
            "reference_model_filtering": reference_model_filtering,
            "record_iteration_test_metrics": record_iteration_test_metrics,
            "trim_iteration_observer": _callable_name(
                trim_iteration_observer
            ),
            "release_target_k": (
                None if release_target_k is None else int(release_target_k)
            ),
            "release_target_semantics": (
                "candidate_ordinary_min_k"
                if release_target_k is None
                else "fixed_k"
            ),
            "search_utility_split": "validation",
            "final_utility_split": "test",
        }, run_dir / "config.json")
        _dump_json({
            "termination_condition": termination_condition,
            "iteration_count": iteration_count,
            "selected_row_count": len(selected_row_ids),
            "ordinary_selected_row_count": len(
                ordinary_final_selected_row_ids
            ),
            "initial_row_count": len(initial_row_ids),
            "initialization_row_count": len(initialization_row_ids),
            "baseline_val_loss": baseline_val_loss,
            "loss_threshold_val": loss_threshold_val,
            "baseline_test_loss": baseline_test_loss,
            "loss_threshold_test": loss_threshold_test,
            "final_actual_model_loss": final_actual_model_loss,
            "utility_constraint_met": utility_constraint_met,
            "validation_utility_constraint_met": (
                validation_utility_constraint_met
            ),
            "release_target_k": final_release_target_k,
            "release_suppressed_row_count": len(
                release_suppressed_row_ids
            ),
            "original_leak_k": original_leak_k,
            "original_leak_k_p1": original_leak_k_p1,
            "original_leak_k_p2": original_leak_k_p2,
            "original_leak_k_p3": original_leak_k_p3,
            "original_leak_k_p4": original_leak_k_p4,
            "original_leak_k_p5": original_leak_k_p5,
            "final_leak_k": final_leak_k,
            "final_leak_k_p5": leak_k_p5,
            "tail_risk_p99": tail_risk_p99,
            "tail_risk_semantics": final_release_result.get("tail_risk_semantics"),
            "tail_risk_population": final_release_result.get("tail_risk_population"),
            "tail_risk_population_size": final_release_result.get("tail_risk_population_size"),
            "tail_risk_percentile_method": final_release_result.get("tail_risk_percentile_method"),
            "final_leak_k_95_percentile": final_leak_k_95_percentile,
            "timings": timings,
        }, run_dir / "metrics.json")
        _dump_json({"selected_row_ids": [str(rid) for rid in selected_row_ids]},
                   run_dir / "selected_row_ids.json")
        _dump_json({
            "ordinary_selected_row_ids": [
                str(row_id) for row_id in ordinary_final_selected_row_ids
            ]
        }, run_dir / "ordinary_selected_row_ids.json")
        _dump_json({"generalization_level": dict(state.current_generalization.generalization_level)},
                   run_dir / "generalization_level.json")
        _dump_json({"leak_distribution": leak_distribution},
                   run_dir / "leak_distribution.json")
        _dump_json({
            "iterations": trim_iteration_results,
        }, run_dir / "trim_iterations.json")
        _dump_json(final_release_public, run_dir / "trim_release.json")

    return TrimPipelineResult(
        task_type="classification", loss_metric=loss_metric_name,
        selected_row_ids=selected_row_ids,
        generalization_level=dict(state.current_generalization.generalization_level),
        state=state, initial_row_ids=initial_row_ids,
        selected_retention_class_keys=selected_retention_class_keys,
        termination_condition=termination_condition,
        final_actual_model_loss=final_actual_model_loss,
        utility_constraint_met=utility_constraint_met,
        validation_utility_constraint_met=validation_utility_constraint_met,
        baseline_val_loss=baseline_val_loss,
        loss_threshold_val=loss_threshold_val,
        baseline_test_loss=baseline_test_loss,
        loss_threshold_test=loss_threshold_test,
        original_leak_k=original_leak_k,
        original_leak_k_p1=original_leak_k_p1,
        original_leak_k_p2=original_leak_k_p2,
        original_leak_k_p3=original_leak_k_p3,
        original_leak_k_p4=original_leak_k_p4,
        original_leak_k_p5=original_leak_k_p5,
        final_leak_k=final_leak_k, leak_k_p5=leak_k_p5,
        tail_risk_p99=tail_risk_p99,
        final_leak_k_sorted=final_leak_k_sorted,
        final_leak_k_95_percentile=final_leak_k_95_percentile,
        release_target_k=final_release_target_k,
        release_assignment_counts=dict(
            final_release_result.get("assignment_counts", {})
        ),
        release_suppressed_row_ids=release_suppressed_row_ids,
        release_snapshot_history=public_release_snapshot_history,
        iteration_count=iteration_count, timings=timings,
        run_dir=str(run_dir) if run_dir is not None else None,
    )


# =========================================================================
# Helpers.
# =========================================================================
def _experiment_published_row_ids(current_encode, selected_row_ids, initial_row_id_set):
    """Return selected non-S0 rows plus S0 rows in a published final class."""
    published_non_initial = [rid for rid in selected_row_ids if rid not in initial_row_id_set]
    if not published_non_initial:
        return []
    published_keys = {
        tuple(v) for v in current_encode.loc[published_non_initial].itertuples(index=False, name=None)}
    merged = []
    merged_set = set()
    for rid in published_non_initial:
        if rid not in merged_set:
            merged.append(rid)
            merged_set.add(rid)
    for rid in initial_row_id_set:
        if rid not in current_encode.index:
            continue
        row_key = tuple(current_encode.loc[[rid]].itertuples(index=False, name=None))[0]
        if row_key in published_keys and rid not in merged_set:
            merged.append(rid)
            merged_set.add(rid)
    return merged


def _swap_generalization_attributes(generalization):
    numeric = tuple(getattr(generalization, "numeric_attributes", ()) or ())
    categorical = tuple(getattr(generalization, "categorical_attributes", ()) or ())
    attributes = numeric + categorical
    if attributes:
        return attributes
    return tuple(getattr(generalization, "generalization_level", {}) or ())


def _swap_attribute_columns(encoded, attribute):
    if attribute in encoded.columns:
        return [attribute]
    return [
        column
        for column in encoded.columns
        if isinstance(column, str) and column.startswith(f"{attribute}=")
    ]


def _swap_factorize_frame(frame):
    import pandas as pd

    if frame.shape[1] == 1:
        codes, _ = pd.factorize(
            frame.iloc[:, 0],
            sort=False,
            use_na_sentinel=False,
        )
    else:
        codes, _ = pd.factorize(
            pd.MultiIndex.from_frame(frame),
            sort=False,
            use_na_sentinel=False,
        )
    return codes.astype(np.intp, copy=False)


def _swap_categorical_frame_and_code(generalization, X_train_raw, attribute):
    import pandas as pd

    category_maps = getattr(generalization, "category_maps", {})
    categories = category_maps.get(attribute)
    if categories is None:
        frame = generalization._generalized_categorical_frame(
            X_train_raw, attribute)
        return frame, _swap_factorize_frame(frame)

    rules = getattr(generalization, "generalization_rules", {}).get(attribute, {})
    columns = [f"{attribute}={category}" for category in categories]
    sentinel = object()
    value_to_vector = {}
    vector_to_code = {}
    rows = []
    codes = np.empty(len(X_train_raw), dtype=np.intp)
    cache_key_func = getattr(
        generalization,
        "_categorical_cache_key",
        lambda value: value,
    )
    one_hot_func = getattr(generalization, "_one_hot_vector", None)

    for position, value in enumerate(X_train_raw[attribute]):
        cache_key = cache_key_func(value)
        vector = value_to_vector.get(cache_key)
        if vector is None:
            try:
                raw_vector = rules.get(value, sentinel)
            except TypeError:
                raw_vector = sentinel
            if raw_vector is sentinel:
                if one_hot_func is not None:
                    raw_vector = one_hot_func(attribute, value)
                else:
                    raw_vector = [0.0] * len(categories)
                    if value in categories:
                        raw_vector[categories[value]] = 1.0
            vector = tuple(raw_vector)
            value_to_vector[cache_key] = vector

        code = vector_to_code.get(vector)
        if code is None:
            code = len(vector_to_code)
            vector_to_code[vector] = code
        rows.append(vector)
        codes[position] = code

    return pd.DataFrame(rows, columns=columns, index=X_train_raw.index), codes


def _swap_attribute_frame_and_code(generalization, X_train_raw, attribute):
    import pandas as pd

    numeric_attributes = tuple(
        getattr(generalization, "numeric_attributes", ()) or ())
    categorical_attributes = tuple(
        getattr(generalization, "categorical_attributes", ()) or ())
    rules_by_attribute = getattr(generalization, "generalization_rules", {})

    if (
        attribute in numeric_attributes
        and hasattr(generalization, "_generalized_numeric_series")
    ):
        series = generalization._generalized_numeric_series(
            X_train_raw,
            attribute,
            rules_by_attribute.get(attribute, {}),
        )
        frame = pd.DataFrame({attribute: series}, index=X_train_raw.index)
        return frame, _swap_factorize_frame(frame)

    if (
        attribute in categorical_attributes
        and hasattr(generalization, "_generalized_categorical_frame")
    ):
        return _swap_categorical_frame_and_code(
            generalization, X_train_raw, attribute)

    full_encode = generalization.encode(X_train_raw)
    columns = _swap_attribute_columns(full_encode, attribute)
    if not columns:
        raise KeyError(f"Cannot find encoded columns for attribute {attribute!r}")
    frame = full_encode.loc[:, columns]
    return frame, _swap_factorize_frame(frame)


def _build_swap_encode_cache(generalization, X_train_raw, selected_attributes):
    import pandas as pd

    selected_attributes = tuple(selected_attributes)
    frames = []
    code_columns = {}
    attribute_columns = {}
    for attribute in _swap_generalization_attributes(generalization):
        frame, codes = _swap_attribute_frame_and_code(
            generalization, X_train_raw, attribute)
        frames.append(frame)
        attribute_columns[attribute] = list(frame.columns)
        if attribute in selected_attributes:
            code_columns[attribute] = codes

    if frames:
        encoded = pd.concat(frames, axis=1)
    else:
        encoded = generalization.encode(X_train_raw)

    for attribute in selected_attributes:
        if attribute in code_columns:
            continue
        columns = _swap_attribute_columns(encoded, attribute)
        if not columns:
            raise KeyError(
                f"Cannot find encoded columns for selected attribute {attribute!r}")
        frame = encoded.loc[:, columns]
        attribute_columns[attribute] = list(frame.columns)
        code_columns[attribute] = _swap_factorize_frame(frame)

    codes = pd.DataFrame(code_columns, index=X_train_raw.index)
    row_ids = np.asarray(list(encoded.index), dtype=object)
    row_id_to_position = {
        row_id: position
        for position, row_id in enumerate(row_ids)
    }
    return _SwapEncodeCache(
        encoded=encoded,
        codes=codes,
        row_ids=row_ids,
        row_id_to_position=row_id_to_position,
        selected_attributes=selected_attributes,
        attribute_columns=attribute_columns,
    )


def _update_swap_encode_cache(cache, generalization, X_train_raw, attribute):
    frame, codes = _swap_attribute_frame_and_code(
        generalization, X_train_raw, attribute)
    expected_columns = cache.attribute_columns.get(attribute)
    if expected_columns is None or expected_columns != list(frame.columns):
        rebuilt = _build_swap_encode_cache(
            generalization, X_train_raw, cache.selected_attributes)
        cache.encoded = rebuilt.encoded
        cache.codes = rebuilt.codes
        cache.row_ids = rebuilt.row_ids
        cache.row_id_to_position = rebuilt.row_id_to_position
        cache.attribute_columns = rebuilt.attribute_columns
        return cache.encoded

    cache.encoded.loc[:, expected_columns] = frame.to_numpy()
    if attribute in cache.codes.columns:
        cache.codes.loc[:, attribute] = codes
    return cache.encoded


def _swap_position_index_from_cache(cache):
    row_count = len(cache.row_ids)
    if row_count == 0:
        row_class_ids = np.empty(0, dtype=np.intp)
        class_member_positions = []
        class_total_sizes = np.empty(0, dtype=np.intp)
    elif cache.codes.shape[1] == 0:
        row_class_ids = np.zeros(row_count, dtype=np.intp)
        class_member_positions = [np.arange(row_count, dtype=np.intp)]
        class_total_sizes = np.asarray([row_count], dtype=np.intp)
    else:
        row_class_ids = cache.codes.groupby(
            list(cache.codes.columns),
            sort=False,
            dropna=False,
        ).ngroup().to_numpy(dtype=np.intp)
        class_total_sizes = np.bincount(row_class_ids)
        ordered_positions = np.argsort(row_class_ids, kind="stable")
        class_member_positions = np.split(
            ordered_positions,
            np.cumsum(class_total_sizes)[:-1],
        )

    return {
        "row_ids": cache.row_ids,
        "row_id_to_position": cache.row_id_to_position,
        "row_class_ids": row_class_ids,
        "class_member_positions": class_member_positions,
        "class_total_sizes": class_total_sizes,
    }


def _retention_class_index_groups_from_position_index(
    position_index,
    excluded_row_ids=None,
):
    excluded_row_ids = set(excluded_row_ids or ())
    row_ids = position_index["row_ids"]
    class_keys = position_index.get("class_keys")
    groups = []
    for class_id, class_positions in enumerate(
        position_index["class_member_positions"]
    ):
        class_row_ids = []
        for position in class_positions:
            row_id = row_ids[int(position)]
            if row_id not in excluded_row_ids:
                class_row_ids.append(row_id)
        if class_row_ids:
            class_key = class_keys[class_id] if class_keys is not None else class_id
            groups.append((class_key, class_row_ids))
    return groups


def _can_use_position_index_leakage(cache):
    if cache is None:
        return False
    return set(cache.selected_attributes) == set(cache.attribute_columns)


def _leakage_from_position_index(row_ids, position_index, initial_row_id_set):
    counts = _class_counts_from_position_index(
        row_ids, position_index, initial_row_id_set)
    return _leakage_from_class_counts(counts)


def _class_counts_from_position_index(row_ids, position_index, initial_row_id_set):
    row_id_to_position = position_index["row_id_to_position"]
    row_class_ids = position_index["row_class_ids"]
    class_count = len(position_index["class_total_sizes"])
    initial_row_id_set = set(initial_row_id_set or ())
    positions = np.fromiter(
        (
            row_id_to_position[row_id]
            for row_id in row_ids
            if row_id not in initial_row_id_set
        ),
        dtype=np.intp,
    )
    if positions.size == 0:
        return np.zeros(class_count, dtype=np.intp)
    return np.bincount(row_class_ids[positions], minlength=class_count)


def _leakage_from_class_counts(counts):
    positive_counts = counts[counts > 0]
    if positive_counts.size == 0:
        return 0
    return int(positive_counts.min())


def _build_model(factory_or_model):
    return factory_or_model() if callable(factory_or_model) else factory_or_model


def _xgboost_model_input(generalization, X_raw_subset):
    if hasattr(generalization, "encode_xgboost_leaf_space"):
        encoded = generalization.encode_xgboost_leaf_space(X_raw_subset)
        try:
            from scipy import sparse
        except ImportError:
            sparse = None
        if sparse is not None and sparse.issparse(encoded):
            encoded = encoded.toarray()
        return np.ascontiguousarray(np.asarray(encoded), dtype=np.float32)
    if hasattr(generalization, "encode_xgboost_model_input"):
        return generalization.encode_xgboost_model_input(X_raw_subset)
    return generalization.encode(X_raw_subset)


def _lga_theta(backend_model):
    return model_theta_tensor(backend_model)


def _val_gradient(theta, val_score_encode, y_val_tensor, *,
                  classes, class_count, use_multiclass):
    gradient_func = multiclass_logistic_gradient if use_multiclass else logistic_gradient
    X_val = to_device_tensor(val_score_encode, device=theta.device, dtype=theta.dtype)
    y_val = to_device_tensor(y_val_tensor, device=theta.device, dtype=theta.dtype).reshape(-1)
    if use_multiclass:
        y_val = _class_indices_tensor(y_val, classes=classes)
    return gradient_func(X_val, y_val, theta)


def _retention_class_row_ids(retention_class):
    index = getattr(retention_class, "index", None)
    if index is not None and not callable(index):
        return list(index)
    if hasattr(retention_class, "tolist"):
        return list(retention_class.tolist())
    return list(retention_class)


def _apply_vertical_swap_inplace(
    state, X_train_raw, selected_attributes, *,
    swap_level_encode=None, swap_position_index=None, swap_codes=None,
    initial_row_id_set=None, precomputed_selected_positions=None,
    precomputed_initial_mask=None,
):
    """Apply post-vertical swap to a candidate state without global bookkeeping."""
    try:
        from .enumeration_horizontal import build_retention_class_position_index
    except ImportError:  # pragma: no cover
        from enumeration_horizontal import build_retention_class_position_index

    swap_t0 = time.perf_counter()
    swap_iterations = 0
    swap_move_out_row_count = 0
    swap_move_in_row_count = 0
    swap_break_reason = "max_iterations"
    initial_row_id_set = set(initial_row_id_set or ())
    if len(state.selected_row_ids) >= len(X_train_raw):
        return {
            "swap_iterations": swap_iterations,
            "elapsed_seconds": time.perf_counter() - swap_t0,
            "encode_time_seconds": 0.0,
            "position_index_time_seconds": 0.0,
            "loop_time_seconds": 0.0,
            "sync_time_seconds": 0.0,
            "break_reason": "no_candidates",
            "selected_row_count": len(state.selected_row_ids),
            "candidate_row_count": 0,
            "move_out_row_count": 0,
            "move_in_row_count": 0,
            "swap_level_encode": (
                swap_level_encode
                if swap_level_encode is not None
                else state.current_train_encode
            ),
            "swap_position_index": swap_position_index,
            "swap_codes": swap_codes,
        }
    swap_encode_t0 = time.perf_counter()
    if swap_level_encode is None:
        swap_level_encode = state.current_generalization.encode(X_train_raw)
    swap_encode_time = time.perf_counter() - swap_encode_t0
    swap_position_index_t0 = time.perf_counter()
    if swap_position_index is None:
        swap_position_index = build_retention_class_position_index(
            swap_level_encode, selected_attributes=selected_attributes,
            sort_keys=False)
    swap_position_index_time = time.perf_counter() - swap_position_index_t0
    swap_row_ids = swap_position_index["row_ids"]
    swap_row_id_to_position = swap_position_index["row_id_to_position"]
    swap_row_class_ids = swap_position_index["row_class_ids"]
    swap_class_member_positions = swap_position_index["class_member_positions"]
    swap_class_total_sizes = swap_position_index["class_total_sizes"]
    swap_class_count = len(swap_class_total_sizes)
    swap_row_count = len(swap_row_ids)
    if precomputed_selected_positions is None:
        swap_selected_positions = np.fromiter(
            (swap_row_id_to_position[rid] for rid in state.selected_row_ids),
            dtype=np.intp, count=len(state.selected_row_ids))
    else:
        swap_selected_positions = np.asarray(
            precomputed_selected_positions, dtype=np.intp).copy()
    swap_selected_mask = np.zeros(swap_row_count, dtype=bool)
    swap_selected_mask[swap_selected_positions] = True
    swap_selected_order = np.full(swap_row_count, -1, dtype=np.int64)
    swap_selected_order[swap_selected_positions] = np.arange(
        len(swap_selected_positions), dtype=np.int64)
    swap_next_order = int(len(swap_selected_positions))
    swap_selected_count = int(len(swap_selected_positions))
    if precomputed_initial_mask is not None:
        swap_initial_mask = np.asarray(precomputed_initial_mask, dtype=bool)
    else:
        swap_initial_mask = np.zeros(swap_row_count, dtype=bool)
        if initial_row_id_set:
            swap_initial_positions = [
                swap_row_id_to_position[row_id]
                for row_id in initial_row_id_set
                if row_id in swap_row_id_to_position
            ]
            swap_initial_mask[swap_initial_positions] = True
    swap_member_arrays = [
        np.asarray(positions, dtype=np.intp)
        for positions in swap_class_member_positions
    ]
    swap_swappable_sizes = np.zeros(swap_class_count, dtype=np.intp)
    swap_candidate_sizes = np.zeros(swap_class_count, dtype=np.intp)
    swap_swappable_first_order = np.full(swap_class_count, -1, dtype=np.int64)
    swap_candidate_first_position = np.full(swap_class_count, -1, dtype=np.int64)

    swap_swappable_heap = []
    swap_candidate_heap = []

    if swap_class_count:
        selected_class_ids = swap_row_class_ids[swap_selected_positions]
        selected_counts = np.bincount(
            selected_class_ids, minlength=swap_class_count)
        swap_candidate_sizes[:] = swap_class_total_sizes - selected_counts

        swap_candidate_positions = np.flatnonzero(~swap_selected_mask)
        if len(swap_candidate_positions):
            candidate_class_ids = swap_row_class_ids[swap_candidate_positions]
            first_positions = np.full(
                swap_class_count, swap_row_count, dtype=np.int64)
            np.minimum.at(
                first_positions, candidate_class_ids,
                swap_candidate_positions)
            candidate_first_mask = first_positions != swap_row_count
            swap_candidate_first_position[candidate_first_mask] = (
                first_positions[candidate_first_mask])

        swap_swappable_positions = swap_selected_positions[
            ~swap_initial_mask[swap_selected_positions]]
        if len(swap_swappable_positions):
            swappable_class_ids = swap_row_class_ids[swap_swappable_positions]
            swap_swappable_sizes[:] = np.bincount(
                swappable_class_ids, minlength=swap_class_count)
            first_orders = np.full(
                swap_class_count, swap_row_count, dtype=np.int64)
            np.minimum.at(
                first_orders, swappable_class_ids,
                swap_selected_order[swap_swappable_positions])
            swappable_first_mask = first_orders != swap_row_count
            swap_swappable_first_order[swappable_first_mask] = (
                first_orders[swappable_first_mask])

    def _recompute_swap_class(swap_class_id):
        swap_positions = swap_member_arrays[swap_class_id]
        if swap_positions.size == 0:
            swap_swappable_sizes[swap_class_id] = 0
            swap_candidate_sizes[swap_class_id] = 0
            swap_swappable_first_order[swap_class_id] = -1
            swap_candidate_first_position[swap_class_id] = -1
            return
        swap_selected = swap_selected_mask[swap_positions]
        swap_swappable_positions = swap_positions[
            swap_selected & ~swap_initial_mask[swap_positions]]
        swap_swappable_sizes[swap_class_id] = len(swap_swappable_positions)
        if len(swap_swappable_positions):
            swap_swappable_first_order[swap_class_id] = int(
                swap_selected_order[swap_swappable_positions].min())
        else:
            swap_swappable_first_order[swap_class_id] = -1
        swap_candidate_positions = swap_positions[~swap_selected]
        swap_candidate_sizes[swap_class_id] = len(swap_candidate_positions)
        if len(swap_candidate_positions):
            swap_candidate_first_position[swap_class_id] = int(
                swap_candidate_positions[0])
        else:
            swap_candidate_first_position[swap_class_id] = -1

    def _push_swappable_class(swap_class_id):
        swap_size = int(swap_swappable_sizes[swap_class_id])
        swap_first_order = int(swap_swappable_first_order[swap_class_id])
        if swap_size > 0 and swap_first_order >= 0:
            heapq.heappush(
                swap_swappable_heap,
                (swap_size, swap_first_order, swap_class_id),
            )

    def _push_candidate_class(swap_class_id):
        swap_size = int(swap_candidate_sizes[swap_class_id])
        swap_first_position = int(swap_candidate_first_position[swap_class_id])
        if swap_size > 0 and swap_first_position >= 0:
            heapq.heappush(
                swap_candidate_heap,
                (-swap_size, swap_first_position, swap_class_id),
            )

    def _peek_swappable_class():
        while swap_swappable_heap:
            swap_size, swap_first_order, swap_class_id = swap_swappable_heap[0]
            if (
                int(swap_swappable_sizes[swap_class_id]) == swap_size
                and int(swap_swappable_first_order[swap_class_id])
                == swap_first_order
                and swap_size > 0
            ):
                return swap_size, swap_class_id
            heapq.heappop(swap_swappable_heap)
        return None

    def _peek_candidate_class():
        while swap_candidate_heap:
            swap_negative_size, swap_first_position, swap_class_id = (
                swap_candidate_heap[0]
            )
            if (
                int(swap_candidate_sizes[swap_class_id]) == -swap_negative_size
                and int(swap_candidate_first_position[swap_class_id])
                == swap_first_position
                and swap_negative_size < 0
            ):
                return -swap_negative_size, swap_class_id
            heapq.heappop(swap_candidate_heap)
        return None

    swap_swappable_heap = [
        (
            int(swap_swappable_sizes[swap_class_id]),
            int(swap_swappable_first_order[swap_class_id]),
            int(swap_class_id),
        )
        for swap_class_id in np.flatnonzero(
            (swap_swappable_sizes > 0)
            & (swap_swappable_first_order >= 0))
    ]
    heapq.heapify(swap_swappable_heap)
    swap_candidate_heap = [
        (
            -int(swap_candidate_sizes[swap_class_id]),
            int(swap_candidate_first_position[swap_class_id]),
            int(swap_class_id),
        )
        for swap_class_id in np.flatnonzero(
            (swap_candidate_sizes > 0)
            & (swap_candidate_first_position >= 0))
    ]
    heapq.heapify(swap_candidate_heap)

    swap_sync_time = 0.0
    swap_sync_position_list = []
    swap_loop_t0 = time.perf_counter()
    for _ in range(VERTICAL_SWAP_MAX_ITERATIONS):
        if swap_selected_count >= swap_row_count:
            swap_break_reason = "no_candidates"
            break
        if swap_selected_count == 0 or swap_class_count == 0:
            swap_break_reason = "no_selected_groups"
            break
        swap_swappable_top = _peek_swappable_class()
        if swap_swappable_top is None:
            swap_break_reason = "no_swappable_selected_groups"
            break
        swap_candidate_top = _peek_candidate_class()
        if swap_candidate_top is None:
            swap_break_reason = "no_candidate_groups"
            break
        a = int(swap_swappable_top[0])
        b, b_class_id = swap_candidate_top
        b = int(b)
        b_class_id = int(b_class_id)
        if not (a < b):
            swap_break_reason = "converged"
            break
        swap_chosen_class_ids = []
        swap_move_out_total = 0
        while True:
            swap_next_swappable = _peek_swappable_class()
            if swap_next_swappable is None:
                break
            swap_size, swap_class_id = swap_next_swappable
            swap_size = int(swap_size)
            swap_class_id = int(swap_class_id)
            if swap_move_out_total + swap_size <= b:
                heapq.heappop(swap_swappable_heap)
                swap_chosen_class_ids.append(swap_class_id)
                swap_move_out_total += swap_size
            else:
                break
        if not swap_chosen_class_ids:
            swap_break_reason = "smallest_selected_exceeds_candidate"
            break
        swap_touched_class_ids = set(swap_chosen_class_ids)
        swap_touched_class_ids.add(b_class_id)
        swap_move_out_parts = []
        for swap_class_id in swap_chosen_class_ids:
            swap_class_positions = swap_member_arrays[swap_class_id]
            swap_class_selected_positions = swap_class_positions[
                swap_selected_mask[swap_class_positions]
                & ~swap_initial_mask[swap_class_positions]]
            if len(swap_class_selected_positions):
                swap_move_out_parts.append(swap_class_selected_positions)
        if not swap_move_out_parts:
            swap_break_reason = "no_move_out"
            break
        swap_move_out_positions = np.concatenate(swap_move_out_parts)
        swap_candidate_class_positions = swap_member_arrays[b_class_id]
        swap_move_in_positions = swap_candidate_class_positions[
            ~swap_selected_mask[swap_candidate_class_positions]]
        if len(swap_move_in_positions) == 0:
            swap_break_reason = "no_move_in"
            break
        swap_selected_mask[swap_move_out_positions] = False
        swap_selected_order[swap_move_out_positions] = -1
        swap_selected_mask[swap_move_in_positions] = True
        swap_selected_order[swap_move_in_positions] = np.arange(
            swap_next_order,
            swap_next_order + len(swap_move_in_positions),
            dtype=np.int64)
        swap_next_order += int(len(swap_move_in_positions))
        swap_selected_count += int(
            len(swap_move_in_positions) - len(swap_move_out_positions))
        swap_move_out_row_count += int(len(swap_move_out_positions))
        swap_move_in_row_count += int(len(swap_move_in_positions))
        for swap_class_id in swap_touched_class_ids:
            _recompute_swap_class(swap_class_id)
            _push_swappable_class(swap_class_id)
            _push_candidate_class(swap_class_id)
        swap_sync_position_list.extend(
            int(position) for position in swap_move_in_positions.tolist())
        swap_iterations += 1
    swap_loop_time = max(0.0, time.perf_counter() - swap_loop_t0)
    if swap_sync_position_list:
        swap_sync_t0 = time.perf_counter()
        swap_sync_positions = list(dict.fromkeys(swap_sync_position_list))
        state.current_train_encode.iloc[swap_sync_positions, :] = (
            swap_level_encode.iloc[swap_sync_positions, :].to_numpy())
        state.current_train_tensor[swap_sync_positions] = to_device_tensor(
            swap_level_encode.iloc[swap_sync_positions, :],
            device=state.current_train_tensor.device,
            dtype=state.current_train_tensor.dtype)
        swap_sync_time += time.perf_counter() - swap_sync_t0
    swap_final_positions = np.flatnonzero(swap_selected_mask)
    swap_final_positions = swap_final_positions[
        np.argsort(swap_selected_order[swap_final_positions], kind="stable")]
    state.selected_row_ids = np.asarray(
        swap_row_ids, dtype=object)[swap_final_positions].tolist()
    return {
        "swap_iterations": swap_iterations,
        "elapsed_seconds": time.perf_counter() - swap_t0,
        "encode_time_seconds": swap_encode_time,
        "position_index_time_seconds": swap_position_index_time,
        "loop_time_seconds": swap_loop_time,
        "sync_time_seconds": swap_sync_time,
        "break_reason": swap_break_reason,
        "selected_row_count": len(state.selected_row_ids),
        "candidate_row_count": len(swap_row_ids) - len(state.selected_row_ids),
        "move_out_row_count": swap_move_out_row_count,
        "move_in_row_count": swap_move_in_row_count,
        "swap_level_encode": swap_level_encode,
        "swap_position_index": swap_position_index,
        "swap_codes": swap_codes,
    }


def _apply_vertical_postprocessing_inplace(
    state,
    X_train_raw,
    selected_attributes,
    *,
    policy,
    move_out_threshold=None,
    swap_level_encode=None,
    swap_position_index=None,
    swap_codes=None,
    initial_row_id_set=None,
    precomputed_selected_positions=None,
    precomputed_initial_mask=None,
):
    """Apply the configured vertical postprocessing policy in place."""

    if policy == FULL_SWAP_POSTPROCESSING:
        stats = _apply_vertical_swap_inplace(
            state,
            X_train_raw,
            selected_attributes,
            swap_level_encode=swap_level_encode,
            swap_position_index=swap_position_index,
            swap_codes=swap_codes,
            initial_row_id_set=initial_row_id_set,
            precomputed_selected_positions=precomputed_selected_positions,
            precomputed_initial_mask=precomputed_initial_mask,
        )
        stats.update({
            "postprocessing_policy": policy,
            "move_out_threshold": None,
            "extra_swap_in_count": 0,
        })
        return stats

    postprocessing_started_at = time.perf_counter()
    encode_started_at = time.perf_counter()
    if swap_level_encode is None:
        swap_level_encode = state.current_generalization.encode(X_train_raw)
    encode_time = time.perf_counter() - encode_started_at

    position_index_started_at = time.perf_counter()
    if swap_position_index is None:
        swap_position_index = build_retention_class_position_index(
            swap_level_encode,
            selected_attributes=selected_attributes,
            sort_keys=False,
        )
    position_index_time = time.perf_counter() - position_index_started_at

    loop_started_at = time.perf_counter()
    move_out_row_count = 0
    affected_class_count = 0
    break_reason = "postprocessing_disabled"
    if policy == EVICT_LOW_K_POSTPROCESSING:
        row_ids = swap_position_index["row_ids"]
        row_id_to_position = swap_position_index["row_id_to_position"]
        row_class_ids = swap_position_index["row_class_ids"]
        class_count = len(swap_position_index["class_total_sizes"])
        selected_positions = np.fromiter(
            (row_id_to_position[row_id] for row_id in state.selected_row_ids),
            dtype=np.intp,
            count=len(state.selected_row_ids),
        )
        initial_row_id_set = set(initial_row_id_set or ())
        movable_positions = np.asarray([
            position
            for position in selected_positions
            if row_ids[position] not in initial_row_id_set
        ], dtype=np.intp)
        movable_class_sizes = np.zeros(class_count, dtype=np.intp)
        if len(movable_positions):
            movable_class_sizes = np.bincount(
                row_class_ids[movable_positions],
                minlength=class_count,
            )
        evicted_class_ids = np.flatnonzero(
            (movable_class_sizes > 0)
            & (movable_class_sizes < int(move_out_threshold))
        )
        affected_class_count = int(len(evicted_class_ids))
        if affected_class_count:
            evicted_class_id_set = set(evicted_class_ids.tolist())
            evicted_row_ids = {
                row_ids[position]
                for position in movable_positions
                if int(row_class_ids[position]) in evicted_class_id_set
            }
            move_out_row_count = len(evicted_row_ids)
            state.selected_row_ids = [
                row_id
                for row_id in state.selected_row_ids
                if row_id not in evicted_row_ids
            ]
            break_reason = "low_k_groups_evicted"
        else:
            break_reason = "no_low_k_groups"
    loop_time = time.perf_counter() - loop_started_at

    return {
        "swap_iterations": affected_class_count,
        "elapsed_seconds": time.perf_counter() - postprocessing_started_at,
        "encode_time_seconds": encode_time,
        "position_index_time_seconds": position_index_time,
        "loop_time_seconds": loop_time,
        "sync_time_seconds": 0.0,
        "break_reason": break_reason,
        "selected_row_count": len(state.selected_row_ids),
        "candidate_row_count": len(X_train_raw) - len(state.selected_row_ids),
        "move_out_row_count": move_out_row_count,
        "move_in_row_count": 0,
        "extra_swap_in_count": 0,
        "postprocessing_policy": policy,
        "move_out_threshold": move_out_threshold,
        "swap_level_encode": swap_level_encode,
        "swap_position_index": swap_position_index,
        "swap_codes": swap_codes,
    }


def _greedy_select(
    state, *, remaining_retention_classes, X_train_raw, X_val_raw,
    train_original_encode, backend_model, reference_proxy_model,
    reference_model_filtering, proxy_factory,
    proxy_uses_xgboost_input, leakage_func, rank_top_k, standardizer,
    row_id_to_position, train_original_tensor, val_original_tensor,
    train_y_tensor, val_y_tensor, classification_classes,
    class_count, use_multiclass, selection_timing, action_log_path, iteration_count,
    current_train_level_encode=None, swap_encode_cache=None,
    initial_row_id_set=None, candidate_release_evaluator=None,
    enabled_action_types=None, vertical_postprocessing=FULL_SWAP_POSTPROCESSING,
    move_out_threshold=None, whole_retention_group_admission=False,
    candidate_shortlist=None, candidate_state_scorer=None,
    selection_score=None, candidate_observer=None,
):
    """Rank candidates with LGA, then proxy-score the top-k TRIM releases."""
    try:
        from .lga import cal_lga_batch
        from .enumeration_horizontal import build_retention_class_position_index
    except ImportError:  # pragma: no cover - direct script import compatibility
        from lga import cal_lga_batch
        from enumeration_horizontal import build_retention_class_position_index

    t_start = time.perf_counter()
    timing_rank = 0.0
    timing_reference_filter = 0.0
    timing_lga = 0.0
    timing_proxy_retrain = 0.0
    timing_leakage = 0.0
    lga_calls = 0
    lga_candidate_evaluations = 0
    proxy_retrain_calls = 0
    leakage_calls = 0
    optimized_privacy_time = 0.0
    candidate_observer_time = 0.0
    reference_filter_remaining_row_count = 0
    reference_filter_reference_correct_row_count = 0
    reference_filter_qualified_row_count = 0
    reference_filter_retention_class_count = 0
    initial_row_id_set = set(initial_row_id_set or ())
    enabled_action_types = frozenset(
        enabled_action_types
        if enabled_action_types is not None
        else (RETENTION_CLASS_ACTION, VERTICAL_REFINEMENT_ACTION)
    )
    if candidate_release_evaluator is None:
        raise ValueError("candidate_release_evaluator is required.")

    current_generalization = state.current_generalization
    current_level = dict(current_generalization.generalization_level)
    selected_level_encode = (
        current_train_level_encode
        if current_train_level_encode is not None
        else current_generalization.encode(X_train_raw)
    )
    selected_level_tensor = to_device_tensor(
        selected_level_encode, device=state.current_train_tensor.device,
        dtype=state.current_train_tensor.dtype)
    # LGA defines D_t as the currently selected rows. The full tensor remains
    # the row-indexed state cache used to materialize candidate updates.
    selected_ids = list(state.selected_row_ids)
    selected_row_positions = [row_id_to_position[rid] for rid in selected_ids]
    selected_row_position_tensor = torch.as_tensor(
        selected_row_positions,
        device=state.current_train_tensor.device,
        dtype=torch.long,
    )
    selected_current_tensor = state.current_train_tensor.index_select(
        0,
        selected_row_position_tensor,
    )
    selected_train_y_tensor = train_y_tensor.index_select(
        0,
        selected_row_position_tensor,
    )
    train_current_score_encode = standardizer.transform(selected_current_tensor)
    train_original_score_encode = standardizer.transform(train_original_tensor)
    current_val_encode = current_generalization.encode(X_val_raw)
    current_val_tensor = to_device_tensor(
        current_val_encode,
        device=state.current_train_tensor.device,
        dtype=state.current_train_tensor.dtype)
    val_score_encode = standardizer.transform(current_val_tensor)
    encoded_columns = state.current_train_encode.columns
    attribute_columns = {
        attribute: _swap_attribute_columns(state.current_train_encode, attribute)
        for attribute in current_level
    }
    attribute_column_positions = {}
    for attribute, columns in attribute_columns.items():
        if not columns:
            continue
        positions = encoded_columns.get_indexer(columns)
        if np.any(positions < 0):
            continue
        attribute_column_positions[attribute] = positions.tolist()

    current_position_index = None
    current_class_counts = None
    if _can_use_position_index_leakage(swap_encode_cache):
        _lt = time.perf_counter()
        current_position_index = _swap_position_index_from_cache(swap_encode_cache)
        optimized_privacy_time += time.perf_counter() - _lt

    # Current published k and model loss (proxy val loss).
    if current_position_index is not None:
        _lt = time.perf_counter()
        current_class_counts = _class_counts_from_position_index(
            state.selected_row_ids, current_position_index, initial_row_id_set)
        current_k = _leakage_from_class_counts(current_class_counts)
        optimized_privacy_time += time.perf_counter() - _lt
    else:
        current_selected_encode = state.current_train_encode.loc[state.selected_row_ids]
        current_k = (leakage_func(current_selected_encode)
                     if not current_selected_encode.empty else 0)
    current_model_loss = state.actual_model_loss

    theta = _lga_theta(backend_model)
    current_val_gradient = _val_gradient(
        theta, val_score_encode, val_y_tensor,
        classes=classification_classes, class_count=class_count,
        use_multiclass=use_multiclass)

    # ---- Retention-class candidates. --------------------------------------
    retention_items = []
    if RETENTION_CLASS_ACTION in enabled_action_types:
        retention_items = (list(remaining_retention_classes.items())
                           if hasattr(remaining_retention_classes, "items")
                           else list(remaining_retention_classes))
        _t = time.perf_counter()
        remaining_row_ids = []
        seen_remaining_row_ids = set()
        for _rc_key, rc in (
            retention_items if reference_model_filtering else ()
        ):
            for row_id in _retention_class_row_ids(rc):
                if row_id in selected_ids or row_id in seen_remaining_row_ids:
                    continue
                remaining_row_ids.append(row_id)
                seen_remaining_row_ids.add(row_id)

        reference_filter_remaining_row_count = len(remaining_row_ids)
        qualified_row_ids = set()
        if remaining_row_ids:
            remaining_positions = [
                row_id_to_position[row_id] for row_id in remaining_row_ids
            ]
            remaining_position_tensor = torch.as_tensor(
                remaining_positions,
                device=state.current_train_tensor.device,
                dtype=torch.long,
            )
            backend_filter_encode = standardizer.transform(
                selected_level_tensor.index_select(
                    0,
                    remaining_position_tensor,
                )
            )
            if proxy_uses_xgboost_input:
                reference_filter_encode = _xgboost_model_input(
                    current_generalization,
                    X_train_raw.loc[remaining_row_ids],
                )
            else:
                reference_filter_encode = backend_filter_encode

            reference_predictions = (
                reference_proxy_model.predict_tensor(reference_filter_encode)
                .detach()
                .cpu()
            )
            backend_predictions = (
                backend_model.predict_tensor(backend_filter_encode)
                .detach()
                .cpu()
            )
            remaining_labels = (
                train_y_tensor.index_select(0, remaining_position_tensor)
                .detach()
                .cpu()
            )
            reference_correct = reference_predictions.eq(remaining_labels)
            qualified_mask = reference_correct & backend_predictions.ne(
                remaining_labels
            )
            reference_filter_reference_correct_row_count = int(
                reference_correct.sum().item()
            )
            qualified_row_ids = {
                row_id
                for row_id, qualified in zip(
                    remaining_row_ids,
                    qualified_mask.tolist(),
                )
                if qualified
            }
            reference_filter_qualified_row_count = len(qualified_row_ids)

        retention_items = [
            (rc_key, rc)
            for rc_key, rc in retention_items
            if (
                not reference_model_filtering
                or any(
                    row_id in qualified_row_ids
                    for row_id in _retention_class_row_ids(rc)
                )
            )
        ]
        if reference_model_filtering:
            reference_filter_retention_class_count = len(retention_items)
            timing_reference_filter = time.perf_counter() - _t
    retention_candidates = []
    selected_set = set(selected_ids)
    for position, (rc_key, rc) in enumerate(retention_items):
        admit_candidates = [
            rid for rid in _retention_class_row_ids(rc)
            if rid not in selected_set
        ]
        rc_size = len(admit_candidates)
        admit_count = (
            rc_size
            if whole_retention_group_admission
            else (min(rc_size, current_k) if current_k > 0 else rc_size)
        )
        admitted = admit_candidates[:admit_count]
        row_positions = [row_id_to_position[rid] for rid in admitted]
        row_position_tensor = torch.as_tensor(
            row_positions, device=state.current_train_tensor.device, dtype=torch.long)
        next_changed_tensor = selected_level_tensor.index_select(0, row_position_tensor)
        retention_candidates.append({
            "candidate_type": "retention_class",
            "retention_class_position": position,
            "retention_class_key": rc_key,
            "rank_candidate_id": rc_key,
            "retention_class_index": admitted,
            "row_position_tensor": row_position_tensor,
            "next_changed_tensor": next_changed_tensor,
            "lga_score": None,
        })

    # Batched LGA for all retention candidates at once.
    if retention_candidates:
        _t = time.perf_counter()
        selected_level_score_tensor = standardizer.transform(selected_level_tensor)
        lga_scores = cal_lga_batch(
            train_original_encode=train_original_score_encode,
            val_original_encode=val_score_encode,
            train_y=selected_train_y_tensor, val_y=val_y_tensor,
            train_current_encode=train_current_score_encode,
            backend_model=backend_model,
            candidate_changed_positions=[c["row_position_tensor"] for c in retention_candidates],
            candidate_next_changed_encodes=[
                selected_level_score_tensor.index_select(0, c["row_position_tensor"])
                for c in retention_candidates],
            candidate_admitted_labels=[
                train_y_tensor.index_select(0, c["row_position_tensor"])
                for c in retention_candidates
            ],
            precomputed_train_original_tensor=train_original_score_encode,
            precomputed_val_original_tensor=val_score_encode,
            precomputed_train_current_tensor=train_current_score_encode,
            precomputed_train_y_tensor=selected_train_y_tensor,
            precomputed_val_y_tensor=val_y_tensor,
            precomputed_theta=theta,
            precomputed_val_gradient=current_val_gradient,
        )
        timing_lga += time.perf_counter() - _t
        lga_calls += 1
        lga_candidate_evaluations += len(retention_candidates)
        for c, s in zip(retention_candidates, lga_scores):
            c["lga_score"] = float(s)
            c["rank_score"] = float(s)

    # ---- Vertical-refinement candidates. ----------------------------------
    vertical_candidates = []
    selected_train_raw = X_train_raw.loc[selected_ids]
    selected_lga_position_tensor = torch.arange(
        len(selected_ids),
        device=state.current_train_tensor.device,
        dtype=torch.long,
    )
    vertical_levels = (
        current_level.items()
        if VERTICAL_REFINEMENT_ACTION in enabled_action_types
        else ()
    )
    for attribute, level in vertical_levels:
        if int(level) <= 0:
            continue
        next_level = dict(current_level)
        next_level[attribute] = int(level) - 1
        next_generalization = current_generalization.change_level(next_level)
        candidate_attribute_columns = attribute_columns.get(attribute, [])
        candidate_attribute_positions = attribute_column_positions.get(attribute)
        if candidate_attribute_columns and candidate_attribute_positions is not None:
            next_val_attribute_frame, _ = _swap_attribute_frame_and_code(
                next_generalization, X_val_raw, attribute)
            if list(next_val_attribute_frame.columns) == candidate_attribute_columns:
                next_val_tensor = current_val_tensor.clone()
                next_val_tensor[:, candidate_attribute_positions] = to_device_tensor(
                    next_val_attribute_frame,
                    device=state.current_train_tensor.device,
                    dtype=state.current_train_tensor.dtype)
            else:
                next_val_tensor = to_device_tensor(
                    next_generalization.encode(X_val_raw),
                    device=state.current_train_tensor.device,
                    dtype=state.current_train_tensor.dtype)
                candidate_attribute_columns = []
                candidate_attribute_positions = None
        else:
            next_val_tensor = to_device_tensor(
                next_generalization.encode(X_val_raw),
                device=state.current_train_tensor.device,
                dtype=state.current_train_tensor.dtype)
        next_val_score_encode = standardizer.transform(next_val_tensor)
        next_val_gradient = _val_gradient(
            theta, next_val_score_encode, val_y_tensor,
            classes=classification_classes, class_count=class_count,
            use_multiclass=use_multiclass)
        if candidate_attribute_columns and candidate_attribute_positions is not None:
            next_selected_attribute_frame, _ = _swap_attribute_frame_and_code(
                next_generalization, selected_train_raw, attribute)
            if list(next_selected_attribute_frame.columns) == candidate_attribute_columns:
                next_changed_tensor = selected_current_tensor.clone()
                next_changed_tensor[:, candidate_attribute_positions] = to_device_tensor(
                    next_selected_attribute_frame,
                    device=state.current_train_tensor.device,
                    dtype=state.current_train_tensor.dtype)
            else:
                next_selected_encode = next_generalization.encode(selected_train_raw)
                next_changed_tensor = to_device_tensor(
                    next_selected_encode, device=state.current_train_tensor.device,
                    dtype=state.current_train_tensor.dtype)
                candidate_attribute_columns = []
                candidate_attribute_positions = None
        else:
            next_selected_encode = next_generalization.encode(selected_train_raw)
            next_changed_tensor = to_device_tensor(
                next_selected_encode, device=state.current_train_tensor.device,
                dtype=state.current_train_tensor.dtype)
        # Per-candidate LGA (vertical candidates are few, one per attribute).
        _t = time.perf_counter()
        try:
            from .lga import cal_lga
        except ImportError:  # pragma: no cover
            from lga import cal_lga
        lga_score = float(cal_lga(
            train_original_encode=train_original_score_encode,
            val_original_encode=next_val_score_encode,
            train_y=selected_train_y_tensor, val_y=val_y_tensor,
            train_current_encode=train_current_score_encode,
            train_next_encode=None,
            backend_model=backend_model,
            changed_row_positions=selected_lga_position_tensor,
            train_next_changed_encode=standardizer.transform(next_changed_tensor),
            precomputed_train_original_tensor=train_original_score_encode,
            precomputed_val_original_tensor=next_val_score_encode,
            precomputed_train_current_tensor=train_current_score_encode,
            precomputed_train_y_tensor=selected_train_y_tensor,
            precomputed_val_y_tensor=val_y_tensor,
            precomputed_theta=theta,
            precomputed_val_gradient=next_val_gradient,
        ))
        timing_lga += time.perf_counter() - _t
        lga_calls += 1
        lga_candidate_evaluations += 1
        vertical_candidates.append({
            "candidate_type": "vertical_refinement",
            "attribute": attribute,
            "rank_candidate_id": attribute,
            "next_generalization": next_generalization,
            "next_val_score_encode": next_val_score_encode,
            "selected_row_ids": selected_ids,
            "row_position_tensor": selected_row_position_tensor,
            "next_changed_tensor": next_changed_tensor,
            "attribute_columns": candidate_attribute_columns,
            "attribute_column_positions": candidate_attribute_positions,
            "lga_score": lga_score,
            "rank_score": lga_score,
        })

    timing_rank = (
        time.perf_counter()
        - t_start
        - timing_reference_filter
        - timing_lga
    )

    # ---- Optional full-pool shortlist, otherwise per-type LGA top-k. -------
    if candidate_shortlist is None:
        top_retention = sorted(
            retention_candidates,
            key=lambda candidate: candidate["rank_score"],
            reverse=True,
        )[:rank_top_k]
        top_vertical = sorted(
            vertical_candidates,
            key=lambda candidate: candidate["rank_score"],
            reverse=True,
        )[:rank_top_k]
    else:
        all_ranked_candidates = [*retention_candidates, *vertical_candidates]
        shortlist_positions = tuple(candidate_shortlist(
            CandidateShortlistContext(
                iteration=int(iteration_count),
                rank_top_k=int(rank_top_k),
                candidates=tuple(
                    RankedCandidate(
                        position=position,
                        candidate_type=candidate["candidate_type"],
                        candidate_id=candidate["rank_candidate_id"],
                        rank_score=float(candidate["rank_score"]),
                    )
                    for position, candidate in enumerate(all_ranked_candidates)
                ),
            )
        ))
        if any(isinstance(position, bool) or not isinstance(position, int)
               for position in shortlist_positions):
            raise TypeError("candidate_shortlist must return integer positions.")
        if len(shortlist_positions) != len(set(shortlist_positions)):
            raise ValueError("candidate_shortlist returned duplicate positions.")
        if any(position < 0 or position >= len(all_ranked_candidates)
               for position in shortlist_positions):
            raise IndexError("candidate_shortlist returned an invalid position.")
        shortlisted_candidates = [
            all_ranked_candidates[position]
            for position in shortlist_positions
        ]
        top_retention = [
            candidate for candidate in shortlisted_candidates
            if candidate["candidate_type"] == RETENTION_CLASS_ACTION
        ]
        top_vertical = [
            candidate for candidate in shortlisted_candidates
            if candidate["candidate_type"] == VERTICAL_REFINEMENT_ACTION
        ]

    candidates = []
    best = None
    best_state = None
    best_metadata = None
    selected_position = None

    current_state_score_encode = None
    current_state_score_y = None
    if candidate_state_scorer is not None:
        current_selected_positions = [
            row_id_to_position[row_id]
            for row_id in selected_ids
        ]
        current_selected_position_tensor = torch.as_tensor(
            current_selected_positions,
            device=state.current_train_tensor.device,
            dtype=torch.long,
        )
        current_state_score_encode = standardizer.transform(
            state.current_train_tensor.index_select(
                0,
                current_selected_position_tensor,
            )
        )
        current_state_score_y = train_y_tensor.index_select(
            0,
            current_selected_position_tensor,
        )

    def _candidate_log_state(
        *,
        predicted_loss,
        actual_loss,
        is_trusted,
        generalization,
        selected_row_ids,
    ):
        return SelectionState(
            predicted_model_loss=predicted_loss,
            actual_model_loss=actual_loss,
            is_actual_model_loss_trusted=is_trusted,
            current_generalization=generalization,
            selected_row_ids=list(selected_row_ids),
            current_train_encode=state.current_train_encode,
            current_train_tensor=None,
            action_log_path=state.action_log_path,
        )

    def _add_scored_candidate(
        candidate_record,
        state_factory,
        *,
        predicted_loss,
        actual_loss,
        is_trusted,
        generalization,
        selected_row_ids,
        selected_metadata=None,
    ):
        nonlocal best, best_state, best_metadata, selected_position
        candidate_record["state"] = _candidate_log_state(
            predicted_loss=predicted_loss,
            actual_loss=actual_loss,
            is_trusted=is_trusted,
            generalization=generalization,
            selected_row_ids=selected_row_ids,
        )
        candidates.append(candidate_record)
        if best is None or candidate_record["score"] > best["score"]:
            best = candidate_record
            best_state = state_factory()
            best_metadata = selected_metadata
            selected_position = len(candidates) - 1
            return True
        return False

    def _candidate_state_score(
        *, candidate_type, candidate_id, candidate_state,
        candidate_validation_encode,
    ):
        if candidate_state_scorer is None:
            return None
        candidate_positions = [
            row_id_to_position[row_id]
            for row_id in candidate_state.selected_row_ids
        ]
        candidate_position_tensor = torch.as_tensor(
            candidate_positions,
            device=candidate_state.current_train_tensor.device,
            dtype=torch.long,
        )
        candidate_encode = standardizer.transform(
            candidate_state.current_train_tensor.index_select(
                0,
                candidate_position_tensor,
            )
        )
        score = float(candidate_state_scorer(CandidateStateScoreContext(
            iteration=int(iteration_count),
            candidate_type=candidate_type,
            candidate_id=candidate_id,
            current_encode=current_state_score_encode,
            current_y=current_state_score_y,
            candidate_encode=candidate_encode,
            candidate_y=train_y_tensor.index_select(
                0,
                candidate_position_tensor,
            ),
            current_validation_encode=val_score_encode,
            candidate_validation_encode=candidate_validation_encode,
            validation_y=val_y_tensor,
        )))
        if not math.isfinite(score):
            raise ValueError("candidate_state_scorer must return a finite number.")
        return score

    def _final_selection_score(
        *, candidate_type, candidate_id, utility_gain, next_k,
        candidate_published_min_k, release_result, candidate_state_score,
    ):
        privacy_cost = math.log(max(current_k - next_k, 2))
        context = SelectionScoreContext(
            iteration=int(iteration_count),
            candidate_type=candidate_type,
            candidate_id=candidate_id,
            utility_gain=float(utility_gain),
            current_k=int(current_k),
            next_k=int(next_k),
            privacy_cost=float(privacy_cost),
            candidate_published_min_k=int(candidate_published_min_k),
            release_target_k=int(release_result["target_k"]),
            release_validation_loss=float(
                release_result["validation_loss"]
            ),
            release_row_count=int(release_result["row_count"]),
            release_suppressed_row_count=int(
                release_result["suppressed_row_count"]
            ),
            release_coverage_fraction=float(
                release_result["coverage_fraction"]
            ),
            candidate_state_score=candidate_state_score,
        )
        if selection_score is None:
            score = default_selection_score(context)
        else:
            score = float(selection_score(context))
            if not math.isfinite(score):
                raise ValueError("selection_score must return a finite number.")
        return float(score), float(privacy_cost)

    for candidate in top_retention:
        next_row_ids = selected_ids + candidate["retention_class_index"]
        if current_position_index is not None:
            _lt = time.perf_counter()
            candidate_class_counts = _class_counts_from_position_index(
                candidate["retention_class_index"],
                current_position_index,
                initial_row_id_set,
            )
            next_k = _leakage_from_class_counts(
                current_class_counts + candidate_class_counts)
            leakage_elapsed = time.perf_counter() - _lt
            timing_leakage += leakage_elapsed
            optimized_privacy_time += leakage_elapsed
            next_train_encode = None
        else:
            _lt = time.perf_counter()
            next_train_encode = state.current_train_encode.copy()
            next_train_encode.loc[candidate["retention_class_index"], :] = (
                selected_level_encode.loc[candidate["retention_class_index"], :].to_numpy())
            next_selected_encode = next_train_encode.loc[next_row_ids]
            next_k = (leakage_func(next_selected_encode)
                      if not next_selected_encode.empty else 0)
            timing_leakage += time.perf_counter() - _lt
        leakage_calls += 1
        if current_position_index is not None:
            _, published_stats, _ = _published_leak_snapshot_from_position_index(
                next_row_ids,
                current_position_index,
                initial_row_id_set,
            )
        else:
            _, published_stats, _ = _published_leak_snapshot(
                select_encoded_attributes(
                    next_train_encode,
                    tuple(current_level),
                ),
                next_row_ids,
                initial_row_id_set,
            )
        candidate_published_min_k = int(published_stats["leak_k"])
        _t = time.perf_counter()
        release_result = candidate_release_evaluator(
            current_generalization,
            candidate_published_min_k,
        )
        timing_proxy_retrain += time.perf_counter() - _t
        proxy_retrain_calls += 1
        if release_result.get("status") != "ok":
            continue
        actual_loss = float(release_result["validation_loss"])
        utility_gain = float(current_model_loss - actual_loss)
        if utility_gain <= 0.0:
            continue

        def _retention_state(
            *,
            candidate=candidate,
            next_row_ids=next_row_ids,
            next_train_encode=next_train_encode,
            actual_loss=actual_loss,
        ):
            candidate_train_encode = next_train_encode
            if candidate_train_encode is None:
                candidate_train_encode = state.current_train_encode.copy()
                candidate_train_encode.loc[candidate["retention_class_index"], :] = (
                    selected_level_encode.loc[
                        candidate["retention_class_index"], :
                    ].to_numpy()
                )
            next_train_tensor = state.current_train_tensor.clone()
            next_train_tensor[candidate["row_position_tensor"]] = (
                candidate["next_changed_tensor"]
            )
            return SelectionState(
                predicted_model_loss=actual_loss,
                actual_model_loss=actual_loss,
                is_actual_model_loss_trusted=True,
                current_generalization=current_generalization,
                selected_row_ids=list(next_row_ids),
                current_train_encode=candidate_train_encode,
                current_train_tensor=next_train_tensor,
                action_log_path=state.action_log_path,
            )

        retained_candidate_state = (
            _retention_state()
            if candidate_state_scorer is not None
            else None
        )
        candidate_external_score = _candidate_state_score(
            candidate_type=RETENTION_CLASS_ACTION,
            candidate_id=candidate["retention_class_key"],
            candidate_state=retained_candidate_state,
            candidate_validation_encode=val_score_encode,
        ) if retained_candidate_state is not None else None
        score, privacy_cost = _final_selection_score(
            candidate_type=RETENTION_CLASS_ACTION,
            candidate_id=candidate["retention_class_key"],
            utility_gain=utility_gain,
            next_k=next_k,
            candidate_published_min_k=candidate_published_min_k,
            release_result=release_result,
            candidate_state_score=candidate_external_score,
        )

        _add_scored_candidate({
            "action": "retention_class",
            "retention_class_position": candidate["retention_class_position"],
            "retention_class_key": candidate["retention_class_key"],
            "rank_candidate_id": candidate.get("rank_candidate_id"),
            "lga_score": candidate["lga_score"], "rank_method": "lga",
            "rank_score": candidate["rank_score"], "rank_position": None,
            "in_rank_top_k": True, "exact_rank_score": None,
            "exact_rank_position": None, "in_exact_top_k": None,
            "utility_gain": utility_gain, "current_k": current_k, "next_k": next_k,
            "candidate_published_min_k": candidate_published_min_k,
            "candidate_state_score": candidate_external_score,
            "privacy_cost": privacy_cost, "score": score,
            "release_target_k": int(release_result["target_k"]),
            "release_validation_loss": actual_loss,
            "release_row_count": int(release_result["row_count"]),
            "release_suppressed_row_count": int(
                release_result["suppressed_row_count"]
            ),
            "release_coverage_fraction": float(
                release_result["coverage_fraction"]
            ),
        }, (
            (lambda candidate_state=retained_candidate_state: candidate_state)
            if retained_candidate_state is not None
            else _retention_state
        ), predicted_loss=actual_loss, actual_loss=actual_loss,
            is_trusted=True, generalization=current_generalization,
            selected_row_ids=next_row_ids,
            selected_metadata={
                "position_index": current_position_index,
                "release_result": release_result,
            })

    precomputed_swap_selected_positions = None
    precomputed_swap_initial_mask = None
    if top_vertical:
        precomputed_swap_selected_positions = np.fromiter(
            (row_id_to_position[rid] for rid in selected_ids),
            dtype=np.intp,
            count=len(selected_ids))
        precomputed_swap_initial_mask = np.zeros(len(selected_level_encode), dtype=bool)
        if initial_row_id_set:
            precomputed_initial_positions = [
                row_id_to_position[row_id]
                for row_id in initial_row_id_set
                if row_id in row_id_to_position
            ]
            precomputed_swap_initial_mask[precomputed_initial_positions] = True

    for candidate in top_vertical:
        row_ids = candidate["selected_row_ids"]
        vertical_extra_swap_cap = len(row_ids)
        # Re-encode the selected rows at the new (lower) level.
        next_train_encode = state.current_train_encode.copy()
        if candidate.get("attribute_columns") and candidate.get("attribute_column_positions") is not None:
            attribute_values = (
                candidate["next_changed_tensor"][:, candidate["attribute_column_positions"]]
                .detach()
                .cpu()
                .numpy()
            )
            next_train_encode.loc[row_ids, candidate["attribute_columns"]] = attribute_values
        else:
            next_train_encode.loc[row_ids, :] = candidate["next_generalization"].encode(
                X_train_raw).loc[row_ids, :]
        next_train_tensor = state.current_train_tensor.clone()
        next_train_tensor[candidate["row_position_tensor"]] = candidate["next_changed_tensor"]
        candidate_state = SelectionState(
            predicted_model_loss=current_model_loss,
            actual_model_loss=current_model_loss,
            is_actual_model_loss_trusted=False,
            current_generalization=candidate["next_generalization"],
            selected_row_ids=list(row_ids),
            current_train_encode=next_train_encode,
            current_train_tensor=next_train_tensor,
            action_log_path=state.action_log_path)
        candidate_level_encode = None
        candidate_position_index = None
        candidate_codes = None
        if (
            current_position_index is not None
            and swap_encode_cache is not None
            and candidate.get("attribute") in swap_encode_cache.attribute_columns
            and candidate.get("attribute") in swap_encode_cache.codes.columns
        ):
            _lt = time.perf_counter()
            candidate_level_encode = selected_level_encode.copy()
            candidate_codes = swap_encode_cache.codes.copy()
            expected_columns = swap_encode_cache.attribute_columns[candidate["attribute"]]
            if candidate.get("attribute_columns") == expected_columns:
                candidate_attribute_frame, candidate_attribute_codes = (
                    _swap_attribute_frame_and_code(
                        candidate["next_generalization"], X_train_raw,
                        candidate["attribute"]))
                candidate_level_encode.loc[:, expected_columns] = (
                    candidate_attribute_frame.to_numpy())
                candidate_codes.loc[:, candidate["attribute"]] = candidate_attribute_codes
                candidate_position_index = _swap_position_index_from_cache(
                    _SwapEncodeCache(
                        encoded=candidate_level_encode,
                        codes=candidate_codes,
                        row_ids=swap_encode_cache.row_ids,
                        row_id_to_position=swap_encode_cache.row_id_to_position,
                        selected_attributes=swap_encode_cache.selected_attributes,
                        attribute_columns=dict(swap_encode_cache.attribute_columns),
                    ))
                leakage_elapsed = time.perf_counter() - _lt
                timing_leakage += leakage_elapsed
                optimized_privacy_time += leakage_elapsed
            else:
                timing_leakage += time.perf_counter() - _lt
        swap_stats = _apply_vertical_postprocessing_inplace(
            candidate_state, X_train_raw,
            (swap_encode_cache.selected_attributes if swap_encode_cache is not None else None),
            policy=vertical_postprocessing,
            move_out_threshold=move_out_threshold,
            swap_level_encode=candidate_level_encode,
            swap_position_index=candidate_position_index,
            swap_codes=candidate_codes,
            initial_row_id_set=initial_row_id_set,
            precomputed_selected_positions=precomputed_swap_selected_positions,
            precomputed_initial_mask=precomputed_swap_initial_mask)
        candidate_position_index = (
            candidate_position_index
            if candidate_position_index is not None
            else swap_stats.get("swap_position_index")
        )
        swap_stats["extra_swap_in_count"] = 0
        if (
            vertical_postprocessing == FULL_SWAP_POSTPROCESSING
            and candidate_position_index is not None
            and vertical_extra_swap_cap > 0
        ):
            swap_row_ids = candidate_position_index["row_ids"]
            swap_row_class_ids = candidate_position_index["row_class_ids"]
            swap_class_member_positions = candidate_position_index["class_member_positions"]
            swap_class_total_sizes = candidate_position_index["class_total_sizes"]
            swap_selected_positions = np.fromiter(
                (
                    candidate_position_index["row_id_to_position"][rid]
                    for rid in candidate_state.selected_row_ids
                ),
                dtype=np.intp,
                count=len(candidate_state.selected_row_ids),
            )
            swap_selected_mask = np.zeros(len(swap_row_ids), dtype=bool)
            swap_selected_mask[swap_selected_positions] = True
            if len(swap_class_total_sizes) > 0:
                swap_selected_sizes = np.bincount(
                    swap_row_class_ids[swap_selected_positions],
                    minlength=len(swap_class_total_sizes),
                )
                swap_candidate_sizes = swap_class_total_sizes - swap_selected_sizes
                swap_extra_class_id = int(np.argmax(swap_candidate_sizes))
                swap_extra_candidate_size = int(swap_candidate_sizes[swap_extra_class_id])
                if swap_extra_candidate_size > vertical_extra_swap_cap:
                    swap_extra_class_positions = swap_class_member_positions[
                        swap_extra_class_id]
                    swap_extra_move_in_positions = swap_extra_class_positions[
                        ~swap_selected_mask[swap_extra_class_positions]
                    ][:vertical_extra_swap_cap]
                    if len(swap_extra_move_in_positions):
                        candidate_state.selected_row_ids = (
                            candidate_state.selected_row_ids
                            + swap_row_ids[swap_extra_move_in_positions].tolist()
                        )
                        swap_sync_positions = swap_extra_move_in_positions.tolist()
                        swap_level_encode = swap_stats["swap_level_encode"]
                        candidate_state.current_train_encode.iloc[
                            swap_sync_positions, :
                        ] = swap_level_encode.iloc[
                            swap_sync_positions, :
                        ].to_numpy()
                        candidate_state.current_train_tensor[
                            swap_sync_positions
                        ] = to_device_tensor(
                            swap_level_encode.iloc[swap_sync_positions, :],
                            device=candidate_state.current_train_tensor.device,
                            dtype=candidate_state.current_train_tensor.dtype)
                        swap_stats["extra_swap_in_count"] = int(
                            len(swap_extra_move_in_positions))
                        swap_stats["move_in_row_count"] = int(
                            swap_stats.get("move_in_row_count", 0)
                            + len(swap_extra_move_in_positions)
                        )
                        swap_stats["selected_row_count"] = len(
                            candidate_state.selected_row_ids)
                        swap_stats["candidate_row_count"] = (
                            len(swap_row_ids) - len(candidate_state.selected_row_ids))
        swapped_row_ids = candidate_state.selected_row_ids
        _lt = time.perf_counter()
        if candidate_position_index is not None:
            next_k = _leakage_from_position_index(
                swapped_row_ids, candidate_position_index, initial_row_id_set)
            leakage_elapsed = time.perf_counter() - _lt
            timing_leakage += leakage_elapsed
            if current_position_index is not None:
                optimized_privacy_time += leakage_elapsed
        else:
            next_selected_encode = candidate_state.current_train_encode.loc[swapped_row_ids]
            next_k = (leakage_func(next_selected_encode)
                      if not next_selected_encode.empty else 0)
            timing_leakage += time.perf_counter() - _lt
        leakage_calls += 1
        if candidate_position_index is not None:
            _, published_stats, _ = _published_leak_snapshot_from_position_index(
                swapped_row_ids,
                candidate_position_index,
                initial_row_id_set,
            )
        else:
            _, published_stats, _ = _published_leak_snapshot(
                select_encoded_attributes(
                    candidate_state.current_train_encode,
                    tuple(current_level),
                ),
                swapped_row_ids,
                initial_row_id_set,
            )
        candidate_published_min_k = int(published_stats["leak_k"])
        _t = time.perf_counter()
        release_result = candidate_release_evaluator(
            candidate["next_generalization"],
            candidate_published_min_k,
        )
        timing_proxy_retrain += time.perf_counter() - _t
        proxy_retrain_calls += 1
        if release_result.get("status") != "ok":
            del candidate_state
            del next_train_tensor
            continue
        actual_loss = float(release_result["validation_loss"])
        utility_gain = float(current_model_loss - actual_loss)
        if utility_gain <= 0.0:
            del candidate_state
            del next_train_tensor
            continue
        candidate_external_score = _candidate_state_score(
            candidate_type=VERTICAL_REFINEMENT_ACTION,
            candidate_id=candidate["attribute"],
            candidate_state=candidate_state,
            candidate_validation_encode=candidate["next_val_score_encode"],
        )
        score, privacy_cost = _final_selection_score(
            candidate_type=VERTICAL_REFINEMENT_ACTION,
            candidate_id=candidate["attribute"],
            utility_gain=utility_gain,
            next_k=next_k,
            candidate_published_min_k=candidate_published_min_k,
            release_result=release_result,
            candidate_state_score=candidate_external_score,
        )
        candidate_state.predicted_model_loss = actual_loss
        candidate_state.actual_model_loss = actual_loss
        candidate_state.is_actual_model_loss_trusted = True
        kept_candidate_state = _add_scored_candidate({
            "action": "vertical_refinement",
            "attribute": candidate["attribute"],
            "retention_class_position": None,
            "rank_candidate_id": candidate.get("rank_candidate_id"),
            "lga_score": candidate["lga_score"], "rank_method": "lga",
            "rank_score": candidate["rank_score"], "rank_position": None,
            "in_rank_top_k": True, "exact_rank_score": None,
            "exact_rank_position": None, "in_exact_top_k": None,
            "utility_gain": utility_gain, "current_k": current_k, "next_k": next_k,
            "candidate_published_min_k": candidate_published_min_k,
            "candidate_state_score": candidate_external_score,
            "privacy_cost": privacy_cost, "score": score,
            "swap_stats": swap_stats,
            "release_target_k": int(release_result["target_k"]),
            "release_validation_loss": actual_loss,
            "release_row_count": int(release_result["row_count"]),
            "release_suppressed_row_count": int(
                release_result["suppressed_row_count"]
            ),
            "release_coverage_fraction": float(
                release_result["coverage_fraction"]
            ),
        }, lambda candidate_state=candidate_state: candidate_state,
            predicted_loss=actual_loss, actual_loss=actual_loss,
            is_trusted=True, generalization=candidate["next_generalization"],
            selected_row_ids=swapped_row_ids,
            selected_metadata={
                "position_index": candidate_position_index,
                "swap_level_encode": candidate_level_encode,
                "swap_codes": candidate_codes,
                "release_result": release_result,
            })
        if not kept_candidate_state:
            del candidate_state
            del next_train_tensor

    if candidate_observer is not None:
        observer_started_at = time.perf_counter()
        try:
            for position, candidate in enumerate(candidates):
                candidate_observer(CandidateObservation(
                    iteration=int(iteration_count),
                    candidate_type=candidate["action"],
                    candidate_id=candidate.get("rank_candidate_id"),
                    rank_score=float(candidate["rank_score"]),
                    selection_score=float(candidate["score"]),
                    candidate_state_score=(
                        float(candidate["candidate_state_score"])
                        if candidate.get("candidate_state_score") is not None
                        else None
                    ),
                    utility_gain=float(candidate["utility_gain"]),
                    current_k=int(candidate["current_k"]),
                    next_k=int(candidate["next_k"]),
                    privacy_cost=float(candidate["privacy_cost"]),
                    candidate_published_min_k=int(
                        candidate["candidate_published_min_k"]
                    ),
                    release_target_k=int(candidate["release_target_k"]),
                    release_validation_loss=float(
                        candidate["release_validation_loss"]
                    ),
                    release_row_count=int(candidate["release_row_count"]),
                    release_suppressed_row_count=int(
                        candidate["release_suppressed_row_count"]
                    ),
                    release_coverage_fraction=float(
                        candidate["release_coverage_fraction"]
                    ),
                    selected=position == selected_position,
                ))
        finally:
            candidate_observer_time += time.perf_counter() - observer_started_at

    # Persist this call's timing breakdown alongside the selection records.
    if action_log_path is not None:
        with Path(action_log_path).open("a", encoding="utf-8") as _log:
            _log.write(json.dumps({
                "event": "greedy_selection_breakdown",
                "rank_method": "lga", "rank_top_k": rank_top_k,
                "reference_model_filtering_enabled": bool(
                    reference_model_filtering
                ),
                "reference_filter_time_seconds": timing_reference_filter,
                "reference_filter_remaining_row_count": (
                    reference_filter_remaining_row_count
                ),
                "reference_filter_reference_correct_row_count": (
                    reference_filter_reference_correct_row_count
                ),
                "reference_filter_qualified_row_count": (
                    reference_filter_qualified_row_count
                ),
                "reference_filter_retention_class_count": (
                    reference_filter_retention_class_count
                ),
                "ranking_time_seconds": timing_rank,
                "lga_time_seconds": timing_lga, "lga_calls": lga_calls,
                "lga_candidate_evaluations": lga_candidate_evaluations,
                "lga_batched": bool(retention_candidates),
                "proxy_retrain_time_seconds": timing_proxy_retrain,
                "proxy_retrain_calls": proxy_retrain_calls,
                "leakage_time_seconds": timing_leakage, "leakage_calls": leakage_calls,
                "candidate_count": len(candidates),
                "candidate_observer_time_seconds": candidate_observer_time,
            }, default=str, sort_keys=True))
            _log.write("\n")

    # Observer time is reported separately. The caller additionally subtracts
    # privacy time credited by leakage_func and the optimized class-id path.
    selection_timing["selection"] = max(
        0.0,
        time.perf_counter() - t_start - candidate_observer_time,
    )
    selection_timing["reference_filter"] = timing_reference_filter
    selection_timing["lga"] = timing_lga
    selection_timing["proxy_retrain"] = timing_proxy_retrain
    selection_timing["leakage"] = timing_leakage
    selection_timing["ranking"] = timing_rank
    selection_timing["optimized_privacy"] = optimized_privacy_time
    selection_timing["candidate_observer"] = candidate_observer_time

    if not candidates:
        return {"action": None, "state": state, "candidates": [],
                "selected_candidate_position": None}

    selected_action = {
        **best, "state": best_state, "candidates": candidates,
        "selected_candidate_position": selected_position,
    }
    if best_metadata:
        selected_action.update(best_metadata)
    return selected_action


def _post_vertical_swap(state, X_train_raw, selected_attributes,
                        selected_by_action_row_ids, initial_row_id_set,
                        action_log_path, iteration_count, selected_action,
                        swap_encode_cache=None):
    """Restore pre-refinement min-K by exchanging complete retention classes.

    Replace the smallest selected class with the largest available class until
    the selected minimum is at least the largest remaining class.
    """
    swap_t0 = time.perf_counter()
    swap_iterations = 0
    swap_break_reason = "max_iterations"
    swap_encode_t0 = time.perf_counter()
    if swap_encode_cache is None:
        swap_level_encode = state.current_generalization.encode(X_train_raw)
    else:
        swap_level_encode = _update_swap_encode_cache(
            swap_encode_cache,
            state.current_generalization,
            X_train_raw,
            selected_action.get("attribute"),
        )
    swap_encode_time = time.perf_counter() - swap_encode_t0
    swap_position_index_t0 = time.perf_counter()
    if swap_encode_cache is None:
        swap_position_index = build_retention_class_position_index(
            swap_level_encode, selected_attributes=selected_attributes, sort_keys=False)
    else:
        swap_position_index = _swap_position_index_from_cache(swap_encode_cache)
    swap_position_index_time = time.perf_counter() - swap_position_index_t0
    swap_row_ids = swap_position_index["row_ids"]
    swap_row_id_to_position = swap_position_index["row_id_to_position"]
    swap_row_class_ids = swap_position_index["row_class_ids"]
    swap_class_member_positions = swap_position_index["class_member_positions"]
    swap_class_total_sizes = swap_position_index["class_total_sizes"]
    swap_class_count = len(swap_class_total_sizes)
    swap_selected_positions = np.fromiter(
        (swap_row_id_to_position[rid] for rid in state.selected_row_ids),
        dtype=np.intp, count=len(state.selected_row_ids))
    swap_selected_mask = np.zeros(len(swap_row_ids), dtype=bool)
    swap_selected_mask[swap_selected_positions] = True
    swap_sync_time = 0.0
    swap_loop_t0 = time.perf_counter()
    for _ in range(VERTICAL_SWAP_MAX_ITERATIONS):
        if len(swap_selected_positions) >= len(swap_row_ids):
            swap_break_reason = "no_candidates"
            break
        if len(swap_selected_positions) == 0 or swap_class_count == 0:
            swap_break_reason = "no_selected_groups"
            break
        swap_selected_class_ordered = swap_row_class_ids[swap_selected_positions]
        swap_selected_sizes = np.bincount(swap_selected_class_ordered, minlength=swap_class_count)
        swap_selected_class_ids, swap_selected_first_indices = np.unique(
            swap_selected_class_ordered, return_index=True)
        swap_selected_order = swap_selected_class_ids[np.argsort(swap_selected_first_indices)]
        swap_candidate_positions = np.flatnonzero(~swap_selected_mask)
        if len(swap_candidate_positions) == 0:
            swap_break_reason = "no_candidates"
            break
        swap_candidate_class_ordered = swap_row_class_ids[swap_candidate_positions]
        swap_candidate_class_ids, swap_candidate_first_indices = np.unique(
            swap_candidate_class_ordered, return_index=True)
        swap_candidate_order = swap_candidate_class_ids[np.argsort(swap_candidate_first_indices)]
        if len(swap_candidate_order) == 0:
            swap_break_reason = "no_candidate_groups"
            break
        swap_candidate_sizes = swap_class_total_sizes - swap_selected_sizes
        a = int(swap_selected_sizes[swap_selected_order].min())
        b = -1
        b_class_id = None
        for swap_class_id in swap_candidate_order:
            swap_candidate_size = int(swap_candidate_sizes[swap_class_id])
            if swap_candidate_size > b:
                b = swap_candidate_size
                b_class_id = int(swap_class_id)
        if not (a < b):
            swap_break_reason = "converged"
            break
        swap_chosen_class_ids = []
        swap_move_out_total = 0
        for swap_class_id in sorted(swap_selected_order.tolist(),
                                    key=lambda cid: int(swap_selected_sizes[cid])):
            swap_size = int(swap_selected_sizes[swap_class_id])
            if swap_move_out_total + swap_size <= b:
                swap_chosen_class_ids.append(swap_class_id)
                swap_move_out_total += swap_size
            else:
                break
        if not swap_chosen_class_ids:
            swap_break_reason = "smallest_selected_exceeds_candidate"
            break
        swap_move_out_parts = []
        for swap_class_id in swap_chosen_class_ids:
            swap_class_positions = swap_class_member_positions[swap_class_id]
            swap_class_selected_positions = swap_class_positions[
                swap_selected_mask[swap_class_positions]]
            if len(swap_class_selected_positions):
                swap_move_out_parts.append(swap_class_selected_positions)
        if not swap_move_out_parts:
            swap_break_reason = "no_move_out"
            break
        swap_move_out_positions = np.concatenate(swap_move_out_parts)
        swap_candidate_class_positions = swap_class_member_positions[b_class_id]
        swap_move_in_positions = swap_candidate_class_positions[
            ~swap_selected_mask[swap_candidate_class_positions]]
        if len(swap_move_in_positions) == 0:
            swap_break_reason = "no_move_in"
            break
        swap_move_out_mask = np.zeros(len(swap_row_ids), dtype=bool)
        swap_move_out_mask[swap_move_out_positions] = True
        swap_selected_positions = np.concatenate([
            swap_selected_positions[~swap_move_out_mask[swap_selected_positions]],
            swap_move_in_positions])
        swap_selected_mask[swap_move_out_positions] = False
        swap_selected_mask[swap_move_in_positions] = True
        swap_sync_t0 = time.perf_counter()
        swap_move_out = set(swap_row_ids[swap_move_out_positions].tolist())
        swap_move_in = swap_row_ids[swap_move_in_positions].tolist()
        state.selected_row_ids = swap_row_ids[swap_selected_positions].tolist()
        selected_by_action_row_ids.difference_update(swap_move_out)
        selected_by_action_row_ids.update(
            rid for rid in swap_move_in if rid not in initial_row_id_set)
        swap_sync_positions = swap_move_in_positions.tolist()
        state.current_train_encode.iloc[swap_sync_positions, :] = (
            swap_level_encode.iloc[swap_sync_positions, :].to_numpy())
        state.current_train_tensor[swap_sync_positions] = to_device_tensor(
            swap_level_encode.iloc[swap_sync_positions, :],
            device=state.current_train_tensor.device,
            dtype=state.current_train_tensor.dtype)
        swap_sync_time += time.perf_counter() - swap_sync_t0
        swap_iterations += 1
    swap_loop_time = max(0.0, time.perf_counter() - swap_loop_t0 - swap_sync_time)
    if action_log_path is not None:
        with Path(action_log_path).open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps({
                "event": "swap_timing", "iteration": iteration_count,
                "action": selected_action["action"],
                "attribute": selected_action.get("attribute"),
                "swap_iterations": swap_iterations,
                "elapsed_seconds": time.perf_counter() - swap_t0,
                "encode_time_seconds": swap_encode_time,
                "position_index_time_seconds": swap_position_index_time,
                "loop_time_seconds": swap_loop_time,
                "sync_time_seconds": swap_sync_time,
                "break_reason": swap_break_reason,
                "selected_row_count": len(state.selected_row_ids),
                "candidate_row_count": len(swap_row_ids) - len(state.selected_row_ids),
            }, default=str, sort_keys=True))
            log_file.write("\n")
    return swap_level_encode
