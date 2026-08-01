"""Classification-only, greedy-ratio TRIM removal view.

The removal view starts from the complete level-0 training publication. Its
active and removed row IDs always form an ordered partition of the original
training rows. It permits only three information-decreasing actions:

* remove one complete active Retention Class;
* coarsen one attribute by exactly one hierarchy level;
* exchange one complete active class for one complete removed-side class when
  active row count does not increase and min-K does not decrease.

Candidates with positive augmented privacy progress are ranked by their
estimated normalized utility cost. The first feasible top-K batch is retrained
with the proxy model, then the candidate with the largest exact ratio of
privacy progress to normalized utility cost is selected. Validation utility
remains a hard feasibility gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from prototype.backend_training import PassthroughStandardizer, fit_standardizer
from prototype.dataloader import load_generalization_rules_from_file
from prototype.enumeration_horizontal import _retention_group_columns
from prototype.gpu_logistic import TorchLogisticRegression
from prototype.gpu_math import (
    _class_indices_tensor,
    classification_log_loss_tensor,
    logistic_gradient,
    model_theta_tensor,
    multiclass_logistic_gradient,
    to_device_tensor,
)
from prototype.greedy_selection import _load_max_generalization_level

from ..common import append_jsonl, write_json


DEFAULT_MAX_ITERATIONS = 50
DEFAULT_RANK_TOP_K = 5
DEFAULT_EXCHANGE_POOL_SIZE = 12
DEFAULT_GREEDY_RATIO_EPSILON = 1e-6
SELECTION_RULE = "utility_cost_ranked_greedy_ratio"


@dataclass
class RemovalViewState:
    """One materialized removal-view state."""

    active_row_ids: list[Any] = field(default_factory=list)
    removed_row_ids: list[Any] = field(default_factory=list)
    generalization_level: dict[str, int] = field(default_factory=dict)
    current_generalization: Any | None = None
    proxy_val_loss: float | None = None
    privacy_stats: dict[str, Any] = field(default_factory=dict)
    iteration: int = 0


@dataclass
class RemovalViewResult:
    task_type: str = "classification"
    loss_metric: str = "log_loss"
    selection_rule: str = SELECTION_RULE
    selection_rule_config: dict[str, Any] = field(default_factory=dict)
    active_row_ids: list[Any] = field(default_factory=list)
    removed_row_ids: list[Any] = field(default_factory=list)
    selected_row_ids: list[Any] = field(default_factory=list)
    generalization_level: dict[str, int] = field(default_factory=dict)
    state: RemovalViewState | None = None
    termination_condition: str | None = None
    iteration_count: int = 0
    exchange_count: int = 0
    baseline_proxy_val_loss: float | None = None
    proxy_loss_threshold_val: float | None = None
    final_proxy_val_loss: float | None = None
    baseline_val_loss: float | None = None
    loss_threshold_val: float | None = None
    final_actual_val_loss: float | None = None
    validation_utility_constraint_met: bool | None = None
    baseline_test_loss: float | None = None
    loss_threshold_test: float | None = None
    final_actual_model_loss: float | None = None
    utility_constraint_met: bool | None = None
    original_leak_k: int | None = None
    original_bottleneck_count: int | None = None
    final_leak_k: int | None = None
    final_bottleneck_count: int | None = None
    leak_k_p1: float | None = None
    leak_k_p5: float | None = None
    final_leak_k_sorted: list[int] = field(default_factory=list)
    final_leak_k_95_percentile: int | None = None
    action_history: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    run_dir: str | None = None


def _empty_privacy_stats() -> dict[str, Any]:
    return {
        "leak_k": 0,
        "min_k_class_count": 0,
        "class_count": 0,
        "leak_k_p1": 0.0,
        "leak_k_p2": 0.0,
        "leak_k_p3": 0.0,
        "leak_k_p4": 0.0,
        "leak_k_p5": 0.0,
        "leak_k_95_percentile": None,
        "leak_k_sorted": [],
    }


def _privacy_stats_from_group_sizes(
    group_sizes: Iterable[int],
    *,
    include_distribution: bool = False,
) -> dict[str, Any]:
    sizes = np.asarray(list(group_sizes), dtype=np.int64).reshape(-1)
    sizes = sizes[sizes > 0]
    if sizes.size == 0:
        return _empty_privacy_stats()
    unique_sizes, class_counts = np.unique(sizes, return_counts=True)
    return _privacy_stats_from_size_counts(
        dict(zip(unique_sizes.tolist(), class_counts.tolist())),
        include_distribution=include_distribution,
    )


def _privacy_stats_from_size_counts(
    size_counts: dict[int, int],
    *,
    include_distribution: bool = False,
) -> dict[str, Any]:
    positive_items = sorted(
        (int(size), int(count))
        for size, count in size_counts.items()
        if int(size) > 0 and int(count) > 0
    )
    if not positive_items:
        return _empty_privacy_stats()

    sizes = np.asarray([size for size, _count in positive_items], dtype=np.int64)
    class_counts = np.asarray(
        [count for _size, count in positive_items],
        dtype=np.int64,
    )
    cumulative = np.cumsum(sizes * class_counts)
    record_count = int(cumulative[-1])

    def value_at_record_position(position: int) -> float:
        class_position = int(
            np.searchsorted(cumulative, position + 1, side="left")
        )
        return float(sizes[class_position])

    def weighted_percentile(percentile: float) -> float:
        location = (record_count - 1) * (float(percentile) / 100.0)
        lower = int(math.floor(location))
        upper = int(math.ceil(location))
        lower_value = value_at_record_position(lower)
        upper_value = value_at_record_position(upper)
        return lower_value + (location - lower) * (upper_value - lower_value)

    minimum = int(sizes[0])
    stats = {
        "leak_k": minimum,
        "min_k_class_count": int(class_counts[0]),
        "class_count": int(class_counts.sum()),
        "leak_k_p1": weighted_percentile(1),
        "leak_k_p2": weighted_percentile(2),
        "leak_k_p3": weighted_percentile(3),
        "leak_k_p4": weighted_percentile(4),
        "leak_k_p5": weighted_percentile(5),
        "leak_k_95_percentile": int(
            value_at_record_position(math.ceil(0.95 * record_count) - 1)
        ),
        "leak_k_sorted": [],
    }
    if include_distribution:
        stats["leak_k_sorted"] = np.repeat(
            sizes,
            sizes * class_counts,
        ).astype(int).tolist()
    return stats


def _privacy_signature(stats: dict[str, Any]) -> tuple[int, int]:
    return (
        int(stats.get("leak_k") or 0),
        -int(stats.get("min_k_class_count") or 0),
    )


def _augmented_privacy_progress(
    current_privacy_stats: dict[str, Any],
    next_privacy_stats: dict[str, Any],
) -> float:
    current_k = int(current_privacy_stats.get("leak_k") or 0)
    next_k = int(next_privacy_stats.get("leak_k") or 0)
    if next_k < current_k:
        raise ValueError("A removal-view candidate must not decrease min-K.")
    if next_k > current_k:
        return float(next_k - current_k)

    current_bottlenecks = int(
        current_privacy_stats.get("min_k_class_count") or 0
    )
    next_bottlenecks = int(
        next_privacy_stats.get("min_k_class_count") or 0
    )
    if current_bottlenecks <= 0:
        raise ValueError(
            "Current min-K bottleneck count must be positive when min-K "
            "is unchanged."
        )
    return float(
        (current_bottlenecks - next_bottlenecks) / current_bottlenecks
    )


def _normalized_utility_cost(
    utility_loss_delta: float,
    *,
    tolerance: float,
    epsilon: float,
) -> float | None:
    if not math.isfinite(float(utility_loss_delta)):
        return None
    tolerance_scale = max(float(tolerance), float(epsilon))
    return max(
        float(utility_loss_delta) / tolerance_scale,
        float(epsilon),
    )


def _ordered_partition(
    all_row_ids: Sequence[Any],
    active_row_ids: Iterable[Any],
    removed_row_ids: Iterable[Any],
) -> tuple[list[Any], list[Any]]:
    active_set = set(active_row_ids)
    removed_set = set(removed_row_ids)
    full_set = set(all_row_ids)
    if active_set & removed_set:
        raise ValueError("active_row_ids and removed_row_ids must be disjoint.")
    if active_set | removed_set != full_set:
        raise ValueError(
            "active_row_ids and removed_row_ids must partition all training rows."
        )
    return (
        [row_id for row_id in all_row_ids if row_id in active_set],
        [row_id for row_id in all_row_ids if row_id in removed_set],
    )


def _apply_membership_action(
    *,
    all_row_ids: Sequence[Any],
    active_row_ids: Sequence[Any],
    removed_row_ids: Sequence[Any],
    remove_row_ids: Iterable[Any] = (),
    add_row_ids: Iterable[Any] = (),
) -> tuple[list[Any], list[Any]]:
    active_set = set(active_row_ids)
    removed_set = set(removed_row_ids)
    remove_set = set(remove_row_ids)
    add_set = set(add_row_ids)
    if remove_set & add_set:
        raise ValueError("An action cannot remove and add the same row.")
    if not remove_set <= active_set:
        raise ValueError("remove_row_ids must be a subset of the active rows.")
    if not add_set <= removed_set:
        raise ValueError("add_row_ids must be a subset of the removed rows.")
    next_active = (active_set - remove_set) | add_set
    next_removed = (removed_set - add_set) | remove_set
    if not next_active:
        raise ValueError("An action cannot produce an empty active release.")
    return _ordered_partition(all_row_ids, next_active, next_removed)


def _exchange_rejection_reason(
    *,
    out_row_ids: Iterable[Any],
    in_row_ids: Iterable[Any],
    active_class_row_ids: Iterable[Any],
    removed_class_row_ids: Iterable[Any],
    current_min_k: int,
    next_min_k: int,
    candidate_loss: float | None = None,
    utility_threshold: float | None = None,
) -> str | None:
    out_set = set(out_row_ids)
    in_set = set(in_row_ids)
    if out_set != set(active_class_row_ids):
        return "incomplete_out_class"
    if in_set != set(removed_class_row_ids):
        return "incomplete_in_class"
    if not out_set:
        return "empty_out_class"
    if not in_set:
        return "empty_in_class"
    if len(in_set) > len(out_set):
        return "row_count_increase"
    if int(next_min_k) < int(current_min_k):
        return "min_k_decrease"
    if (
        candidate_loss is not None
        and utility_threshold is not None
        and (
            not math.isfinite(float(candidate_loss))
            or float(candidate_loss) > float(utility_threshold)
        )
    ):
        return "utility_threshold"
    return None


def _xgboost_model_input(generalization, raw_rows):
    if hasattr(generalization, "encode_xgboost_leaf_space"):
        encoded = generalization.encode_xgboost_leaf_space(raw_rows)
        try:
            from scipy import sparse
        except ImportError:
            sparse = None
        if sparse is not None and sparse.issparse(encoded):
            return encoded.tocsr().astype(np.float32, copy=False)
        return np.ascontiguousarray(np.asarray(encoded), dtype=np.float32)
    if hasattr(generalization, "encode_xgboost_model_input"):
        return generalization.encode_xgboost_model_input(raw_rows)
    return generalization.encode(raw_rows)


def _factory_family(factory) -> str | None:
    family = getattr(factory, "model_family", None)
    if family is not None:
        return str(family)
    probe = factory()
    return {
        "XGBoostGPUClassifier": "xgboost",
        "TorchLogisticRegression": "torch_logistic",
        "TorchMLP": "mlp",
    }.get(probe.__class__.__name__)


def _group_row_ids(encoded, selected_attributes) -> dict[Any, list[Any]]:
    group_columns = _retention_group_columns(
        encoded,
        selected_attributes=selected_attributes,
    )
    if not group_columns:
        return {(): encoded.index.tolist()}
    grouped = encoded.groupby(group_columns, sort=True, dropna=False)
    return {
        key if isinstance(key, tuple) else (key,): encoded.index.take(
            np.asarray(positions, dtype=np.intp)
        ).tolist()
        for key, positions in grouped.indices.items()
    }


def _candidate_fingerprint(
    active_row_ids: Iterable[Any],
    generalization_level: dict[str, int],
):
    return (
        frozenset(active_row_ids),
        tuple(
            sorted(
                (str(key), int(value))
                for key, value in generalization_level.items()
            )
        ),
    )


def _public_candidate_record(candidate: dict[str, Any]) -> dict[str, Any]:
    record = {
        key: value
        for key, value in candidate.items()
        if not key.startswith("_")
    }
    record.pop("privacy_stats", None)
    record["next_privacy_stats"] = candidate["privacy_stats"]
    return record


def run_removal_view(
    data_loader,
    generalization_tree_path,
    *,
    results_dir,
    run_id=None,
    tolerance=0.01,
    nrows=None,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    max_iterations=DEFAULT_MAX_ITERATIONS,
    model_max_iter=300,
    rank_top_k=DEFAULT_RANK_TOP_K,
    exchange_pool_size=DEFAULT_EXCHANGE_POOL_SIZE,
    enable_exchange=True,
    greedy_ratio_epsilon=DEFAULT_GREEDY_RATIO_EPSILON,
    model_factory=None,
    proxy_model_factory=None,
    estimator_model_factory=None,
    selected_attributes=None,
    standardizer="auto",
    device="cuda",
    dtype="float32",
    allow_cpu_fallback=True,
):
    """Run the appendix removal-view algorithm and persist raw outputs."""

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle as shuffle_rows

    pipeline_started_at = time.perf_counter()
    if float(tolerance) < 0:
        raise ValueError("tolerance must be non-negative.")
    if not 0.0 < float(val_size) < 1.0:
        raise ValueError("val_size must be between 0 and 1.")
    if not 0.0 < float(test_size) < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if float(val_size) + float(test_size) >= 1.0:
        raise ValueError("val_size + test_size must be less than 1.")
    if max_iterations is not None and int(max_iterations) < 0:
        raise ValueError("max_iterations must be non-negative or None.")
    if rank_top_k is not None and int(rank_top_k) < 1:
        raise ValueError("rank_top_k must be positive or None.")
    if exchange_pool_size is not None and int(exchange_pool_size) < 1:
        raise ValueError("exchange_pool_size must be positive or None.")
    if not isinstance(enable_exchange, bool):
        raise TypeError("enable_exchange must be bool.")
    greedy_ratio_epsilon = float(greedy_ratio_epsilon)
    if not math.isfinite(greedy_ratio_epsilon) or greedy_ratio_epsilon <= 0:
        raise ValueError("greedy_ratio_epsilon must be finite and positive.")
    selection_rule_config = {
        "privacy_progress_when_k_increases": "next_k-current_k",
        "privacy_progress_when_k_is_equal": (
            "(current_bottleneck_count-next_bottleneck_count)"
            "/current_bottleneck_count"
        ),
        "fast_ranking": "normalized_fast_utility_cost_ascending",
        "normalized_utility_cost": (
            "max((next_loss-current_loss)"
            "/max(tolerance,epsilon),epsilon)"
        ),
        "exact_selection": "largest_privacy_progress_over_exact_cost",
        "finite_rank_top_k_semantics": (
            "first_feasible_utility_cost_ranked_batch"
        ),
        "epsilon": greedy_ratio_epsilon,
    }
    for name, factory in (
        ("model_factory", model_factory),
        ("proxy_model_factory", proxy_model_factory),
        ("estimator_model_factory", estimator_model_factory),
    ):
        if factory is not None and not callable(factory):
            raise TypeError(f"{name} must be a zero-argument callable or None.")
    if getattr(data_loader, "task_type", "classification") != "classification":
        raise ValueError("The TRIM removal view supports classification only.")

    run_dir = Path(results_dir).expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(f"Removal-view run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    action_log_path = run_dir / "removal_actions.jsonl"

    X_raw, y_raw = data_loader.load(nrows=nrows)
    if not isinstance(X_raw, pd.DataFrame):
        X_raw = pd.DataFrame(
            X_raw,
            columns=getattr(data_loader, "feature_columns", None),
        )
    if not X_raw.index.is_unique:
        raise ValueError("The data loader must provide unique row indexes.")
    if isinstance(y_raw, pd.Series):
        if not y_raw.index.equals(X_raw.index):
            raise ValueError("X and y indexes must have the same order.")
        y_series = y_raw.copy()
    else:
        y_values = np.asarray(y_raw).reshape(-1)
        if len(y_values) != len(X_raw):
            raise ValueError("X and y must contain the same number of rows.")
        y_series = pd.Series(y_values, index=X_raw.index)
    if bool(y_series.isna().any()):
        raise ValueError("Target values must not contain missing entries.")

    X_raw, y_series = shuffle_rows(X_raw, y_series, random_state=random_state)
    if not X_raw.index.equals(y_series.index):
        raise RuntimeError("Shuffling changed X/y row alignment.")
    original_class_values = np.unique(np.asarray(y_series))
    if len(original_class_values) < 2:
        raise ValueError("Classification requires at least two classes.")
    class_to_index = {
        value.item() if hasattr(value, "item") else value: position
        for position, value in enumerate(original_class_values.tolist())
    }

    X_trainval_raw, X_test_raw, y_trainval_raw, y_test_raw = train_test_split(
        X_raw,
        y_series,
        test_size=test_size,
        random_state=random_state,
        stratify=y_series,
    )
    val_fraction = val_size / (1.0 - test_size)
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_trainval_raw,
        y_trainval_raw,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y_trainval_raw,
    )
    for split_name, split_X, split_y in (
        ("train", X_train_raw, y_train_raw),
        ("validation", X_val_raw, y_val_raw),
        ("test", X_test_raw, y_test_raw),
    ):
        if not split_X.index.equals(split_y.index):
            raise RuntimeError(f"{split_name} X/y indexes are not aligned.")

    def encoded_labels(values) -> np.ndarray:
        return np.asarray(
            [
                class_to_index[
                    value.item() if hasattr(value, "item") else value
                ]
                for value in np.asarray(values).reshape(-1).tolist()
            ],
            dtype=np.int64,
        )

    y_train = encoded_labels(y_train_raw)
    y_val = encoded_labels(y_val_raw)
    y_test = encoded_labels(y_test_raw)
    classification_classes = np.arange(
        len(original_class_values),
        dtype=np.int64,
    )
    class_count = int(len(classification_classes))
    use_multiclass = class_count > 2

    feature_attributes = tuple(getattr(data_loader, "feature_columns", ()))
    if not feature_attributes:
        feature_attributes = tuple(X_train_raw.columns)
    if selected_attributes is None:
        selected_attributes = getattr(data_loader, "qi_attributes", None)
    if selected_attributes is None:
        selected_attributes = feature_attributes
    selected_attributes = tuple(selected_attributes)
    unknown_attributes = sorted(set(selected_attributes) - set(feature_attributes))
    if unknown_attributes:
        raise ValueError(f"Unknown selected_attributes: {unknown_attributes}")

    level_zero = {attribute: 0 for attribute in selected_attributes}
    max_generalization_level = _load_max_generalization_level(
        generalization_tree_path,
        selected_attributes,
    )
    original_generalization = load_generalization_rules_from_file(
        file_path=generalization_tree_path,
        data_loader=data_loader,
        generalization_level=level_zero,
    )
    train_level_zero_encode = original_generalization.encode(X_train_raw)
    val_level_zero_encode = original_generalization.encode(X_val_raw)
    test_level_zero_encode = original_generalization.encode(X_test_raw)
    if list(train_level_zero_encode.index) != list(X_train_raw.index):
        raise RuntimeError("The encoder changed training row order.")

    if standardizer == "auto":
        standardizer = fit_standardizer(train_level_zero_encode, data_loader)
    if standardizer is None:
        standardizer = PassthroughStandardizer()
    if not callable(getattr(standardizer, "transform", None)):
        raise TypeError("standardizer must be 'auto', None, or expose transform().")

    if model_factory is None:
        model_factory = lambda: TorchLogisticRegression(
            max_iter=model_max_iter,
            warm_start=False,
            device=device,
            dtype=dtype,
            allow_cpu_fallback=allow_cpu_fallback,
        )
    downstream_factory = model_factory
    proxy_factory = proxy_model_factory or downstream_factory
    if estimator_model_factory is None:
        estimator_model_factory = lambda: TorchLogisticRegression(
            max_iter=model_max_iter,
            warm_start=False,
            device=device,
            dtype=dtype,
            allow_cpu_fallback=allow_cpu_fallback,
        )
    estimator_factory = estimator_model_factory

    downstream_family = _factory_family(downstream_factory)
    proxy_family = _factory_family(proxy_factory)
    estimator_family = _factory_family(estimator_factory)
    allowed_predictive_families = {"torch_logistic", "mlp", "xgboost"}
    for role, family in (
        ("downstream", downstream_family),
        ("proxy", proxy_family),
    ):
        if family not in allowed_predictive_families:
            raise ValueError(
                f"Unsupported classification family for {role}: {family!r}."
            )
    if estimator_family != "torch_logistic":
        raise ValueError(
            "The removal-view estimator must be flat torch_logistic."
        )
    proxy_uses_xgboost_input = proxy_family == "xgboost"
    downstream_uses_xgboost_input = downstream_family == "xgboost"

    def build_model(factory):
        model = factory()
        if hasattr(model, "set_classes"):
            model.set_classes(classification_classes)
        model_task = getattr(model, "task_type", None)
        if model_task not in {None, "classification"}:
            raise ValueError(
                f"Model task_type is {model_task!r}, expected 'classification'."
            )
        return model

    resolved_device_tensor = to_device_tensor(
        train_level_zero_encode,
        device=device,
        dtype=dtype,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    tensor_device = resolved_device_tensor.device
    tensor_dtype = resolved_device_tensor.dtype
    y_train_tensor = to_device_tensor(
        y_train,
        device=tensor_device,
        dtype=tensor_dtype,
    ).reshape(-1)
    y_val_tensor = to_device_tensor(
        y_val,
        device=tensor_device,
        dtype=tensor_dtype,
    ).reshape(-1)
    y_test_tensor = to_device_tensor(
        y_test,
        device=tensor_device,
        dtype=tensor_dtype,
    ).reshape(-1)

    def model_role_encode(
        *,
        uses_xgboost_input,
        generalization,
        raw_rows,
        ordinary_encode,
    ):
        if uses_xgboost_input:
            return _xgboost_model_input(generalization, raw_rows)
        return standardizer.transform(ordinary_encode)

    def model_loss(model, y_true_tensor, model_encode) -> float:
        probabilities = (
            model.predict_proba_tensor(model_encode)
            if hasattr(model, "predict_proba_tensor")
            else to_device_tensor(
                model.predict_proba(model_encode),
                device=tensor_device,
                dtype=tensor_dtype,
            )
        )
        loss = classification_log_loss_tensor(
            y_true_tensor,
            probabilities,
            classes=classification_classes,
            n_classes=class_count,
        )
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise ValueError("Model evaluation produced a non-finite loss.")
        return value

    baseline_t0 = time.perf_counter()
    downstream_baseline_model = build_model(downstream_factory)
    downstream_baseline_train = model_role_encode(
        uses_xgboost_input=downstream_uses_xgboost_input,
        generalization=original_generalization,
        raw_rows=X_train_raw,
        ordinary_encode=train_level_zero_encode,
    )
    downstream_baseline_val = model_role_encode(
        uses_xgboost_input=downstream_uses_xgboost_input,
        generalization=original_generalization,
        raw_rows=X_val_raw,
        ordinary_encode=val_level_zero_encode,
    )
    downstream_baseline_test = model_role_encode(
        uses_xgboost_input=downstream_uses_xgboost_input,
        generalization=original_generalization,
        raw_rows=X_test_raw,
        ordinary_encode=test_level_zero_encode,
    )
    downstream_baseline_model.fit(downstream_baseline_train, y_train)
    baseline_val_loss = model_loss(
        downstream_baseline_model,
        y_val_tensor,
        downstream_baseline_val,
    )
    baseline_test_loss = model_loss(
        downstream_baseline_model,
        y_test_tensor,
        downstream_baseline_test,
    )

    proxy_baseline_model = build_model(proxy_factory)
    proxy_baseline_train = model_role_encode(
        uses_xgboost_input=proxy_uses_xgboost_input,
        generalization=original_generalization,
        raw_rows=X_train_raw,
        ordinary_encode=train_level_zero_encode,
    )
    proxy_baseline_val = model_role_encode(
        uses_xgboost_input=proxy_uses_xgboost_input,
        generalization=original_generalization,
        raw_rows=X_val_raw,
        ordinary_encode=val_level_zero_encode,
    )
    proxy_baseline_model.fit(proxy_baseline_train, y_train)
    baseline_proxy_val_loss = model_loss(
        proxy_baseline_model,
        y_val_tensor,
        proxy_baseline_val,
    )
    baseline_seconds = time.perf_counter() - baseline_t0
    proxy_loss_threshold_val = baseline_proxy_val_loss + float(tolerance)
    loss_threshold_val = baseline_val_loss + float(tolerance)
    loss_threshold_test = baseline_test_loss + float(tolerance)

    all_train_row_ids = list(X_train_raw.index)
    row_id_to_position = {
        row_id: position
        for position, row_id in enumerate(all_train_row_ids)
    }

    def label_tensor_for(row_ids: Sequence[Any]) -> torch.Tensor:
        positions = torch.as_tensor(
            [row_id_to_position[row_id] for row_id in row_ids],
            device=tensor_device,
            dtype=torch.long,
        )
        return y_train_tensor.index_select(0, positions)

    def label_array_for(row_ids: Sequence[Any]) -> np.ndarray:
        return y_train[
            [row_id_to_position[row_id] for row_id in row_ids]
        ]

    initial_groups = _group_row_ids(
        train_level_zero_encode,
        selected_attributes,
    )
    initial_privacy_stats = _privacy_stats_from_group_sizes(
        (len(row_ids) for row_ids in initial_groups.values()),
    )
    state = RemovalViewState(
        active_row_ids=list(all_train_row_ids),
        removed_row_ids=[],
        generalization_level=dict(level_zero),
        current_generalization=original_generalization,
        proxy_val_loss=baseline_proxy_val_loss,
        privacy_stats=initial_privacy_stats,
        iteration=0,
    )
    _ordered_partition(
        all_train_row_ids,
        state.active_row_ids,
        state.removed_row_ids,
    )

    write_json(
        run_dir / "config.json",
        {
            "schema_version": "trim_removal_view_config.v1",
            "algorithm": "trim_removal_view",
            "selection_rule": SELECTION_RULE,
            "selection_rule_config": selection_rule_config,
            "run_id": run_id,
            "tolerance": tolerance,
            "nrows": nrows,
            "val_size": val_size,
            "test_size": test_size,
            "random_state": random_state,
            "max_iterations": max_iterations,
            "rank_top_k": rank_top_k,
            "exchange_pool_size": exchange_pool_size,
            "enable_exchange": enable_exchange,
            "greedy_ratio_epsilon": greedy_ratio_epsilon,
            "selected_attributes": list(selected_attributes),
            "max_generalization_level": max_generalization_level,
            "task_type": "classification",
            "device": str(tensor_device),
            "dtype": str(tensor_dtype),
            "allow_cpu_fallback": bool(allow_cpu_fallback),
            "downstream_family": downstream_family,
            "proxy_family": proxy_family,
            "estimator_family": estimator_family,
            "utility_gate": "proxy_validation_relative_to_full_level0_proxy",
        },
    )
    append_jsonl(
        action_log_path,
        {
            "event": "removal_pipeline_start",
            "selection_rule": SELECTION_RULE,
            "selection_rule_config": selection_rule_config,
            "active_row_count": len(state.active_row_ids),
            "removed_row_count": 0,
            "generalization_level": state.generalization_level,
            "privacy_stats": state.privacy_stats,
            "baseline_proxy_val_loss": baseline_proxy_val_loss,
            "proxy_loss_threshold_val": proxy_loss_threshold_val,
            "baseline_downstream_val_loss": baseline_val_loss,
            "downstream_loss_threshold_val": loss_threshold_val,
        },
    )

    timings = {
        "baseline_seconds": baseline_seconds,
        "enumeration_seconds": 0.0,
        "ranking_seconds": 0.0,
        "exact_proxy_seconds": 0.0,
        "final_verification_seconds": 0.0,
    }
    action_history = []
    visited_states = {
        _candidate_fingerprint(
            state.active_row_ids,
            state.generalization_level,
        )
    }
    exchange_count = 0
    termination_condition = None

    while max_iterations is None or state.iteration < int(max_iterations):
        iteration_started_at = time.perf_counter()
        current_level = dict(state.generalization_level)
        current_generalization = state.current_generalization
        active_row_ids = list(state.active_row_ids)
        removed_row_ids = list(state.removed_row_ids)
        active_count = len(active_row_ids)
        if active_count == 0:
            raise RuntimeError("The active release cannot be empty.")
        append_jsonl(
            action_log_path,
            {
                "event": "removal_iteration_progress",
                "iteration": state.iteration + 1,
                "stage": "ranking_started",
                "active_row_count": active_count,
                "removed_row_count": len(removed_row_ids),
            },
        )

        full_current_encode = current_generalization.encode(X_train_raw)
        if list(full_current_encode.index) != all_train_row_ids:
            raise RuntimeError("The encoder changed training row order.")
        active_current_encode = full_current_encode.loc[active_row_ids]
        current_groups = _group_row_ids(
            active_current_encode,
            selected_attributes,
        )
        current_privacy_stats = _privacy_stats_from_group_sizes(
            (len(row_ids) for row_ids in current_groups.values()),
        )
        if current_privacy_stats["leak_k"] != state.privacy_stats["leak_k"]:
            raise RuntimeError("Stored and recomputed privacy states disagree.")

        gradient_t0 = time.perf_counter()
        active_score_encode = standardizer.transform(active_current_encode)
        X_active_score = to_device_tensor(
            active_score_encode,
            device=tensor_device,
            dtype=tensor_dtype,
        )
        y_active = label_tensor_for(active_row_ids)
        estimator_model = build_model(estimator_factory)
        estimator_model.fit(
            active_score_encode,
            label_array_for(active_row_ids),
        )
        try:
            theta = model_theta_tensor(estimator_model)
        except (AttributeError, ValueError) as error:
            raise ValueError(
                "The estimator must expose flat coef/intercept parameters."
            ) from error
        gradient_func = (
            multiclass_logistic_gradient
            if use_multiclass
            else logistic_gradient
        )
        y_active_gradient = (
            _class_indices_tensor(y_active, classes=classification_classes)
            if use_multiclass
            else y_active
        )
        current_train_gradient = gradient_func(
            X_active_score.to(device=theta.device, dtype=theta.dtype),
            y_active_gradient.to(device=theta.device),
            theta,
        )
        current_val_encode = current_generalization.encode(X_val_raw)
        X_current_val = to_device_tensor(
            standardizer.transform(current_val_encode),
            device=theta.device,
            dtype=theta.dtype,
        )
        y_val_gradient = y_val_tensor.to(device=theta.device)
        if use_multiclass:
            y_val_gradient = _class_indices_tensor(
                y_val_gradient,
                classes=classification_classes,
            )
        current_val_gradient = gradient_func(
            X_current_val,
            y_val_gradient,
            theta,
        )
        timings["ranking_seconds"] += time.perf_counter() - gradient_t0

        enumerate_t0 = time.perf_counter()
        candidates: list[dict[str, Any]] = []
        active_local_position = {
            row_id: position
            for position, row_id in enumerate(active_row_ids)
        }
        removal_fast_scores = {}
        removal_group_gradients = {}
        current_size_counts: dict[int, int] = {}
        for class_row_ids in current_groups.values():
            class_size = len(class_row_ids)
            current_size_counts[class_size] = (
                current_size_counts.get(class_size, 0) + 1
            )
        removal_privacy_stats_by_size = {}
        for removed_size, removed_size_count in current_size_counts.items():
            next_size_counts = dict(current_size_counts)
            if removed_size_count == 1:
                del next_size_counts[removed_size]
            else:
                next_size_counts[removed_size] = removed_size_count - 1
            removal_privacy_stats_by_size[removed_size] = (
                _privacy_stats_from_size_counts(next_size_counts)
            )

        for class_key, class_row_ids in current_groups.items():
            positions = torch.as_tensor(
                [active_local_position[row_id] for row_id in class_row_ids],
                device=theta.device,
                dtype=torch.long,
            )
            X_out = X_active_score.to(
                device=theta.device,
                dtype=theta.dtype,
            ).index_select(0, positions)
            y_out = y_active_gradient.to(device=theta.device).index_select(
                0,
                positions,
            )
            out_gradient = gradient_func(X_out, y_out, theta)
            removal_group_gradients[class_key] = out_gradient
            if len(class_row_ids) == active_count:
                removal_fast_scores[class_key] = 0.0
                continue

            next_stats = removal_privacy_stats_by_size[len(class_row_ids)]
            if next_stats["leak_k"] < current_privacy_stats["leak_k"]:
                raise AssertionError("Removing a complete class decreased min-K.")
            next_count = active_count - len(class_row_ids)
            next_gradient = (
                active_count * current_train_gradient
                - len(class_row_ids) * out_gradient
            ) / next_count
            loss_delta = current_val_gradient @ (
                next_gradient - current_train_gradient
            )
            fast_utility_score = -float(loss_delta.detach().cpu())
            removal_fast_scores[class_key] = fast_utility_score
            candidates.append(
                {
                    "action": "remove_retention_class",
                    "retention_class_key": class_key,
                    "removed_class_row_count": len(class_row_ids),
                    "next_active_row_count": next_count,
                    "next_removed_row_count": (
                        len(removed_row_ids) + len(class_row_ids)
                    ),
                    "next_generalization_level": dict(current_level),
                    "privacy_stats": next_stats,
                    "privacy_signature": _privacy_signature(next_stats),
                    "fast_utility_score": fast_utility_score,
                    "proxy_val_loss": None,
                    "feasible": None,
                    "rejection_reason": None,
                    "_remove_row_ids": list(class_row_ids),
                    "_add_row_ids": [],
                    "_next_generalization": current_generalization,
                }
            )

        for attribute, maximum in max_generalization_level.items():
            current_value = int(current_level.get(attribute, 0))
            if current_value >= int(maximum):
                continue
            next_level = dict(current_level)
            next_level[attribute] = current_value + 1
            next_generalization = current_generalization.change_level(next_level)
            next_active_encode = next_generalization.encode(
                X_train_raw.loc[active_row_ids]
            )
            if list(next_active_encode.index) != active_row_ids:
                raise RuntimeError("Coarsening changed active row order.")
            next_groups = _group_row_ids(
                next_active_encode,
                selected_attributes,
            )
            next_stats = _privacy_stats_from_group_sizes(
                (len(row_ids) for row_ids in next_groups.values()),
            )
            if next_stats["leak_k"] < current_privacy_stats["leak_k"]:
                raise AssertionError("Level+1 coarsening decreased min-K.")

            X_next_active = to_device_tensor(
                standardizer.transform(next_active_encode),
                device=theta.device,
                dtype=theta.dtype,
            )
            next_train_gradient = gradient_func(
                X_next_active,
                y_active_gradient.to(device=theta.device),
                theta,
            )
            next_val_encode = next_generalization.encode(X_val_raw)
            X_next_val = to_device_tensor(
                standardizer.transform(next_val_encode),
                device=theta.device,
                dtype=theta.dtype,
            )
            next_val_gradient = gradient_func(
                X_next_val,
                y_val_gradient,
                theta,
            )
            loss_delta = next_val_gradient @ (
                next_train_gradient - current_train_gradient
            )
            candidates.append(
                {
                    "action": "attribute_coarsen",
                    "attribute": attribute,
                    "from_level": current_value,
                    "to_level": current_value + 1,
                    "next_active_row_count": active_count,
                    "next_removed_row_count": len(removed_row_ids),
                    "next_generalization_level": next_level,
                    "privacy_stats": next_stats,
                    "privacy_signature": _privacy_signature(next_stats),
                    "fast_utility_score": -float(loss_delta.detach().cpu()),
                    "proxy_val_loss": None,
                    "feasible": None,
                    "rejection_reason": None,
                    "_remove_row_ids": [],
                    "_add_row_ids": [],
                    "_next_generalization": next_generalization,
                    "_train_encode": next_active_encode,
                    "_val_encode": next_val_encode,
                }
            )

        removed_groups = {}
        if enable_exchange and removed_row_ids:
            removed_current_encode = full_current_encode.loc[removed_row_ids]
            removed_groups = _group_row_ids(
                removed_current_encode,
                selected_attributes,
            )
            removed_local_position = {
                row_id: position
                for position, row_id in enumerate(removed_row_ids)
            }
            X_removed_score = to_device_tensor(
                standardizer.transform(removed_current_encode),
                device=theta.device,
                dtype=theta.dtype,
            )
            y_removed = label_tensor_for(removed_row_ids).to(device=theta.device)
            if use_multiclass:
                y_removed = _class_indices_tensor(
                    y_removed,
                    classes=classification_classes,
                )

            addition_fast_scores = {}
            addition_group_gradients = {}
            for class_key, class_row_ids in removed_groups.items():
                positions = torch.as_tensor(
                    [removed_local_position[row_id] for row_id in class_row_ids],
                    device=theta.device,
                    dtype=torch.long,
                )
                X_in = X_removed_score.index_select(0, positions)
                y_in = y_removed.index_select(0, positions)
                in_gradient = gradient_func(X_in, y_in, theta)
                added_gradient = (
                    active_count * current_train_gradient
                    + len(class_row_ids) * in_gradient
                ) / (active_count + len(class_row_ids))
                loss_delta = current_val_gradient @ (
                    added_gradient - current_train_gradient
                )
                addition_fast_scores[class_key] = -float(
                    loss_delta.detach().cpu()
                )
                addition_group_gradients[class_key] = in_gradient

            outgoing_keys = sorted(
                current_groups,
                key=lambda key: (
                    removal_fast_scores.get(key, -math.inf),
                    -len(current_groups[key]),
                    repr(key),
                ),
                reverse=True,
            )
            incoming_keys = sorted(
                removed_groups,
                key=lambda key: (
                    addition_fast_scores.get(key, -math.inf),
                    len(removed_groups[key]),
                    repr(key),
                ),
                reverse=True,
            )
            if exchange_pool_size is not None:
                outgoing_keys = outgoing_keys[: int(exchange_pool_size)]

            for out_key in outgoing_keys:
                out_row_ids = current_groups[out_key]
                eligible_incoming_keys = [
                    in_key
                    for in_key in incoming_keys
                    if len(removed_groups[in_key]) <= len(out_row_ids)
                ]
                if exchange_pool_size is not None:
                    eligible_incoming_keys = eligible_incoming_keys[
                        : int(exchange_pool_size)
                    ]
                for in_key in eligible_incoming_keys:
                    in_row_ids = removed_groups[in_key]
                    next_active_ids, next_removed_ids = _apply_membership_action(
                        all_row_ids=all_train_row_ids,
                        active_row_ids=active_row_ids,
                        removed_row_ids=removed_row_ids,
                        remove_row_ids=out_row_ids,
                        add_row_ids=in_row_ids,
                    )
                    next_exchange_groups = _group_row_ids(
                        full_current_encode.loc[next_active_ids],
                        selected_attributes,
                    )
                    next_stats = _privacy_stats_from_group_sizes(
                        (
                            len(row_ids)
                            for row_ids in next_exchange_groups.values()
                        ),
                    )
                    rejection_reason = _exchange_rejection_reason(
                        out_row_ids=out_row_ids,
                        in_row_ids=in_row_ids,
                        active_class_row_ids=current_groups[out_key],
                        removed_class_row_ids=removed_groups[in_key],
                        current_min_k=current_privacy_stats["leak_k"],
                        next_min_k=next_stats["leak_k"],
                    )
                    if rejection_reason is not None:
                        continue
                    next_count = (
                        active_count - len(out_row_ids) + len(in_row_ids)
                    )
                    next_gradient = (
                        active_count * current_train_gradient
                        - len(out_row_ids) * removal_group_gradients[out_key]
                        + len(in_row_ids) * addition_group_gradients[in_key]
                    ) / next_count
                    loss_delta = current_val_gradient @ (
                        next_gradient - current_train_gradient
                    )
                    candidates.append(
                        {
                            "action": "k_safe_class_exchange",
                            "out_retention_class_key": out_key,
                            "in_retention_class_key": in_key,
                            "out_row_count": len(out_row_ids),
                            "in_row_count": len(in_row_ids),
                            "next_active_row_count": len(next_active_ids),
                            "next_removed_row_count": len(next_removed_ids),
                            "next_generalization_level": dict(current_level),
                            "privacy_stats": next_stats,
                            "privacy_signature": _privacy_signature(next_stats),
                            "fast_utility_score": -float(
                                loss_delta.detach().cpu()
                            ),
                            "proxy_val_loss": None,
                            "feasible": None,
                            "rejection_reason": None,
                            "_remove_row_ids": list(out_row_ids),
                            "_add_row_ids": list(in_row_ids),
                            "_next_generalization": current_generalization,
                        }
                    )

        unseen_candidates = []
        active_row_id_set = set(active_row_ids)
        for candidate in candidates:
            if candidate["action"] == "k_safe_class_exchange":
                candidate_active_set = (
                    active_row_id_set - set(candidate["_remove_row_ids"])
                ) | set(candidate["_add_row_ids"])
                fingerprint = _candidate_fingerprint(
                    candidate_active_set,
                    candidate["next_generalization_level"],
                )
                if fingerprint in visited_states:
                    candidate["rejection_reason"] = "visited_state"
                    continue
            if (
                _privacy_signature(candidate["privacy_stats"])
                < _privacy_signature(current_privacy_stats)
            ):
                candidate["rejection_reason"] = "privacy_signature_decrease"
                continue
            unseen_candidates.append(candidate)
        enumerated_candidate_count = len(unseen_candidates)
        candidates = []
        for candidate in unseen_candidates:
            privacy_progress = _augmented_privacy_progress(
                current_privacy_stats,
                candidate["privacy_stats"],
            )
            candidate["privacy_progress"] = privacy_progress
            if privacy_progress <= 0:
                candidate["rejection_reason"] = (
                    "nonpositive_privacy_progress"
                )
                continue

            estimated_utility_loss_delta = -float(
                candidate["fast_utility_score"]
            )
            candidate["estimated_utility_loss_delta"] = (
                estimated_utility_loss_delta
                if math.isfinite(estimated_utility_loss_delta)
                else None
            )
            candidate["normalized_fast_utility_cost"] = (
                _normalized_utility_cost(
                    estimated_utility_loss_delta,
                    tolerance=float(tolerance),
                    epsilon=greedy_ratio_epsilon,
                )
            )
            candidates.append(candidate)
        timings["enumeration_seconds"] += time.perf_counter() - enumerate_t0
        append_jsonl(
            action_log_path,
            {
                "event": "removal_iteration_progress",
                "iteration": state.iteration + 1,
                "stage": "enumeration_complete",
                "candidate_count": len(candidates),
                "enumerated_candidate_count": enumerated_candidate_count,
                "enumeration_seconds_total": timings["enumeration_seconds"],
            },
        )

        if not candidates:
            termination_condition = (
                "no_information_decreasing_candidate"
                if enumerated_candidate_count == 0
                else "no_positive_privacy_progress_candidate"
            )
            append_jsonl(
                action_log_path,
                {
                    "event": "removal_iteration_state",
                    "iteration": state.iteration,
                    "selected": False,
                    "termination_condition": termination_condition,
                    "active_row_count": active_count,
                    "removed_row_count": len(removed_row_ids),
                    "generalization_level": current_level,
                    "privacy_stats": current_privacy_stats,
                },
            )
            break

        exact_t0 = time.perf_counter()
        ranked_candidates = sorted(
            candidates,
            key=lambda candidate: (
                (
                    float(candidate["normalized_fast_utility_cost"])
                    if candidate["normalized_fast_utility_cost"] is not None
                    else math.inf
                ),
                int(candidate["next_active_row_count"]),
                candidate["action"],
                repr(
                    candidate.get(
                        "retention_class_key",
                        candidate.get(
                            "attribute",
                            (
                                candidate.get("out_retention_class_key"),
                                candidate.get("in_retention_class_key"),
                            ),
                        ),
                    )
                ),
            ),
        )

        selected_candidate = None
        evaluated_candidates = []
        batch_size = (
            len(ranked_candidates)
            if rank_top_k is None
            else min(int(rank_top_k), len(ranked_candidates))
        )
        for batch_start in range(0, len(ranked_candidates), batch_size):
            feasible_batch = []
            for candidate in ranked_candidates[
                batch_start : batch_start + batch_size
            ]:
                next_active_ids, _next_removed_ids = (
                    _apply_membership_action(
                        all_row_ids=all_train_row_ids,
                        active_row_ids=active_row_ids,
                        removed_row_ids=removed_row_ids,
                        remove_row_ids=candidate["_remove_row_ids"],
                        add_row_ids=candidate["_add_row_ids"],
                    )
                )
                next_generalization = candidate["_next_generalization"]
                next_train_encode = candidate.get("_train_encode")
                if next_train_encode is None:
                    next_train_encode = full_current_encode.loc[next_active_ids]
                next_val_encode = candidate.get("_val_encode")
                if next_val_encode is None:
                    next_val_encode = next_generalization.encode(X_val_raw)
                candidate_model = build_model(proxy_factory)
                candidate_train_input = model_role_encode(
                    uses_xgboost_input=proxy_uses_xgboost_input,
                    generalization=next_generalization,
                    raw_rows=X_train_raw.loc[next_active_ids],
                    ordinary_encode=next_train_encode,
                )
                candidate_val_input = model_role_encode(
                    uses_xgboost_input=proxy_uses_xgboost_input,
                    generalization=next_generalization,
                    raw_rows=X_val_raw,
                    ordinary_encode=next_val_encode,
                )
                try:
                    candidate_model.fit(
                        candidate_train_input,
                        label_array_for(next_active_ids),
                    )
                    candidate_loss = model_loss(
                        candidate_model,
                        y_val_tensor,
                        candidate_val_input,
                    )
                except (RuntimeError, ValueError, FloatingPointError) as error:
                    candidate["proxy_val_loss"] = None
                    candidate["feasible"] = False
                    candidate["rejection_reason"] = (
                        f"model_fit_or_loss_error:{type(error).__name__}"
                    )
                    evaluated_candidates.append(candidate)
                    continue
                candidate["proxy_val_loss"] = candidate_loss
                utility_loss_delta = float(
                    candidate_loss - state.proxy_val_loss
                )
                normalized_utility_cost = _normalized_utility_cost(
                    utility_loss_delta,
                    tolerance=float(tolerance),
                    epsilon=greedy_ratio_epsilon,
                )
                candidate["utility_loss_delta"] = utility_loss_delta
                candidate["normalized_utility_cost"] = (
                    normalized_utility_cost
                )
                candidate["greedy_ratio"] = (
                    float(candidate["privacy_progress"])
                    / normalized_utility_cost
                    if normalized_utility_cost is not None
                    else 0.0
                )
                candidate["utility_slack"] = (
                    proxy_loss_threshold_val - candidate_loss
                )
                candidate["feasible"] = (
                    candidate_loss <= proxy_loss_threshold_val
                )
                if not candidate["feasible"]:
                    candidate["rejection_reason"] = "utility_threshold"
                elif candidate["action"] == "k_safe_class_exchange":
                    candidate["rejection_reason"] = _exchange_rejection_reason(
                        out_row_ids=candidate["_remove_row_ids"],
                        in_row_ids=candidate["_add_row_ids"],
                        active_class_row_ids=current_groups[
                            candidate["out_retention_class_key"]
                        ],
                        removed_class_row_ids=removed_groups[
                            candidate["in_retention_class_key"]
                        ],
                        current_min_k=current_privacy_stats["leak_k"],
                        next_min_k=candidate["privacy_stats"]["leak_k"],
                        candidate_loss=candidate_loss,
                        utility_threshold=proxy_loss_threshold_val,
                    )
                    candidate["feasible"] = (
                        candidate["rejection_reason"] is None
                    )
                evaluated_candidates.append(candidate)
                if candidate["feasible"]:
                    feasible_batch.append(candidate)

            if feasible_batch:
                selected_candidate = min(
                    feasible_batch,
                    key=lambda candidate: (
                        -float(candidate["greedy_ratio"]),
                        -float(candidate["privacy_progress"]),
                        float(candidate["proxy_val_loss"]),
                        int(candidate["next_active_row_count"]),
                        candidate["action"],
                        repr(
                            candidate.get(
                                "retention_class_key",
                                candidate.get(
                                    "attribute",
                                    (
                                        candidate.get(
                                            "out_retention_class_key"
                                        ),
                                        candidate.get(
                                            "in_retention_class_key"
                                        ),
                                    ),
                                ),
                            )
                        ),
                    ),
                )
                break

        timings["exact_proxy_seconds"] += time.perf_counter() - exact_t0
        for candidate in evaluated_candidates:
            record = _public_candidate_record(candidate)
            record.update(
                {
                    "event": "removal_candidate",
                    "iteration": state.iteration + 1,
                    "selected": candidate is selected_candidate,
                    "current_privacy_stats": current_privacy_stats,
                    "current_active_row_count": active_count,
                    "current_proxy_val_loss": state.proxy_val_loss,
                    "proxy_loss_threshold_val": proxy_loss_threshold_val,
                }
            )
            if candidate is selected_candidate:
                record["removed_row_ids"] = candidate["_remove_row_ids"]
                record["reintroduced_row_ids"] = candidate["_add_row_ids"]
            append_jsonl(action_log_path, record)

        if selected_candidate is None:
            termination_condition = "utility_constraint_blocked"
            append_jsonl(
                action_log_path,
                {
                    "event": "removal_iteration_state",
                    "iteration": state.iteration,
                    "selected": False,
                    "termination_condition": termination_condition,
                    "candidate_count": len(candidates),
                    "exactly_evaluated_candidate_count": len(
                        evaluated_candidates
                    ),
                    "active_row_count": active_count,
                    "removed_row_count": len(removed_row_ids),
                    "generalization_level": current_level,
                    "privacy_stats": current_privacy_stats,
                },
            )
            break

        previous_removed_set = set(state.removed_row_ids)
        next_active_ids, next_removed_ids = _apply_membership_action(
            all_row_ids=all_train_row_ids,
            active_row_ids=active_row_ids,
            removed_row_ids=removed_row_ids,
            remove_row_ids=selected_candidate["_remove_row_ids"],
            add_row_ids=selected_candidate["_add_row_ids"],
        )
        if selected_candidate["action"] != "k_safe_class_exchange":
            if not previous_removed_set <= set(next_removed_ids):
                raise AssertionError(
                    "Only an explicit exchange may reintroduce removed rows."
                )
        elif len(next_active_ids) > len(active_row_ids):
            raise AssertionError("A K-safe exchange increased active row count.")
        if (
            _privacy_signature(selected_candidate["privacy_stats"])
            < _privacy_signature(current_privacy_stats)
        ):
            raise AssertionError("A selected action decreased privacy signature.")

        state = RemovalViewState(
            active_row_ids=next_active_ids,
            removed_row_ids=next_removed_ids,
            generalization_level=dict(
                selected_candidate["next_generalization_level"]
            ),
            current_generalization=selected_candidate["_next_generalization"],
            proxy_val_loss=float(selected_candidate["proxy_val_loss"]),
            privacy_stats=dict(selected_candidate["privacy_stats"]),
            iteration=state.iteration + 1,
        )
        fingerprint = _candidate_fingerprint(
            state.active_row_ids,
            state.generalization_level,
        )
        if fingerprint in visited_states:
            raise AssertionError("A visited removal-view state was selected.")
        visited_states.add(fingerprint)
        if selected_candidate["action"] == "k_safe_class_exchange":
            exchange_count += 1

        selected_action_record = _public_candidate_record(selected_candidate)
        selected_action_record.update(
            {
                "iteration": state.iteration,
                "removed_row_ids": list(
                    selected_candidate["_remove_row_ids"]
                ),
                "reintroduced_row_ids": list(
                    selected_candidate["_add_row_ids"]
                ),
            }
        )
        action_history.append(selected_action_record)
        append_jsonl(
            action_log_path,
            {
                "event": "removal_iteration_state",
                "iteration": state.iteration,
                "selected": True,
                "action": selected_candidate["action"],
                "active_row_count": len(state.active_row_ids),
                "removed_row_count": len(state.removed_row_ids),
                "generalization_level": state.generalization_level,
                "privacy_stats": state.privacy_stats,
                "proxy_val_loss": state.proxy_val_loss,
                "proxy_loss_threshold_val": proxy_loss_threshold_val,
                "removed_row_ids": selected_candidate["_remove_row_ids"],
                "reintroduced_row_ids": selected_candidate["_add_row_ids"],
                "iteration_seconds": time.perf_counter()
                - iteration_started_at,
            },
        )

    if termination_condition is None:
        termination_condition = "max_iterations_reached"

    final_t0 = time.perf_counter()
    final_generalization = state.current_generalization
    final_train_encode = final_generalization.encode(
        X_train_raw.loc[state.active_row_ids]
    )
    final_val_encode = final_generalization.encode(X_val_raw)
    final_test_encode = final_generalization.encode(X_test_raw)

    final_proxy_model = build_model(proxy_factory)
    final_proxy_train_input = model_role_encode(
        uses_xgboost_input=proxy_uses_xgboost_input,
        generalization=final_generalization,
        raw_rows=X_train_raw.loc[state.active_row_ids],
        ordinary_encode=final_train_encode,
    )
    final_proxy_val_input = model_role_encode(
        uses_xgboost_input=proxy_uses_xgboost_input,
        generalization=final_generalization,
        raw_rows=X_val_raw,
        ordinary_encode=final_val_encode,
    )
    final_proxy_model.fit(
        final_proxy_train_input,
        label_array_for(state.active_row_ids),
    )
    final_proxy_val_loss = model_loss(
        final_proxy_model,
        y_val_tensor,
        final_proxy_val_input,
    )

    final_downstream_model = build_model(downstream_factory)
    final_downstream_train_input = model_role_encode(
        uses_xgboost_input=downstream_uses_xgboost_input,
        generalization=final_generalization,
        raw_rows=X_train_raw.loc[state.active_row_ids],
        ordinary_encode=final_train_encode,
    )
    final_downstream_val_input = model_role_encode(
        uses_xgboost_input=downstream_uses_xgboost_input,
        generalization=final_generalization,
        raw_rows=X_val_raw,
        ordinary_encode=final_val_encode,
    )
    final_downstream_test_input = model_role_encode(
        uses_xgboost_input=downstream_uses_xgboost_input,
        generalization=final_generalization,
        raw_rows=X_test_raw,
        ordinary_encode=final_test_encode,
    )
    final_downstream_model.fit(
        final_downstream_train_input,
        label_array_for(state.active_row_ids),
    )
    final_actual_val_loss = model_loss(
        final_downstream_model,
        y_val_tensor,
        final_downstream_val_input,
    )
    final_actual_test_loss = model_loss(
        final_downstream_model,
        y_test_tensor,
        final_downstream_test_input,
    )

    final_groups = _group_row_ids(final_train_encode, selected_attributes)
    final_privacy_stats = _privacy_stats_from_group_sizes(
        (len(row_ids) for row_ids in final_groups.values()),
        include_distribution=True,
    )
    if final_privacy_stats["leak_k"] < initial_privacy_stats["leak_k"]:
        raise AssertionError("The final removal view decreased min-K.")
    timings["final_verification_seconds"] = time.perf_counter() - final_t0
    timings["total_seconds"] = time.perf_counter() - pipeline_started_at

    result = RemovalViewResult(
        selection_rule=SELECTION_RULE,
        selection_rule_config=dict(selection_rule_config),
        active_row_ids=list(state.active_row_ids),
        removed_row_ids=list(state.removed_row_ids),
        selected_row_ids=list(state.active_row_ids),
        generalization_level=dict(state.generalization_level),
        state=state,
        termination_condition=termination_condition,
        iteration_count=state.iteration,
        exchange_count=exchange_count,
        baseline_proxy_val_loss=baseline_proxy_val_loss,
        proxy_loss_threshold_val=proxy_loss_threshold_val,
        final_proxy_val_loss=final_proxy_val_loss,
        baseline_val_loss=baseline_val_loss,
        loss_threshold_val=loss_threshold_val,
        final_actual_val_loss=final_actual_val_loss,
        validation_utility_constraint_met=(
            final_actual_val_loss <= loss_threshold_val
        ),
        baseline_test_loss=baseline_test_loss,
        loss_threshold_test=loss_threshold_test,
        final_actual_model_loss=final_actual_test_loss,
        utility_constraint_met=(
            final_actual_test_loss <= loss_threshold_test
        ),
        original_leak_k=initial_privacy_stats["leak_k"],
        original_bottleneck_count=initial_privacy_stats[
            "min_k_class_count"
        ],
        final_leak_k=final_privacy_stats["leak_k"],
        final_bottleneck_count=final_privacy_stats["min_k_class_count"],
        leak_k_p1=final_privacy_stats["leak_k_p1"],
        leak_k_p5=final_privacy_stats["leak_k_p5"],
        final_leak_k_sorted=final_privacy_stats["leak_k_sorted"],
        final_leak_k_95_percentile=final_privacy_stats[
            "leak_k_95_percentile"
        ],
        action_history=action_history,
        timings=timings,
        run_dir=str(run_dir),
    )

    result_payload = {
        result_field.name: getattr(result, result_field.name)
        for result_field in fields(result)
        if result_field.name != "state"
    }
    result_payload["schema_version"] = "trim_removal_view_metrics.v1"
    write_json(run_dir / "metrics.json", result_payload)
    write_json(
        run_dir / "selected_row_ids.json",
        {"selected_row_ids": [str(row_id) for row_id in state.active_row_ids]},
    )
    write_json(
        run_dir / "removed_row_ids.json",
        {"removed_row_ids": [str(row_id) for row_id in state.removed_row_ids]},
    )
    write_json(run_dir / "generalization_level.json", state.generalization_level)
    write_json(run_dir / "leak_distribution.json", final_privacy_stats)
    write_json(run_dir / "removal_iterations.json", action_history)
    append_jsonl(
        action_log_path,
        {
            "event": "removal_pipeline_summary",
            **result_payload,
        },
    )
    return result
