"""Load and materialize persisted hierarchical TRIM releases."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import dataloader as dataloader_module
from .enumeration_horizontal import select_encoded_attributes


@dataclass
class DatasetSplit:
    X_train_raw: Any
    X_val_raw: Any
    X_test_raw: Any
    y_train: Any
    y_val: Any
    y_test: Any


@dataclass
class MaterializedTRIMRelease:
    published_frame: Any
    published_y: Any
    evaluation_frame: Any
    evaluation_y: Any
    published_raw: Any
    evaluation_raw: Any
    evaluation_split: str
    release: dict[str, Any]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_trim_release(
    run_dir: str | Path,
    *,
    iteration: int | None = None,
) -> dict[str, Any]:
    """Load the final release or one explicitly numbered iteration release."""
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    config_path = run_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    config = _read_json(config_path)

    if iteration is None:
        release_path = run_path / "trim_release.json"
        if not release_path.is_file():
            raise FileNotFoundError(f"TRIM release not found: {release_path}")
        release = _read_json(release_path)
    else:
        iterations_path = run_path / "trim_iterations.json"
        if not iterations_path.is_file():
            raise FileNotFoundError(f"TRIM iterations not found: {iterations_path}")
        rows = _read_json(iterations_path).get("iterations")
        if not isinstance(rows, list):
            raise ValueError(f"Invalid TRIM iterations file: {iterations_path}")
        matches = [row for row in rows if int(row.get("iteration", -2)) == int(iteration)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one TRIM release for iteration {iteration}; "
                f"found {len(matches)}."
            )
        release = matches[0]

    required = {
        "status",
        "target_k",
        "assigned_row_ids",
        "suppressed_row_ids",
        "row_snapshot_assignments",
        "snapshot_history",
    }
    missing = sorted(required.difference(release))
    if missing:
        raise ValueError(f"TRIM release is missing fields: {missing}")
    if release["status"] != "ok":
        raise ValueError(f"TRIM release status is not ok: {release['status']!r}")
    return {
        "run_dir": str(run_path),
        "iteration": iteration,
        "config": config,
        "release": release,
    }


def load_dataset_split(data_loader, split_config: Mapping[str, Any]) -> DatasetSplit:
    """Recreate the train/validation/test split used by the pipeline."""
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle as shuffle_rows

    nrows = split_config.get("nrows")
    val_size = float(split_config.get("val_size", 0.15))
    test_size = float(split_config.get("test_size", 0.15))
    random_state = int(split_config.get("random_state", 42))
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1).")
    if not 0.0 < val_size < 1.0 - test_size:
        raise ValueError("val_size must be positive and leave a training split.")

    X_raw, y = data_loader.load(nrows=nrows)
    X_raw, y = shuffle_rows(X_raw, y, random_state=random_state)
    X_trainval_raw, X_test_raw, y_trainval, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    val_size_of_trainval = val_size / (1.0 - test_size)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_trainval_raw,
        y_trainval,
        test_size=val_size_of_trainval,
        random_state=random_state,
        stratify=y_trainval,
    )
    return DatasetSplit(
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def materialize_trim_release(
    data_loader,
    tree_path: str | Path,
    loaded_release: Mapping[str, Any],
    *,
    evaluation_split: str = "test",
) -> MaterializedTRIMRelease:
    """Rebuild the encoded publication and its routed evaluation split."""
    import pandas as pd

    wrapper = dict(loaded_release)
    release = dict(wrapper.get("release") or wrapper)
    split_config = dict(wrapper.get("config") or {})
    if not split_config:
        raise ValueError("Materialization requires the persisted run config.")
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be 'validation' or 'test'.")
    split = load_dataset_split(data_loader, split_config)
    X_eval_raw = split.X_val_raw if evaluation_split == "validation" else split.X_test_raw
    y_eval = split.y_val if evaluation_split == "validation" else split.y_test

    train_id_by_text = {str(row_id): row_id for row_id in split.X_train_raw.index}
    if len(train_id_by_text) != len(split.X_train_raw.index):
        raise ValueError("Training row IDs are not unique after string conversion.")
    assignment_text = dict(release["row_snapshot_assignments"])
    unknown_ids = sorted(set(assignment_text).difference(train_id_by_text))
    if unknown_ids:
        raise ValueError(
            "Release assignments contain row IDs outside the recreated training split: "
            f"{unknown_ids[:5]}"
        )
    assigned_ids = [train_id_by_text[str(row_id)] for row_id in release["assigned_row_ids"]]
    if len(assigned_ids) != len(set(assigned_ids)):
        raise ValueError("TRIM release contains duplicate assigned row IDs.")
    if set(assigned_ids) != {train_id_by_text[row_id] for row_id in assignment_text}:
        raise ValueError("assigned_row_ids and row_snapshot_assignments disagree.")

    history = release["snapshot_history"]
    if not isinstance(history, list) or not history:
        raise ValueError("TRIM release snapshot_history must be non-empty.")
    snapshots = {}
    for snapshot in history:
        iteration = int(snapshot["iteration"])
        if iteration in snapshots:
            raise ValueError(f"Duplicate TRIM snapshot iteration: {iteration}")
        generalization = dataloader_module.load_generalization_rules_from_file(
            file_path=tree_path,
            data_loader=data_loader,
            generalization_level=snapshot["generalization_level"],
        )
        snapshots[iteration] = generalization

    assignment_by_id = {
        train_id_by_text[row_id]: int(iteration)
        for row_id, iteration in assignment_text.items()
    }
    unknown_iterations = sorted(set(assignment_by_id.values()).difference(snapshots))
    if unknown_iterations:
        raise ValueError(
            f"Release assignments reference missing snapshots: {unknown_iterations}"
        )

    ordered_iterations = sorted(snapshots, reverse=True)
    qi_attributes = tuple(
        getattr(data_loader, "qi_attributes", data_loader.feature_columns)
    )
    published_parts = []
    accepted_groups = {}
    for iteration in ordered_iterations:
        row_ids = [
            row_id for row_id in assigned_ids if assignment_by_id[row_id] == iteration
        ]
        if not row_ids:
            continue
        encoded = snapshots[iteration].encode(split.X_train_raw.loc[row_ids])
        published_parts.append(encoded)
        privacy_encode = select_encoded_attributes(encoded, qi_attributes)
        accepted_groups[iteration] = privacy_encode.drop_duplicates().reset_index(
            drop=True
        )
    if not published_parts:
        raise ValueError("TRIM release has no materializable assigned rows.")
    published_frame = pd.concat(published_parts, axis=0).loc[assigned_ids]

    remaining_eval_ids = list(X_eval_raw.index)
    evaluation_parts = []
    for iteration in ordered_iterations:
        accepted = accepted_groups.get(iteration)
        if not remaining_eval_ids or accepted is None or accepted.empty:
            continue
        candidate = snapshots[iteration].encode(X_eval_raw.loc[remaining_eval_ids])
        candidate_privacy = select_encoded_attributes(candidate, qi_attributes)
        candidate_keys = pd.MultiIndex.from_frame(candidate_privacy)
        accepted_keys = pd.MultiIndex.from_frame(accepted)
        matched_ids = candidate.index[candidate_keys.isin(accepted_keys)].tolist()
        if not matched_ids:
            continue
        evaluation_parts.append(candidate.loc[matched_ids])
        matched_set = set(matched_ids)
        remaining_eval_ids = [
            row_id for row_id in remaining_eval_ids if row_id not in matched_set
        ]
    if remaining_eval_ids:
        coarsest_iteration = min(snapshots)
        evaluation_parts.append(
            snapshots[coarsest_iteration].encode(X_eval_raw.loc[remaining_eval_ids])
        )
    evaluation_frame = pd.concat(evaluation_parts, axis=0).loc[X_eval_raw.index]

    published_privacy_frame = select_encoded_attributes(
        published_frame,
        qi_attributes,
    )
    group_sizes = published_privacy_frame.value_counts(
        sort=False,
        dropna=False,
    ).values
    actual_min_k = int(np.min(group_sizes))
    if actual_min_k != int(release["min_k"]):
        raise ValueError(
            "Materialized TRIM min-K disagrees with the release record: "
            f"{actual_min_k} != {release['min_k']}"
        )
    return MaterializedTRIMRelease(
        published_frame=published_frame,
        published_y=split.y_train.loc[assigned_ids],
        evaluation_frame=evaluation_frame,
        evaluation_y=y_eval.loc[X_eval_raw.index],
        published_raw=split.X_train_raw.loc[assigned_ids],
        evaluation_raw=X_eval_raw,
        evaluation_split=evaluation_split,
        release=release,
    )
