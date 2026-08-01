"""Label-covering initialization helpers for TRIM."""

from __future__ import annotations

import random
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


def _aligned_label_codes(
    row_ids: Sequence[Any],
    labels: Any,
) -> tuple[list[Any], np.ndarray, list[Any]]:
    rows = list(row_ids)
    if not rows:
        raise ValueError("Training rows cannot be empty when constructing D0.")
    if len(rows) != len(set(rows)):
        raise ValueError("Training row identifiers must be unique.")

    if isinstance(labels, pd.Series):
        if not labels.index.is_unique:
            raise ValueError("Training-label indices must be unique.")
        missing = [row_id for row_id in rows if row_id not in labels.index]
        if missing:
            raise KeyError(
                "Training labels are missing row identifiers required for D0: "
                f"{missing}"
            )
        values = labels.loc[rows].to_numpy()
    else:
        values = np.asarray(labels)
        if values.ndim != 1:
            raise ValueError("Training labels must be one-dimensional.")
        if len(values) != len(rows):
            raise ValueError(
                "Training rows and labels must have the same length when "
                "constructing D0."
            )

    codes, unique_labels = pd.factorize(values, sort=False)
    if np.any(codes < 0):
        raise ValueError("Training labels cannot be missing when constructing D0.")
    if len(unique_labels) == 0:
        raise ValueError("Training labels contain no classes.")
    return rows, codes.astype(np.intp, copy=False), unique_labels.tolist()


def validate_label_coverage(
    row_ids: Sequence[Any],
    labels: Any,
    selected_row_ids: Iterable[Any],
) -> list[Any]:
    """Return selected row IDs after verifying that they cover every label."""

    rows, label_codes, unique_labels = _aligned_label_codes(row_ids, labels)
    selected = list(selected_row_ids)
    if not selected:
        raise ValueError("D0 must contain at least one row.")
    if len(selected) != len(set(selected)):
        raise ValueError("D0 row identifiers must be unique.")

    code_by_row_id = dict(zip(rows, label_codes.tolist()))
    missing_rows = [row_id for row_id in selected if row_id not in code_by_row_id]
    if missing_rows:
        raise KeyError(f"D0 contains rows outside the training split: {missing_rows}")

    covered_codes = {code_by_row_id[row_id] for row_id in selected}
    missing_codes = sorted(set(range(len(unique_labels))) - covered_codes)
    if missing_codes:
        missing_labels = [unique_labels[code] for code in missing_codes]
        raise ValueError(
            "D0 must contain at least one row from every training label; "
            f"missing labels: {missing_labels!r}."
        )
    return selected


def stratified_sample_row_ids(
    row_ids: Sequence[Any],
    labels: Any,
    sample_size: int,
    random_state: int,
) -> list[Any]:
    """Draw an exact-size, approximately proportional, label-covering D0."""

    rows, label_codes, unique_labels = _aligned_label_codes(row_ids, labels)
    size = int(sample_size)
    if size != sample_size:
        raise ValueError("D0 sample size must be an integer.")
    if size < len(unique_labels):
        raise ValueError(
            "D0 sample size must be at least the number of training labels; "
            f"got sample_size={size}, label_count={len(unique_labels)}."
        )
    if size > len(rows):
        raise ValueError("D0 sample size cannot exceed the training row count.")

    class_counts = np.bincount(label_codes, minlength=len(unique_labels))
    ideal_quotas = size * class_counts.astype(float) / float(len(rows))
    quotas = np.maximum(1, np.floor(ideal_quotas).astype(np.intp))
    quotas = np.minimum(quotas, class_counts)

    while int(quotas.sum()) > size:
        candidates = np.flatnonzero(quotas > 1)
        if candidates.size == 0:
            raise RuntimeError("Could not construct a label-covering D0 quota.")
        surplus = quotas[candidates] - ideal_quotas[candidates]
        quotas[int(candidates[int(np.argmax(surplus))])] -= 1

    while int(quotas.sum()) < size:
        candidates = np.flatnonzero(quotas < class_counts)
        if candidates.size == 0:
            raise RuntimeError("Could not fill the requested D0 sample size.")
        deficit = ideal_quotas[candidates] - quotas[candidates]
        quotas[int(candidates[int(np.argmax(deficit))])] += 1

    rng = random.Random(int(random_state))
    selected_positions = []
    for class_code, quota in enumerate(quotas.tolist()):
        positions = np.flatnonzero(label_codes == class_code).tolist()
        selected_positions.extend(rng.sample(positions, int(quota)))
    rng.shuffle(selected_positions)
    selected = [rows[position] for position in selected_positions]
    return validate_label_coverage(rows, labels, selected)
