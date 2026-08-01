"""Mondrian representation used by Experiment 1.

The implementation partitions the training split only.  Partition boundaries
are derived from training rows, and every training row is published after
local recoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


_NA_SENTINEL = "__mondrian_na__"


@dataclass(frozen=True)
class _GlobalAttrStats:
    numeric_min: float
    numeric_max: float
    value_rank: dict[Any, int]
    distinct_count: int


def _is_na_sentinel(value: Any) -> bool:
    return isinstance(value, str) and value == _NA_SENTINEL


def _is_na_value(value: Any) -> bool:
    import pandas as pd

    return bool(pd.isna(value))


def _build_global_stats(X, attributes, numeric_attributes):
    import pandas as pd

    numeric_set = set(numeric_attributes)
    stats: dict[str, _GlobalAttrStats] = {}
    for attribute in attributes:
        column = X[attribute]
        if attribute in numeric_set:
            numeric = pd.to_numeric(column, errors="coerce")
            stats[attribute] = _GlobalAttrStats(
                numeric_min=(
                    float(numeric.min()) if not numeric.isna().all() else 0.0
                ),
                numeric_max=(
                    float(numeric.max()) if not numeric.isna().all() else 0.0
                ),
                value_rank={},
                distinct_count=0,
            )
            continue

        cleaned = column.astype("object").where(column.notna(), _NA_SENTINEL)
        counts = cleaned.value_counts(dropna=False)
        value_rank = {}
        for rank, (value, _count) in enumerate(counts.items()):
            key = _NA_SENTINEL if _is_na_sentinel(value) else value
            value_rank[key] = rank
        stats[attribute] = _GlobalAttrStats(
            numeric_min=0.0,
            numeric_max=0.0,
            value_rank=value_rank,
            distinct_count=len(value_rank),
        )
    return stats


def _partition_split_key(values, stats: _GlobalAttrStats) -> np.ndarray:
    import pandas as pd

    if stats.value_rank:
        return np.asarray(
            [
                -1 if _is_na_value(value) else stats.value_rank.get(value, -1)
                for value in values
            ],
            dtype=np.intp,
        )
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
        dtype=np.float64
    )
    return np.where(np.isnan(numeric), np.inf, numeric)


def _normalized_range(values, stats: _GlobalAttrStats) -> float:
    import pandas as pd

    if stats.value_rank:
        cleaned = values.astype("object").where(values.notna(), _NA_SENTINEL)
        if stats.distinct_count <= 0:
            return 0.0
        return float(cleaned.nunique(dropna=False)) / float(stats.distinct_count)

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().all():
        return 0.0
    global_span = stats.numeric_max - stats.numeric_min
    if global_span <= 0.0:
        return 0.0
    return (float(numeric.max()) - float(numeric.min())) / global_span


def _median_split_position(sorted_keys: np.ndarray, k: int) -> int | None:
    row_count = len(sorted_keys)
    if row_count < 2 * k:
        return None
    boundaries = np.nonzero(sorted_keys[1:] != sorted_keys[:-1])[0]
    if boundaries.size == 0:
        return None
    left_counts = boundaries + 1
    valid = (left_counts >= k) & ((row_count - left_counts) >= k)
    if not np.any(valid):
        return None
    valid_boundaries = boundaries[valid]
    return int(
        valid_boundaries[
            np.argmin(np.abs(valid_boundaries + 1 - row_count / 2.0))
        ]
    )


def _mondrian_split(
    X,
    row_ids: list[Any],
    k: int,
    numeric_attributes,
    categorical_attributes,
    stats_by_attribute,
) -> tuple[list[Any], list[Any]] | None:
    if len(row_ids) < 2 * k:
        return None
    subset = X.loc[row_ids]
    attributes = list(numeric_attributes) + list(categorical_attributes)
    scored = [
        (
            attribute,
            _normalized_range(
                subset[attribute], stats_by_attribute[attribute]
            ),
        )
        for attribute in attributes
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    if not scored or scored[0][1] <= 0.0:
        return None

    for attribute, _score in scored:
        keys = _partition_split_key(
            subset[attribute].to_numpy(), stats_by_attribute[attribute]
        )
        order = np.argsort(keys, kind="stable")
        split_position = _median_split_position(keys[order], k)
        if split_position is None:
            continue
        left = [row_ids[position] for position in order[: split_position + 1]]
        right = [row_ids[position] for position in order[split_position + 1 :]]
        return left, right
    return None


def mondrian_partition(
    X,
    *,
    k: int,
    numeric_attributes: Iterable[str],
    categorical_attributes: Iterable[str],
) -> list[list[Any]]:
    """Return full-coverage Mondrian leaf partitions, each of size at least K."""
    k = int(k)
    if k <= 0:
        raise ValueError("Mondrian k must be positive.")
    if X.empty:
        raise ValueError("Mondrian requires a non-empty training frame.")
    if not X.index.is_unique:
        raise ValueError("Mondrian requires unique training row IDs.")
    numeric_attributes = tuple(numeric_attributes)
    categorical_attributes = tuple(categorical_attributes)
    attributes = numeric_attributes + categorical_attributes
    if not attributes:
        raise ValueError("Mondrian requires at least one quasi-identifier.")
    missing = [attribute for attribute in attributes if attribute not in X.columns]
    if missing:
        raise ValueError(f"Mondrian attributes are absent from the data: {missing}")
    stats = _build_global_stats(X, attributes, numeric_attributes)

    def recurse(row_ids: list[Any]) -> list[list[Any]]:
        split = _mondrian_split(
            X,
            row_ids,
            k,
            numeric_attributes,
            categorical_attributes,
            stats,
        )
        if split is None:
            return [row_ids]
        left, right = split
        return recurse(left) + recurse(right)

    partitions = recurse(list(X.index))
    flattened = [row_id for partition in partitions for row_id in partition]
    if len(flattened) != len(X) or set(flattened) != set(X.index):
        raise RuntimeError("Mondrian partitions do not cover the training rows once.")
    if min(map(len, partitions)) < k:
        raise RuntimeError("Mondrian produced a partition smaller than K.")
    return partitions


def encode_mondrian(
    X_raw,
    partitions: list[list[Any]],
    *,
    numeric_attributes: Iterable[str],
    categorical_attributes: Iterable[str],
    category_maps: dict[str, dict[Any, int]],
):
    """Encode local recoding with the same columns as ``encode_original``."""
    import pandas as pd

    numeric_attributes = tuple(numeric_attributes)
    categorical_attributes = tuple(categorical_attributes)
    row_to_partition: dict[Any, int] = {}
    for partition_index, row_ids in enumerate(partitions):
        for row_id in row_ids:
            if row_id in row_to_partition:
                raise ValueError(f"Duplicate Mondrian row ID: {row_id!r}")
            row_to_partition[row_id] = partition_index
    if set(row_to_partition) != set(X_raw.index):
        raise ValueError("Mondrian partitions and encoded rows do not agree.")

    numeric_means: list[dict[str, float]] = []
    categorical_values: list[dict[str, list[Any]]] = []
    for row_ids in partitions:
        partition = X_raw.loc[row_ids]
        means = {}
        for attribute in numeric_attributes:
            numeric = pd.to_numeric(partition[attribute], errors="coerce")
            means[attribute] = (
                float(numeric.mean()) if not numeric.isna().all() else 0.0
            )
        numeric_means.append(means)

        values_by_attribute = {}
        for attribute in categorical_attributes:
            values_by_attribute[attribute] = [
                value
                for value in partition[attribute].dropna().unique().tolist()
                if not _is_na_sentinel(value)
            ]
        categorical_values.append(values_by_attribute)

    encoded_parts = []
    for attribute in numeric_attributes:
        encoded_parts.append(
            pd.DataFrame(
                {
                    attribute: [
                        numeric_means[row_to_partition[row_id]][attribute]
                        for row_id in X_raw.index
                    ]
                },
                index=X_raw.index,
            )
        )

    for attribute in categorical_attributes:
        categories = list(category_maps[attribute])
        column_names = [f"{attribute}={category}" for category in categories]
        rows = []
        for row_id in X_raw.index:
            distinct = categorical_values[row_to_partition[row_id]][attribute]
            vector = [0.0] * len(categories)
            if distinct:
                weight = 1.0 / float(len(distinct))
                for value in distinct:
                    category_index = category_maps[attribute].get(value)
                    if category_index is not None:
                        vector[int(category_index)] = weight
            rows.append(vector)
        encoded_parts.append(
            pd.DataFrame(rows, columns=column_names, index=X_raw.index)
        )

    return pd.concat(encoded_parts, axis=1)


def partition_records(
    X_raw,
    partitions: list[list[Any]],
    *,
    numeric_attributes: Iterable[str],
    categorical_attributes: Iterable[str],
) -> list[dict[str, Any]]:
    """Return compact, JSON-ready partition provenance."""
    import pandas as pd

    records = []
    for partition_index, row_ids in enumerate(partitions):
        part = X_raw.loc[row_ids]
        summary: dict[str, Any] = {}
        for attribute in numeric_attributes:
            numeric = pd.to_numeric(part[attribute], errors="coerce")
            summary[attribute] = {
                "min": None if numeric.isna().all() else float(numeric.min()),
                "max": None if numeric.isna().all() else float(numeric.max()),
                "mean": None if numeric.isna().all() else float(numeric.mean()),
            }
        for attribute in categorical_attributes:
            summary[attribute] = {
                "values": sorted(
                    [
                        value.item() if hasattr(value, "item") else value
                        for value in part[attribute].dropna().unique().tolist()
                    ],
                    key=str,
                )
            }
        records.append(
            {
                "partition": partition_index,
                "size": len(row_ids),
                "row_ids": [
                    row_id.item() if hasattr(row_id, "item") else row_id
                    for row_id in row_ids
                ],
                "summary": summary,
            }
        )
    return records
