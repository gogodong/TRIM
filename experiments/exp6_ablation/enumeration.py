"""Fixed horizontal admission units for the Figure 5h ablation."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import random
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from prototype import FixedRowCandidateBuilderContext, FixedRowCandidatePlan
from prototype.dataloader import _coerce_scalar, _parse_leaf_value


INITIAL_GENERALIZATION_MODES = frozenset({"level0", "max"})


def _positive_integer(value, *, name):
    if isinstance(value, bool) or int(value) < 1 or float(value) != int(value):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _initial_generalization(context, mode):
    if mode not in INITIAL_GENERALIZATION_MODES:
        raise ValueError(
            "initial_generalization must be either 'level0' or 'max'."
        )
    if mode == "level0":
        return {attribute: 0 for attribute in context.selected_attributes}
    return {
        attribute: int(context.max_generalization_level[attribute])
        for attribute in context.selected_attributes
    }


def _fixed_plan(context, groups, *, method, initial_generalization, metadata):
    materialized = [
        (key, tuple(row_ids))
        for key, row_ids in groups
        if row_ids
    ]
    if not materialized:
        raise ValueError(f"{method} enumeration produced no candidate groups.")
    materialized.sort(key=lambda item: len(item[1]), reverse=True)
    largest_key, largest_rows = materialized[0]
    return FixedRowCandidatePlan(
        candidate_groups=tuple(materialized),
        initial_row_ids=largest_rows,
        privacy_exempt_row_ids=(),
        generalization_level=_initial_generalization(
            context, initial_generalization
        ),
        whole_group_admission=True,
        close_remaining_groups=False,
        metadata={
            "method": method,
            "initial_generalization": initial_generalization,
            "candidate_count": len(materialized),
            "initial_group_key": largest_key,
            "initial_group_size": len(largest_rows),
            **metadata,
        },
    )


@dataclass(frozen=True)
class KMeansClusterRingBuilder:
    """Partition level-0 rows by KMeans cluster and distance ring."""

    n_clusters: int
    n_rings: int
    random_state: int
    initial_generalization: str = "level0"

    def __post_init__(self):
        object.__setattr__(
            self,
            "n_clusters",
            _positive_integer(self.n_clusters, name="n_clusters"),
        )
        object.__setattr__(
            self,
            "n_rings",
            _positive_integer(self.n_rings, name="n_rings"),
        )
        object.__setattr__(self, "random_state", int(self.random_state))
        _initial_generalization_mode(self.initial_generalization)

    def __call__(self, context: FixedRowCandidateBuilderContext):
        encode = context.train_level0_encode
        if encode.empty:
            raise ValueError("KMeans cluster-ring input cannot be empty.")
        if self.n_clusters > len(encode):
            raise ValueError("n_clusters cannot exceed the training row count.")

        values = encode.astype(float).to_numpy()
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        cluster_ids = kmeans.fit_predict(values)
        distances = np.linalg.norm(
            values - kmeans.cluster_centers_[cluster_ids], axis=1
        )
        assignments = pd.DataFrame({
            "cluster_id": cluster_ids.astype(int),
            "distance": distances,
            "ring_id": 0,
        })
        for cluster_id in sorted(assignments["cluster_id"].unique()):
            positions = assignments.index[
                assignments["cluster_id"] == cluster_id
            ]
            cluster_distances = assignments.loc[positions, "distance"]
            if len(cluster_distances) == 1:
                percentiles = pd.Series(0.0, index=positions)
            else:
                percentiles = (
                    cluster_distances.rank(method="min") - 1
                ) / (len(cluster_distances) - 1)
            ring_ids = np.floor(percentiles * self.n_rings).astype(int)
            assignments.loc[positions, "ring_id"] = ring_ids.clip(
                0, self.n_rings - 1
            ).to_numpy()

        groups = []
        for (cluster_id, ring_id), group in assignments.groupby(
            ["cluster_id", "ring_id"], sort=True
        ):
            groups.append((
                (int(cluster_id), int(ring_id)),
                tuple(encode.iloc[group.index].index.tolist()),
            ))
        return _fixed_plan(
            context,
            groups,
            method="kmeans_cluster_ring",
            initial_generalization=self.initial_generalization,
            metadata={
                "n_clusters": self.n_clusters,
                "n_rings": self.n_rings,
                "random_state": self.random_state,
                "n_init": 10,
            },
        )

    def as_dict(self):
        return {
            "type": "kmeans_cluster_ring",
            "n_clusters": self.n_clusters,
            "n_rings": self.n_rings,
            "random_state": self.random_state,
            "n_init": 10,
            "initial_generalization": self.initial_generalization,
        }


@dataclass(frozen=True)
class SampleLevelBuilder:
    """Use each level-0 training record as one admission unit."""

    initial_generalization: str = "max"

    def __post_init__(self):
        _initial_generalization_mode(self.initial_generalization)

    def __call__(self, context: FixedRowCandidateBuilderContext):
        groups = [
            (position, (row_id,))
            for position, row_id in enumerate(
                context.train_level0_encode.index.tolist()
            )
        ]
        return _fixed_plan(
            context,
            groups,
            method="sample_level",
            initial_generalization=self.initial_generalization,
            metadata={},
        )

    def as_dict(self):
        return {
            "type": "sample_level",
            "initial_generalization": self.initial_generalization,
        }


@dataclass(frozen=True)
class RandomSplitBuilder:
    """Randomly partition level-0 row identifiers into equal-sized units."""

    n_splits: int
    random_state: int
    initial_generalization: str = "max"

    def __post_init__(self):
        object.__setattr__(
            self,
            "n_splits",
            _positive_integer(self.n_splits, name="n_splits"),
        )
        object.__setattr__(self, "random_state", int(self.random_state))
        _initial_generalization_mode(self.initial_generalization)

    def __call__(self, context: FixedRowCandidateBuilderContext):
        encode = context.train_level0_encode
        if encode.empty:
            raise ValueError("RandomSplit input cannot be empty.")
        if self.n_splits > len(encode):
            raise ValueError("n_splits cannot exceed the training row count.")
        row_ids = encode.index.tolist()
        shuffled = row_ids[:]
        random.Random(self.random_state).shuffle(shuffled)
        groups = [
            (part_id, tuple(part.tolist()))
            for part_id, part in enumerate(
                np.array_split(np.asarray(shuffled, dtype=object), self.n_splits)
            )
            if len(part)
        ]
        return _fixed_plan(
            context,
            groups,
            method="random_split",
            initial_generalization=self.initial_generalization,
            metadata={
                "n_splits": self.n_splits,
                "random_state": self.random_state,
            },
        )

    def as_dict(self):
        return {
            "type": "random_split",
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "initial_generalization": self.initial_generalization,
        }


def _initial_generalization_mode(value):
    if value not in INITIAL_GENERALIZATION_MODES:
        raise ValueError(
            "initial_generalization must be either 'level0' or 'max'."
        )
    return value


def _value_to_leaf_id_map(tree, is_numeric):
    mapping = {}
    for node in tree.get("nodes", []):
        if node.get("kind") != "leaf":
            continue
        leaf_id = node["id"]
        if is_numeric:
            kind, *rest = _parse_leaf_value(node["value"])
            if kind == "interval":
                low, high = rest
                for value in range(int(low), int(high) + 1):
                    mapping[value] = leaf_id
            else:
                mapping[_coerce_scalar(rest[0])] = leaf_id
        else:
            mapping[_coerce_scalar(node["value"])] = leaf_id
    return mapping


def _row_leaf_ids(level0_encode, trees, qi_attributes):
    leaf_matrix = [[None] * len(qi_attributes) for _ in range(len(level0_encode))]
    for attribute_index, attribute in enumerate(qi_attributes):
        tree = trees.get(attribute)
        if tree is None:
            continue
        is_numeric = tree.get("attribute_type") == "continuous"
        value_to_leaf = _value_to_leaf_id_map(tree, is_numeric)
        if is_numeric:
            if attribute not in level0_encode.columns:
                continue
            for row_position, value in enumerate(
                level0_encode[attribute].to_numpy()
            ):
                normalized = (
                    int(value) if float(value).is_integer() else value
                )
                leaf_matrix[row_position][attribute_index] = value_to_leaf.get(
                    normalized
                )
            continue

        columns = [
            column
            for column in level0_encode.columns
            if column.startswith(f"{attribute}=")
        ]
        if not columns:
            continue
        values = level0_encode[columns].to_numpy(copy=False)
        hits = values == 1.0
        hit_rows = hits.any(axis=1)
        first_hits = hits.argmax(axis=1)
        leaf_ids = [
            value_to_leaf.get(
                _coerce_scalar(column[len(f"{attribute}="):])
            )
            for column in columns
        ]
        for row_position in np.flatnonzero(hit_rows):
            leaf_matrix[int(row_position)][attribute_index] = leaf_ids[
                int(first_hits[row_position])
            ]
    return leaf_matrix


def _lowest_common_ancestor(leaf_ids, nodes_by_id):
    common = None
    for node_id in leaf_ids:
        chain = set()
        current = node_id
        while current is not None:
            node = nodes_by_id.get(current)
            if node is None:
                return None
            chain.add(current)
            current = node.get("parent")
        common = chain if common is None else common.intersection(chain)
        if not common:
            return None
    return max(
        common,
        key=lambda node_id: int(
            nodes_by_id[node_id].get("depth_from_root", 0)
        ),
    )


def _information_loss_groups(
    encode,
    *,
    selected_attributes,
    trees,
    k_ur,
    merge_partner_sample_size,
    merge_strategy,
    random_state,
):
    qi_attributes = tuple(
        attribute for attribute in selected_attributes if attribute in trees
    )
    if not qi_attributes:
        raise ValueError("IL enumeration requires tree-backed QI attributes.")
    leaf_matrix = _row_leaf_ids(encode, trees, qi_attributes)
    row_ids = encode.index.tolist()

    if merge_strategy == "leaf_bucket":
        leaf_order = {
            attribute: {
                node["id"]: position
                for position, node in enumerate(trees[attribute].get("nodes", []))
                if node.get("kind") == "leaf"
            }
            for attribute in qi_attributes
        }

        def row_key(row_position):
            parts = []
            for attribute_index, attribute in enumerate(qi_attributes):
                leaf_id = leaf_matrix[row_position][attribute_index]
                order = leaf_order[attribute].get(leaf_id)
                parts.append((1, "") if order is None else (0, order))
            parts.append((0, row_position))
            return tuple(parts)

        ordered = sorted(range(len(row_ids)), key=row_key)
        groups = []
        for start in range(0, len(ordered), k_ur):
            members = [row_ids[position] for position in ordered[start:start + k_ur]]
            if len(members) < k_ur and groups:
                groups[-1][1].extend(members)
            else:
                groups.append([len(groups), members])
        return [(group_id, tuple(members)) for group_id, members in groups]

    nodes_by_attribute = {
        attribute: {
            node["id"]: node
            for node in trees[attribute].get("nodes", [])
        }
        for attribute in qi_attributes
    }
    denominators = {}
    for attribute in qi_attributes:
        root = next(
            (
                node
                for node in trees[attribute].get("nodes", [])
                if node.get("kind") == "root"
            ),
            None,
        )
        denominators[attribute] = max(
            int((root or {}).get("leaf_count", 0)) - 1, 0
        )

    def cluster_il(row_positions):
        total = 0.0
        for attribute_index, attribute in enumerate(qi_attributes):
            denominator = denominators[attribute]
            if denominator <= 0:
                continue
            leaf_ids = {
                leaf_matrix[position][attribute_index]
                for position in row_positions
                if leaf_matrix[position][attribute_index] is not None
            }
            if not leaf_ids:
                continue
            ancestor = _lowest_common_ancestor(
                leaf_ids, nodes_by_attribute[attribute]
            )
            if ancestor is not None:
                leaf_count = int(
                    nodes_by_attribute[attribute][ancestor].get("leaf_count", 1)
                )
                total += (leaf_count - 1) / denominator
        return total

    clusters = [
        {"rows": [position], "loss": 0.0}
        for position in range(len(row_ids))
    ]
    alive = set(range(len(clusters)))
    active = set(alive)
    heap = [(1, index) for index in alive]
    heapq.heapify(heap)
    rng = random.Random(random_state)

    def next_seed():
        while heap:
            size, index = heapq.heappop(heap)
            if (
                index in active
                and clusters[index] is not None
                and len(clusters[index]["rows"]) == size
            ):
                return index
        return None

    def partners(pool, seed):
        candidates = [index for index in pool if index != seed]
        if (
            merge_partner_sample_size is None
            or len(candidates) <= merge_partner_sample_size
        ):
            return candidates
        return rng.sample(candidates, merge_partner_sample_size)

    while active and len(alive) > 1:
        seed = next_seed()
        if seed is None:
            break
        pool = active if len(active) > 1 else alive
        best = None
        for partner in partners(pool, seed):
            union_rows = clusters[seed]["rows"] + clusters[partner]["rows"]
            union_loss = len(union_rows) * cluster_il(union_rows)
            delta = (
                union_loss
                - clusters[seed]["loss"]
                - clusters[partner]["loss"]
            )
            key = (delta, len(clusters[partner]["rows"]), partner)
            if best is None or key < best[0]:
                best = (key, partner, union_rows, union_loss)
        if best is None:
            break
        _, partner, union_rows, union_loss = best
        clusters[seed] = {"rows": union_rows, "loss": union_loss}
        clusters[partner] = None
        alive.remove(partner)
        active.discard(partner)
        if len(union_rows) < k_ur:
            active.add(seed)
            heapq.heappush(heap, (len(union_rows), seed))
        else:
            active.discard(seed)

    return [
        (
            group_id,
            tuple(row_ids[position] for position in cluster["rows"]),
        )
        for group_id, cluster in enumerate(
            cluster for cluster in clusters if cluster is not None
        )
    ]


@dataclass(frozen=True)
class InformationLossBuilder:
    """Merge level-0 rows into fixed groups using generalization-tree IL."""

    trees: Mapping[str, Any]
    k_ur: int = 10
    merge_partner_sample_size: int | None = 64
    merge_strategy: str = "leaf_bucket"
    random_state: int = 42
    initial_generalization: str = "max"

    def __post_init__(self):
        if not isinstance(self.trees, Mapping) or not self.trees:
            raise ValueError("trees must be a non-empty mapping.")
        object.__setattr__(self, "k_ur", _positive_integer(self.k_ur, name="k_ur"))
        if self.merge_partner_sample_size is not None:
            object.__setattr__(
                self,
                "merge_partner_sample_size",
                _positive_integer(
                    self.merge_partner_sample_size,
                    name="merge_partner_sample_size",
                ),
            )
        if self.merge_strategy not in {"leaf_bucket", "agglomerative"}:
            raise ValueError(
                "merge_strategy must be 'leaf_bucket' or 'agglomerative'."
            )
        object.__setattr__(self, "random_state", int(self.random_state))
        _initial_generalization_mode(self.initial_generalization)

    def __call__(self, context: FixedRowCandidateBuilderContext):
        encode = context.train_level0_encode
        if encode.empty:
            raise ValueError("IL enumeration input cannot be empty.")
        groups = _information_loss_groups(
            encode,
            selected_attributes=context.selected_attributes,
            trees=self.trees,
            k_ur=self.k_ur,
            merge_partner_sample_size=self.merge_partner_sample_size,
            merge_strategy=self.merge_strategy,
            random_state=self.random_state,
        )
        return _fixed_plan(
            context,
            groups,
            method="information_loss",
            initial_generalization=self.initial_generalization,
            metadata={
                "k_ur": self.k_ur,
                "merge_partner_sample_size": self.merge_partner_sample_size,
                "merge_strategy": self.merge_strategy,
                "random_state": self.random_state,
            },
        )

    def as_dict(self):
        return {
            "type": "information_loss",
            "k_ur": self.k_ur,
            "merge_partner_sample_size": self.merge_partner_sample_size,
            "merge_strategy": self.merge_strategy,
            "random_state": self.random_state,
            "initial_generalization": self.initial_generalization,
        }


__all__ = [
    "InformationLossBuilder",
    "KMeansClusterRingBuilder",
    "RandomSplitBuilder",
    "SampleLevelBuilder",
]
