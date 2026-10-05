"""Fitted Mondrian local recoding with shared QI hierarchies.

Release values are intervals or hierarchy nodes. Model vectors never determine
equivalence classes. Evaluation routing uses only the saved training tree.
"""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pandas as pd

from prototype.dataloader import (
    _coerce_categorical_leaf_value,
    _interval_union_mean,
    _parse_leaf_value,
)


MISSING_NODE = "__mondrian_missing__"


def _entropy(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(axis=-1, keepdims=True)
    probabilities = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    return -(probabilities * np.log2(np.maximum(probabilities, np.finfo(float).tiny))).sum(axis=-1)


class HierarchySchema:
    """Precompute audited leaf domains, descendants and model-column positions."""

    def __init__(self, generalization):
        self.generalization = generalization
        self.loader = generalization.data_loader
        self.attributes = tuple(self.loader.qi_attributes)
        self.numeric = set(self.loader.numeric_attributes) & set(self.attributes)
        self.nodes = {}
        self.leaves = {}
        self.descendants = {}
        self.children = {}
        self.child_lookup = {}
        self.roots = {}
        self.leaf_values = {}
        self.lookup = {}
        for attribute in self.attributes:
            node_list = generalization.trees[attribute]["nodes"]
            nodes = {node["id"]: node for node in node_list}
            if len(nodes) != len(node_list):
                raise ValueError(f"{attribute}: duplicate hierarchy node IDs.")
            roots = [key for key, node in nodes.items() if node.get("parent") is None]
            if len(roots) != 1 or MISSING_NODE in nodes:
                raise ValueError(f"{attribute}: expected one root and no reserved missing ID.")
            self.nodes[attribute] = nodes
            self.roots[attribute] = roots[0]
            leaves = [node for node in nodes.values() if node.get("kind") == "leaf"]
            self.leaves[attribute] = leaves
            self.children[attribute] = {key: [] for key in nodes}
            for child_id, child in nodes.items():
                if child.get("parent") is not None:
                    if child["parent"] not in nodes:
                        raise ValueError(f"{attribute}: missing hierarchy parent.")
                    self.children[attribute][child["parent"]].append(child_id)
            descendants = {key: [] for key in nodes}
            for position, leaf in enumerate(leaves):
                current = leaf["id"]
                visited = set()
                while current is not None:
                    if current in visited or current not in nodes:
                        raise ValueError(f"{attribute}: cyclic or invalid hierarchy parent.")
                    visited.add(current)
                    descendants[current].append(position)
                    current = nodes[current].get("parent")
                if roots[0] not in visited:
                    raise ValueError(f"{attribute}: disconnected leaf.")
            if any(not values for values in descendants.values()):
                raise ValueError(f"{attribute}: hierarchy has a node without leaves.")
            self.descendants[attribute] = descendants
            self.child_lookup[attribute] = {}
            for node_id, children in self.children[attribute].items():
                lookup = np.full(len(leaves), -1, dtype=np.int64)
                for child_position, child_id in enumerate(children):
                    lookup[descendants[child_id]] = child_position
                self.child_lookup[attribute][node_id] = lookup
            if attribute in self.numeric:
                parsed = [_parse_leaf_value(leaf["value"]) for leaf in leaves]
                intervals = [
                    (float(value[1]), float(value[2] if value[0] == "interval" else value[1]))
                    for value in parsed
                ]
                if any(lo > hi or not lo.is_integer() or not hi.is_integer() for lo, hi in intervals):
                    raise ValueError(f"{attribute}: numeric QIs require finite integer leaf domains.")
                ordered = sorted(intervals)
                if any(left[1] >= right[0] for left, right in zip(ordered, ordered[1:])):
                    raise ValueError(f"{attribute}: overlapping numeric leaves.")
                self.leaf_values[attribute] = intervals
            else:
                values = [
                    _coerce_categorical_leaf_value(self.loader, attribute, leaf["value"])
                    for leaf in leaves
                ]
                if len(set(values)) != len(values):
                    raise ValueError(f"{attribute}: duplicate categorical leaves.")
                self.leaf_values[attribute] = values
                self.lookup[attribute] = {value: position for position, value in enumerate(values)}

    def values(self, X):
        """Return numeric values or categorical leaf positions; missing is -1."""
        output = {}
        for attribute in self.attributes:
            column = X[attribute]
            if attribute in self.numeric:
                values = pd.to_numeric(column, errors="raise").to_numpy(dtype=float, na_value=np.nan)
                valid = np.isnan(values)
                for lo, hi in self.leaf_values[attribute]:
                    valid |= (values >= lo) & (values <= hi)
                if np.any(~valid) or np.any(np.isfinite(values) & (values != np.floor(values))):
                    raise ValueError(f"{attribute}: value outside the fixed integer leaf schema.")
                output[attribute] = values
            else:
                if attribute in getattr(self.loader, "string_categorical_attributes", ()):
                    column = column.astype("string")
                mapped = column.map(self.lookup[attribute])
                if (mapped.isna() & column.notna()).any():
                    bad = column.loc[mapped.isna() & column.notna()].unique()[:5].tolist()
                    raise ValueError(f"{attribute}: values outside the fixed hierarchy: {bad!r}")
                output[attribute] = mapped.fillna(-1).to_numpy(dtype=np.int64)
        return output

    def root_domain(self):
        domains = {}
        for attribute in self.attributes:
            if attribute in self.numeric:
                intervals = self.leaf_values[attribute]
                domains[attribute] = {
                    "node": self.roots[attribute],
                    "low": min(lo for lo, _ in intervals),
                    "high": max(hi for _, hi in intervals),
                    "missing": True,
                }
            else:
                domains[attribute] = {"node": self.roots[attribute], "missing": True}
        return domains

    def contains(self, attribute, domain, values):
        if attribute in self.numeric:
            missing = np.isnan(values)
            if domain.get("missing_only"):
                return missing
            return ((values >= domain["low"]) & (values <= domain["high"])) | (
                missing & domain.get("missing", False)
            )
        if domain["node"] == MISSING_NODE:
            return values == -1
        return np.isin(values, self.descendants[attribute][domain["node"]]) | (
            (values == -1) & domain.get("missing", False)
        )

    def numeric_representation(self, attribute, domain):
        if domain.get("missing_only"):
            return np.nan, [], 0
        lo, hi = domain["low"], domain["high"]
        intervals = self.leaf_values[attribute]
        positions = [i for i, (a, b) in enumerate(intervals) if a <= hi and b >= lo]
        if not positions:
            raise ValueError(f"{attribute}: released range contains no hierarchy leaves.")
        covered = [(max(lo, intervals[i][0]), min(hi, intervals[i][1])) for i in positions]
        # Scalar-leaf trees may be sparse; do not invent integers in the gaps.
        if all(a == b for a, b in intervals):
            mean = sum(intervals[i][0] for i in positions) / len(positions)
        else:
            mean = _interval_union_mean(covered)
        if "node" in domain:
            # The global fallback publishes the actual numeric hierarchy root,
            # including its height when the tree has a unary root.
            level = int(self.nodes[attribute][domain["node"]]["height_from_leaf"])
        else:
            common = set(self.nodes[attribute])
            for position in positions:
                ancestors = set()
                current = self.leaves[attribute][position]["id"]
                while current is not None:
                    ancestors.add(current)
                    current = self.nodes[attribute][current].get("parent")
                common &= ancestors
            level = min(int(self.nodes[attribute][key]["height_from_leaf"]) for key in common)
        return mean, positions, level

    def encode(self, X, groups, *, family):
        """Replace QI blocks in the existing level-0 schema; retain non-QIs."""
        if family == "mlp":
            encoded = self.generalization.encode(X).copy()
            for group in groups:
                row_ids = X.index.take(group["positions"])
                for attribute, domain in group["release"].items():
                    if attribute in self.numeric:
                        mean, _, _ = self.numeric_representation(attribute, domain)
                        encoded.loc[row_ids, attribute] = mean
                    else:
                        categories = self.loader.category_maps[attribute]
                        columns = [f"{attribute}={value}" for value in categories]
                        vector = np.zeros(len(categories), dtype=np.float64)
                        if domain["node"] != MISSING_NODE:
                            positions = self.descendants[attribute][domain["node"]]
                            for position in positions:
                                vector[categories[self.leaf_values[attribute][position]]] = 1.0 / len(positions)
                        encoded.loc[row_ids, columns] = vector
            return encoded
        if family != "xgboost":
            raise ValueError("Mondrian supports the shared mlp and xgboost encodings.")
        encoded = self.generalization.encode_xgboost_leaf_space(X).toarray()
        offset = 0
        for attribute in self.loader.feature_columns:
            if attribute not in self.attributes:
                offset += 1 if attribute in self.loader.numeric_attributes else len(self.loader.category_maps[attribute])
                continue
            width = len(self.leaves[attribute])
            for group in groups:
                rows = np.asarray(group["positions"], dtype=np.intp)
                domain = group["release"][attribute]
                if attribute in self.numeric:
                    _, positions, level = self.numeric_representation(attribute, domain)
                elif domain["node"] == MISSING_NODE:
                    positions, level = [], 0
                else:
                    positions = self.descendants[attribute][domain["node"]]
                    level = int(self.nodes[attribute][domain["node"]]["height_from_leaf"])
                encoded[np.ix_(rows, np.arange(offset, offset + width + 1))] = 0.0
                if positions:
                    encoded[np.ix_(rows, offset + np.asarray(positions))] = 1.0 / len(positions)
                encoded[rows, offset + width] = level
            offset += width + 1
        if offset != encoded.shape[1]:
            raise AssertionError("Shared XGBoost schema width changed.")
        return np.ascontiguousarray(encoded, dtype=np.float32)


class FittedMondrian:
    """Median or label-InfoGain splits fitted exclusively to training records."""

    def __init__(self, schema, *, k, variant):
        if variant not in {"median", "infogain"}:
            raise ValueError("variant must be median or infogain.")
        if isinstance(k, bool) or int(k) != k or k < 1:
            raise ValueError("k must be a positive integer.")
        self.schema, self.k, self.variant = schema, int(k), variant
        self.nodes = []
        self.partitions = []

    def fit(self, X, y):
        if not X.index.is_unique or len(X) < self.k:
            raise ValueError("Training IDs must be unique and N_train must be at least k.")
        if not isinstance(y, pd.Series) or not y.index.equals(X.index) or y.isna().any():
            raise ValueError("Labels must be nonmissing and aligned with training IDs.")
        values = self.schema.values(X)
        labels, classes = pd.factorize(y, sort=True)
        class_count = len(classes)
        root_domain = self.schema.root_domain()
        # A separate hierarchy-root fallback preserves the observed release
        # even when the training partition cannot be split (e.g. K=N_train).
        self.nodes = [{
            "id": 0, "kind": "root", "depth": 0, "size": len(X),
            "region": deepcopy(root_domain), "children": [],
        }]
        self.partitions = []
        self.training_ids = X.index.tolist()
        pending = [(np.arange(len(X)), root_domain, 0, {})]
        while pending:
            positions, domain, parent_id, branch = pending.pop()
            node_id = len(self.nodes)
            region = deepcopy(domain)
            for attribute in self.schema.numeric:
                observed = values[attribute][positions]
                present = observed[np.isfinite(observed)]
                if len(present):
                    region[attribute] = {
                        "low": float(present.min()), "high": float(present.max()),
                        "missing": bool(np.isnan(observed).any()),
                    }
                else:
                    region[attribute] = {"missing_only": True}
            node = {
                "id": node_id, "domain": deepcopy(domain), "region": region,
                "depth": self.nodes[parent_id]["depth"] + 1,
                "children": [], "size": len(positions),
            }
            self.nodes.append(node)
            self.nodes[parent_id]["children"].append({"node_id": node_id, **branch})
            parent_counts = np.bincount(labels[positions], minlength=class_count)
            parent_entropy = float(_entropy(parent_counts))
            candidates = []
            unary = None
            for order, attribute in enumerate(self.schema.attributes):
                observed = values[attribute][positions]
                if attribute in self.schema.numeric:
                    present_mask = ~np.isnan(observed)
                    missing_rows = positions[~present_mask]
                    sorted_rows = positions[present_mask][np.argsort(observed[present_mask], kind="stable")]
                    sorted_values = values[attribute][sorted_rows]
                    boundaries = np.flatnonzero(sorted_values[:-1] != sorted_values[1:]) + 1
                    cuts = boundaries[(boundaries >= self.k) & (len(sorted_rows) - boundaries >= self.k)]
                    if not len(cuts) or (0 < len(missing_rows) < self.k):
                        continue
                    width = (sorted_values[-1] - sorted_values[0]) / max(
                        root_domain[attribute]["high"] - root_domain[attribute]["low"], 1.0
                    )
                    median_cut = int(cuts[np.argmin(np.abs(cuts - len(sorted_rows) / 2))])
                    best_cut, best_gain = median_cut, 0.0
                    if self.variant == "infogain":
                        counts = np.zeros((len(sorted_rows), class_count), dtype=np.int64)
                        counts[np.arange(len(sorted_rows)), labels[sorted_rows]] = 1
                        prefix = np.cumsum(counts, axis=0)
                        left = prefix[cuts - 1]
                        right = prefix[-1] - left
                        weighted = (cuts * _entropy(left) + (len(sorted_rows) - cuts) * _entropy(right))
                        if len(missing_rows):
                            missing_counts = np.bincount(labels[missing_rows], minlength=class_count)
                            weighted += len(missing_rows) * _entropy(missing_counts)
                        gains = parent_entropy - weighted / len(positions)
                        best_gain = float(gains.max())
                        tied = np.flatnonzero(np.isclose(gains, best_gain, rtol=0, atol=1e-12))
                        best_cut = int(cuts[tied[np.argmin(np.abs(cuts[tied] - len(sorted_rows) / 2))]])
                    for cut, gain, mode in [(median_cut, 0.0, "median"), (best_cut, best_gain, "infogain")]:
                        if mode == "infogain" and self.variant != "infogain":
                            continue
                        threshold = float((sorted_values[cut - 1] + sorted_values[cut]) / 2)
                        low_domain, high_domain = deepcopy(domain), deepcopy(domain)
                        low_domain[attribute] = {
                            "low": domain[attribute]["low"], "high": math.floor(threshold), "missing": False,
                        }
                        high_domain[attribute] = {
                            "low": math.floor(threshold) + 1, "high": domain[attribute]["high"], "missing": False,
                        }
                        children = [
                            (sorted_rows[:cut], low_domain, {"side": "left"}),
                            (sorted_rows[cut:], high_domain, {"side": "right"}),
                        ]
                        if len(missing_rows):
                            missing_domain = deepcopy(domain)
                            missing_domain[attribute] = {"missing_only": True}
                            children.append((missing_rows, missing_domain, {"side": "missing"}))
                        candidates.append({
                            "attribute": attribute, "kind": "numeric", "threshold": threshold,
                            "width": width, "order": order, "gain": gain, "mode": mode, "branches": children,
                        })
                else:
                    current = domain[attribute]["node"]
                    if current == MISSING_NODE:
                        continue
                    child_ids = self.schema.children[attribute][current]
                    if not child_ids:
                        continue
                    child_codes = self.schema.child_lookup[attribute][current][np.maximum(observed, 0)].copy()
                    child_codes[observed == -1] = len(child_ids)
                    occupied, counts = np.unique(child_codes, return_counts=True)
                    if np.any(occupied < 0):
                        raise AssertionError("Training values do not belong to their fitted categorical domain.")
                    if np.any(counts < self.k):
                        continue
                    children = []
                    for child_position in occupied:
                        child = MISSING_NODE if child_position == len(child_ids) else child_ids[child_position]
                        child_domain = deepcopy(domain)
                        child_domain[attribute] = {"node": child, "missing": child == MISSING_NODE}
                        children.append((
                            positions[child_codes == child_position], child_domain,
                            {"category_node": child},
                        ))
                    candidate = {
                        "attribute": attribute, "kind": "categorical", "order": order, "mode": "both",
                        "width": len(np.unique(observed)) / len(self.schema.leaves[attribute]),
                        "gain": parent_entropy - sum(
                            len(rows) * float(_entropy(np.bincount(labels[rows], minlength=class_count)))
                            for rows, _, _ in children
                        ) / len(positions),
                        "branches": children,
                    }
                    if len(children) == 1:
                        unary = candidate
                        break
                    candidates.append(candidate)
            chosen = unary
            if chosen is None and self.variant == "infogain":
                informative = [c for c in candidates if c["mode"] != "median" and c["gain"] > 1e-12]
                if informative:
                    chosen = max(informative, key=lambda c: (c["gain"], c["width"], -c["order"]))
            if chosen is None:
                medians = [c for c in candidates if c["mode"] != "infogain"]
                if medians:
                    chosen = max(medians, key=lambda c: (c["width"], -c["order"]))
            if chosen is not None:
                node.update({key: chosen[key] for key in ("attribute", "kind")})
                if chosen["kind"] == "numeric":
                    node["threshold"] = chosen["threshold"]
                node["information_gain"] = float(chosen["gain"])
                for child_rows, child_domain, child_branch in reversed(chosen["branches"]):
                    pending.append((child_rows, child_domain, node_id, child_branch))
                continue
            release = deepcopy(region)
            node.update({"kind": "leaf", "release": release})
            partition = {
                "node_id": node_id, "positions": positions.tolist(), "size": len(positions),
                "row_ids": X.index.take(positions).tolist(), "release": release,
            }
            self.partitions.append(partition)
        covered = [position for group in self.partitions for position in group["positions"]]
        if sorted(covered) != list(range(len(X))) or any(group["size"] < self.k for group in self.partitions):
            raise AssertionError("Mondrian must publish every training record exactly once with size >= k.")
        return self

    def transform_groups(self, X):
        """Stop at the smallest fitted region containing every true QI value."""
        if not self.nodes:
            raise ValueError("Fit Mondrian before transforming evaluation records.")
        values = self.schema.values(X)
        groups = []
        pending = [(0, np.arange(len(X)))]
        while pending:
            node_id, positions = pending.pop()
            if not len(positions):
                continue
            node = self.nodes[node_id]
            remaining = np.ones(len(positions), dtype=bool)
            for child in node["children"]:
                child_region = self.nodes[child["node_id"]]["region"]
                mask = np.ones(len(positions), dtype=bool)
                for attribute, published_value in child_region.items():
                    mask &= self.schema.contains(attribute, published_value, values[attribute][positions])
                if np.any(mask & ~remaining):
                    raise AssertionError("Sibling Mondrian regions must not overlap.")
                remaining &= ~mask
                if mask.any():
                    pending.append((child["node_id"], positions[mask]))
            if remaining.any():
                stopped = positions[remaining]
                for attribute, published_value in node["region"].items():
                    if not self.schema.contains(attribute, published_value, values[attribute][stopped]).all():
                        raise AssertionError("A stopped record must belong to its entire fallback region.")
                groups.append({
                    "positions": stopped.tolist(), "release": node["region"],
                    "node_id": node_id, "depth": node["depth"],
                    "stop_kind": "root" if node_id == 0 else "leaf" if node["kind"] == "leaf" else "internal",
                    "fallback": node["kind"] != "leaf",
                })
        return groups

    def routing_stats(self, groups, population_size, *, review_threshold=0.01):
        """Counts at terminal leaves, internal regions and the global root."""
        if not 0 <= review_threshold <= 1:
            raise ValueError("Routing review threshold must be in [0, 1].")
        positions = [position for group in groups for position in group["positions"]]
        if sorted(positions) != list(range(population_size)):
            raise AssertionError("Evaluation routing must assign each record exactly once.")
        counts = {"leaf": 0, "internal": 0, "root": 0}
        assignment_counts, depth_counts = {}, {}
        for group in groups:
            count = len(group["positions"])
            counts[group["stop_kind"]] += count
            node_id, depth = str(group["node_id"]), str(group["depth"])
            assignment_counts[node_id] = assignment_counts.get(node_id, 0) + count
            depth_counts[depth] = depth_counts.get(depth, 0) + count
        fallback_count = counts["internal"] + counts["root"]
        fraction = fallback_count / population_size if population_size else 0.0
        return {
            "population_size": population_size,
            "leaf_count": counts["leaf"], "internal_count": counts["internal"],
            "root_count": counts["root"], "fallback_count": fallback_count,
            "fallback_fraction": fraction, "review_threshold": review_threshold,
            "review_needed": fraction > review_threshold,
            "eval_assignment_counts": assignment_counts,
            "stop_depth_counts": depth_counts,
            "eval_coarsest_release_row_count": counts["root"],
        }

    def per_record_k(self, index):
        """Group canonical published QI values, including any coincident cells."""
        import json

        keys = {}
        for partition in self.partitions:
            key = json.dumps(partition["release"], sort_keys=True, separators=(",", ":"))
            keys.setdefault(key, []).extend(partition["row_ids"])
        result = pd.Series(0, index=index, dtype=np.int64)
        for row_ids in keys.values():
            result.loc[row_ids] = len(row_ids)
        if (result < self.k).any():
            raise AssertionError("Published QI equivalence classes do not meet requested k.")
        return result

    def to_dict(self):
        return {
            "schema_version": "hierarchy_mondrian.v2", "variant": self.variant, "k": self.k,
            "row_semantics": "every_training_row_once", "suppression": False,
            "unseen_policy": "smallest_containing_node_region_all_qis",
            "root_region": "all_qis_at_hierarchy_roots",
            "training_row_ids": self.training_ids, "nodes": self.nodes,
            "partitions": self.partitions,
        }

    @classmethod
    def from_dict(cls, schema, payload):
        if payload.get("schema_version") != "hierarchy_mondrian.v2":
            raise ValueError("Unsupported fitted Mondrian schema.")
        fitted = cls(schema, k=payload["k"], variant=payload["variant"])
        fitted.nodes = deepcopy(payload["nodes"])
        fitted.partitions = deepcopy(payload["partitions"])
        fitted.training_ids = list(payload["training_row_ids"])
        return fitted
