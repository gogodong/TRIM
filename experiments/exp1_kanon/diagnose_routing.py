"""Explain validation fallback using a saved tree, without changing routing."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import resolve_project_path, write_json
from prototype.TRIM_prototype_pipeline import _loaded_input_sha256
from prototype.dataloader import load_generalization_rules_from_file
from prototype.dataset_registry import build_data_loader, resolve_generalization_tree
from prototype.release_artifacts import load_dataset_split

from .artifacts import load_release_payload
from .mondrian import FittedMondrian, HierarchySchema


def diagnose_routing(fitted, training_frame, evaluation_frame):
    """Attribute actual stops to missing branches or observed-range exclusions.

    Geometric fitting domains are inspected only as a counterfactual diagnostic.
    They never replace the observed regions in evaluation/model inputs.
    """
    schema = fitted.schema
    values = schema.values(evaluation_frame)
    globally_seen = {
        attribute: evaluation_frame[attribute].isin(training_frame[attribute]).to_numpy()
        for attribute in schema.attributes
    }
    groups = fitted.transform_groups(evaluation_frame)
    training_groups = fitted.transform_groups(training_frame)
    training_stats = fitted.routing_stats(training_groups, len(training_frame))
    if training_stats["fallback_count"]:
        raise AssertionError("Training records must still reach their published leaves.")
    expected_leaves = np.empty(len(training_frame), dtype=np.intp)
    for partition in fitted.partitions:
        expected_leaves[partition["positions"]] = partition["node_id"]
    for group in training_groups:
        if np.any(expected_leaves[group["positions"]] != group["node_id"]):
            raise AssertionError("Training routing changed the published partition assignment.")

    causes, excluded_attributes, other_numeric_attributes = Counter(), Counter(), Counter()
    empty_branch_attributes = Counter()
    empty_branch_value_seen_elsewhere_count = 0
    samples, stopped_sizes, stopped_depths = [], [], []
    for group in groups:
        if not group["fallback"]:
            continue
        node = fitted.nodes[group["node_id"]]
        positions = np.asarray(group["positions"], dtype=np.intp)
        stopped_sizes.extend([node["size"]] * len(positions))
        stopped_depths.extend([node["depth"]] * len(positions))
        matched_domain = np.zeros(len(positions), dtype=bool)
        for child in node["children"]:
            child_node = fitted.nodes[child["node_id"]]
            domain_matches = np.ones(len(positions), dtype=bool)
            for attribute, domain in child_node["domain"].items():
                domain_matches &= schema.contains(attribute, domain, values[attribute][positions])
            if np.any(domain_matches & matched_domain):
                raise AssertionError("Fitting domains unexpectedly overlap.")
            matched_domain |= domain_matches
            selected = positions[domain_matches]
            if not len(selected):
                continue
            reasons = [[] for _ in selected]
            failures = {}
            for attribute, region in child_node["region"].items():
                fails = ~schema.contains(attribute, region, values[attribute][selected])
                if not fails.any():
                    continue
                failures[attribute] = fails
                excluded_attributes[attribute] += int(fails.sum())
                if node["id"] == 0:
                    reason = "outside_observed_training_root"
                elif attribute in schema.numeric:
                    if attribute == node["attribute"]:
                        reason = "split_numeric_observed_range_gap"
                    else:
                        reason = "non_split_numeric_observed_range_shrink"
                        other_numeric_attributes[attribute] += int(fails.sum())
                else:
                    reason = "unexpected_non_split_categorical_exclusion"
                for index in np.flatnonzero(fails):
                    reasons[index].append(reason)
            for index, reason_list in enumerate(reasons):
                if not reason_list:
                    raise AssertionError("A stopped record matched the whole child region.")
                cause = "+".join(sorted(set(reason_list)))
                causes[cause] += 1
                if len(samples) < 8:
                    position = int(selected[index])
                    samples.append({
                        "evaluation_row_id": evaluation_frame.index[position].item()
                        if hasattr(evaluation_frame.index[position], "item") else evaluation_frame.index[position],
                        "stopped_node_id": node["id"], "stopped_node_size": node["size"],
                        "split_attribute": node.get("attribute"), "candidate_child_id": child_node["id"],
                        "cause": cause,
                        "failed_attributes": {
                            attribute: {
                                "true_value": evaluation_frame.iloc[position][attribute].item()
                                if hasattr(evaluation_frame.iloc[position][attribute], "item")
                                else evaluation_frame.iloc[position][attribute],
                                "child_region": child_node["region"][attribute],
                                "parent_region": node["region"][attribute],
                            }
                            for attribute, fails in failures.items() if fails[index]
                        },
                    })
        unmatched_count = int((~matched_domain).sum())
        if unmatched_count:
            if node.get("attribute") not in schema.numeric and node["id"] != 0:
                causes["unoccupied_categorical_branch"] += unmatched_count
                empty_branch_attributes[node["attribute"]] += unmatched_count
                empty_branch_value_seen_elsewhere_count += int(
                    globally_seen[node["attribute"]][positions[~matched_domain]].sum(),
                )
            else:
                causes["unobserved_numeric_missing_branch"] += unmatched_count

    # Demonstrate whether threshold/domain routing would merely hide a true
    # region mismatch. This diagnostic never trains/scores a different policy.
    domain_leaf_count, domain_leaf_exclusions = 0, 0
    pending = [(0, np.arange(len(evaluation_frame)))]
    while pending:
        node_id, positions = pending.pop()
        node = fitted.nodes[node_id]
        if node["kind"] == "leaf":
            domain_leaf_count += len(positions)
            contains = np.ones(len(positions), dtype=bool)
            for attribute, region in node["region"].items():
                contains &= schema.contains(attribute, region, values[attribute][positions])
            domain_leaf_exclusions += int((~contains).sum())
        for child in node["children"]:
            mask = np.ones(len(positions), dtype=bool)
            for attribute, domain in fitted.nodes[child["node_id"]]["domain"].items():
                mask &= schema.contains(attribute, domain, values[attribute][positions])
            if mask.any():
                pending.append((child["node_id"], positions[mask]))

    routing = fitted.routing_stats(groups, len(evaluation_frame))
    if sum(causes.values()) != routing["fallback_count"]:
        raise AssertionError("Fallback causes must cover every stopped record exactly once.")
    qi = list(schema.attributes)
    seen = pd.MultiIndex.from_frame(evaluation_frame[qi]).isin(pd.MultiIndex.from_frame(training_frame[qi]))
    quantiles = (0, .25, .5, .75, 1)
    return {
        "variant": fitted.variant, "k": fitted.k, "evaluation_split": "validation",
        "routing_policy_changed": False, "training": training_stats, "validation": routing,
        "partition_count": len(fitted.partitions), "exclusive_fallback_causes": dict(causes),
        "excluded_attribute_counts_nonexclusive": dict(excluded_attributes),
        "non_split_numeric_exclusion_counts_nonexclusive": dict(other_numeric_attributes),
        "unoccupied_categorical_branch_attribute_counts": dict(empty_branch_attributes),
        "unoccupied_branch_value_seen_elsewhere_in_training_count": empty_branch_value_seen_elsewhere_count,
        "observed_value_absent_from_training_counts": {
            attribute: int((~globally_seen[attribute]).sum())
            for attribute in schema.attributes
        },
        "exact_raw_qi_combination_seen_count": int(seen.sum()),
        "exact_raw_qi_combination_unseen_count": int((~seen).sum()),
        "fallback_region_training_size_quantiles": dict(zip(
            map(str, quantiles), np.quantile(stopped_sizes, quantiles).tolist(),
        )) if stopped_sizes else {},
        "fallback_depth_quantiles": dict(zip(
            map(str, quantiles), np.quantile(stopped_depths, quantiles).tolist(),
        )) if stopped_depths else {},
        "counterfactual_fitting_domain_routing": {
            "leaf_count": domain_leaf_count, "leaf_region_exclusion_count": domain_leaf_exclusions,
            "used_for_model_evaluation": False,
        },
        "examples": samples,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-path", help="Optional local dataset path override.")
    args = parser.parse_args(argv)
    task_manifest = json.loads((Path(args.task_dir) / "config.json").read_text())
    task = task_manifest["task"]
    loader = build_data_loader(task["dataset"], data_path=resolve_project_path(args.data_path or task.get("data_path")))
    split = load_dataset_split(loader, task.get("pipeline") or {})
    if (split.X_train_raw.index.tolist() != task_manifest["train_row_ids"]
            or split.X_val_raw.index.tolist() != task_manifest["validation_row_ids"]):
        raise ValueError("Diagnostic split IDs differ from the saved release.")
    protocol = task_manifest["protocol"]
    if _loaded_input_sha256(loader.X_raw, loader.y) != protocol["loaded_input_sha256"]:
        raise ValueError("Diagnostic input data differ from the saved release.")
    tree_path = resolve_generalization_tree(task["dataset"], resolve_project_path(task.get("generalization_tree_path")))
    if hashlib.sha256(tree_path.read_bytes()).hexdigest() != protocol["tree_sha256"]:
        raise ValueError("Diagnostic hierarchy differs from the saved release.")
    schema = HierarchySchema(load_generalization_rules_from_file(tree_path, loader, generalization_level=0))
    payload = load_release_payload(args.release)
    fitted = FittedMondrian.from_dict(schema, payload)
    if fitted.training_ids != split.X_train_raw.index.tolist():
        raise ValueError("Saved tree has different training rows.")
    report = diagnose_routing(fitted, split.X_train_raw, split.X_val_raw)
    report.update(release_path=str(Path(args.release).resolve()), tree_sha256=protocol["tree_sha256"])
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in (
        "variant", "k", "exclusive_fallback_causes", "non_split_numeric_exclusion_counts_nonexclusive",
        "counterfactual_fitting_domain_routing",
    )}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
