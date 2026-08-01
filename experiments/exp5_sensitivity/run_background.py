"""Measure re-identification risk as known QI count increases."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from prototype.dataset_registry import (
    build_data_loader,
    resolve_generalization_tree,
)
from prototype.release_artifacts import (
    load_trim_release,
    materialize_trim_release,
)

from .._runner import run_matrix_config
from ..common import (
    load_yaml_mapping,
    resolve_project_path,
    write_csv_rows,
    write_json,
)
from ..run_matrix_sweep import expand_matrix


def _validate_release_selection(config_path):
    _resolved_path, config = load_yaml_mapping(config_path)
    tasks = expand_matrix(config)
    unsupported = [
        task["run_id"]
        for task in tasks
        if str(task.get("release_selection", "")).strip() != "final"
    ]
    if unsupported:
        raise ValueError(
            "This entry supports only the final mixed-level TRIM release. "
            "Set release_selection: final for every task. Selecting the first "
            "test-feasible intermediate release also requires a declared "
            "per-iteration downstream test-loss observation policy; unsupported "
            "runs: "
            f"{unsupported}."
        )


def _attribute_code_frame(data_loader, published_frame):
    attributes = tuple(
        getattr(data_loader, "qi_attributes", data_loader.feature_columns)
    )
    numeric_attributes = set(getattr(data_loader, "numeric_attributes", ()) or ())
    code_columns = {}
    for attribute in attributes:
        if attribute in numeric_attributes:
            encoded_columns = [attribute] if attribute in published_frame.columns else []
        else:
            prefix = f"{attribute}="
            encoded_columns = [
                column
                for column in published_frame.columns
                if str(column).startswith(prefix)
            ]
        if not encoded_columns:
            raise ValueError(
                f"The materialized release has no encoded columns for {attribute!r}."
            )
        if len(encoded_columns) == 1:
            codes, _unique_values = pd.factorize(
                published_frame[encoded_columns[0]],
                sort=False,
            )
        else:
            keys = pd.MultiIndex.from_frame(published_frame.loc[:, encoded_columns])
            codes, _unique_values = pd.factorize(keys, sort=False)
        code_columns[attribute] = codes.astype(np.int64, copy=False)
    return pd.DataFrame(code_columns, index=published_frame.index), attributes


def _summarize_known_attributes(
    code_frame,
    attributes,
    run_id,
    known_attribute_counts,
):
    rows = []
    for known_count in known_attribute_counts:
        subset_count = 0
        worst_candidate_size = math.inf
        worst_success_probability = -math.inf
        worst_subset = None
        worst_row_id = None
        strongest_mean_success = -math.inf
        strongest_mean_subset = None
        strongest_p95_success = -math.inf
        strongest_p95_subset = None
        mean_success_values = []
        p95_success_values = []

        for subset in itertools.combinations(attributes, known_count):
            subset_count += 1
            group_ids = code_frame.groupby(
                list(subset),
                sort=False,
                dropna=False,
            ).ngroup()
            group_ids = group_ids.to_numpy(dtype=np.int64, copy=False)
            counts = np.bincount(group_ids)
            candidate_sizes = counts[group_ids]
            probabilities = 1.0 / candidate_sizes.astype(float)
            min_size = int(candidate_sizes.min())
            max_probability = float(probabilities.max())
            mean_success = float(probabilities.mean())
            p95_success = float(np.percentile(probabilities, 95))
            mean_success_values.append(mean_success)
            p95_success_values.append(p95_success)

            if min_size < worst_candidate_size:
                worst_candidate_size = min_size
                worst_success_probability = max_probability
                worst_subset = subset
                worst_row_id = str(
                    code_frame.index[int(np.argmin(candidate_sizes))]
                )
            if mean_success > strongest_mean_success:
                strongest_mean_success = mean_success
                strongest_mean_subset = subset
            if p95_success > strongest_p95_success:
                strongest_p95_success = p95_success
                strongest_p95_subset = subset

        rows.append(
            {
                "run_id": run_id,
                "method": "TRIM",
                "release_selection": "final",
                "known_attribute_count": known_count,
                "subset_count": subset_count,
                "worst_candidate_size": int(worst_candidate_size),
                "worst_success_probability": float(worst_success_probability),
                "worst_subset": ",".join(worst_subset or ()),
                "worst_row_id": worst_row_id,
                "strongest_mean_success_probability": float(
                    strongest_mean_success
                ),
                "strongest_mean_subset": ",".join(
                    strongest_mean_subset or ()
                ),
                "mean_success_probability_across_subsets": float(
                    np.mean(mean_success_values)
                ),
                "strongest_p95_success_probability": float(
                    strongest_p95_success
                ),
                "strongest_p95_subset": ",".join(
                    strongest_p95_subset or ()
                ),
                "p95_success_probability_across_subsets": float(
                    np.mean(p95_success_values)
                ),
            }
        )
    return rows


def _write_background_summary(task, summary, _experiment_dir):
    report_tolerance = task.get("report_tolerance")
    if report_tolerance is None:
        raise ValueError(f"Task {summary['run_id']!r} is missing report_tolerance.")
    report_tolerance = float(report_tolerance)
    if report_tolerance < 0:
        raise ValueError("report_tolerance must be non-negative.")
    baseline_test_loss = summary.get("baseline_test_loss")
    final_test_loss = summary.get("final_actual_model_loss")
    if baseline_test_loss is None or final_test_loss is None:
        raise ValueError("The completed run is missing baseline or final test loss.")
    test_loss_threshold = float(baseline_test_loss) + report_tolerance
    if float(final_test_loss) > test_loss_threshold:
        raise ValueError(
            f"Final TRIM release for {summary['run_id']!r} has test loss "
            f"{float(final_test_loss):.12g}, above the reporting threshold "
            f"{test_loss_threshold:.12g}."
        )

    data_loader = build_data_loader(
        str(task["dataset"]),
        data_path=resolve_project_path(task.get("data_path")),
    )
    tree_path = resolve_generalization_tree(
        str(task["dataset"]),
        tree_path=resolve_project_path(task.get("generalization_tree_path")),
    )
    run_dir = Path(summary["run_dir"])
    loaded_release = load_trim_release(run_dir)
    materialized = materialize_trim_release(
        data_loader,
        tree_path,
        loaded_release,
        evaluation_split="test",
    )
    code_frame, attributes = _attribute_code_frame(
        data_loader,
        materialized.published_frame,
    )
    known_attribute_counts = [
        int(value) for value in task.get("known_attribute_counts") or []
    ]
    if not known_attribute_counts:
        raise ValueError("known_attribute_counts must be explicitly declared.")
    if len(known_attribute_counts) != len(set(known_attribute_counts)):
        raise ValueError("known_attribute_counts cannot contain duplicates.")
    if any(not 1 <= value <= len(attributes) for value in known_attribute_counts):
        raise ValueError(
            f"known_attribute_counts must be between 1 and {len(attributes)}."
        )
    rows = _summarize_known_attributes(
        code_frame,
        attributes,
        summary["run_id"],
        known_attribute_counts,
    )
    write_csv_rows(run_dir / "background_knowledge.csv", rows)
    write_json(
        run_dir / "background_knowledge.json",
        {
            "schema_version": "trim_background_knowledge.v1",
            "run_id": summary["run_id"],
            "release_selection": "final",
            "report_tolerance": report_tolerance,
            "baseline_test_loss": float(baseline_test_loss),
            "test_loss_threshold": test_loss_threshold,
            "selected_release_test_loss": float(final_test_loss),
            "published_row_count": int(len(code_frame)),
            "release_min_k": int(materialized.release["min_k"]),
            "qi_attributes": list(attributes),
            "known_attribute_counts": known_attribute_counts,
            "rows": rows,
        },
    )
    return {
        "report_tolerance": report_tolerance,
        "release_selection": "final",
        "background_known_attribute_points": len(rows),
        "background_published_row_count": int(len(code_frame)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Exp-5 background-knowledge sensitivity."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--run-label")
    args = parser.parse_args()

    _validate_release_selection(args.config)
    output = run_matrix_config(
        args.config,
        expected_kind="exp5_background",
        device=args.device,
        run_label=args.run_label,
        after_task=_write_background_summary,
    )
    print(output)


if __name__ == "__main__":
    main()
