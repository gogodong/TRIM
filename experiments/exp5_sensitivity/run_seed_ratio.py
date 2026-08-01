"""Run initial-sample-ratio sensitivity with TRIM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .._runner import require_core_parameters, run_matrix_config
from ..common import write_json


def _require_trajectory_artifacts(task, summary, _experiment_dir):
    report_tolerance = task.get("report_tolerance")
    if report_tolerance is None:
        raise ValueError(f"Task {summary['run_id']!r} is missing report_tolerance.")
    report_tolerance = float(report_tolerance)
    if report_tolerance < 0:
        raise ValueError("report_tolerance must be non-negative.")

    iterations_path = Path(summary["run_dir"]) / "trim_iterations.json"
    payload = json.loads(iterations_path.read_text(encoding="utf-8"))
    rows = payload.get("iterations")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No TRIM trajectory was recorded in {iterations_path}.")
    missing_test = [
        int(row.get("iteration", -1)) for row in rows if row.get("test_loss") is None
    ]
    if missing_test:
        raise ValueError(
            "Per-iteration downstream test loss is missing for iterations "
            f"{missing_test} in {iterations_path}. The seed-ratio result cannot "
            "be selected from validation loss."
        )
    if summary.get("experiment_observation_time_seconds") is None:
        raise ValueError(
            "The TRIM run is missing experiment_observation_time_seconds, "
            "which records trajectory observation cost."
        )
    baseline_test_loss = summary.get("baseline_test_loss")
    if baseline_test_loss is None:
        raise ValueError("The completed run is missing baseline_test_loss.")
    threshold = float(baseline_test_loss) + report_tolerance
    feasible = [row for row in rows if float(row["test_loss"]) <= threshold]
    if not feasible:
        raise ValueError(
            f"No TRIM iteration meets test loss <= {threshold:.12g} in "
            f"{iterations_path}."
        )
    selected = min(feasible, key=lambda row: int(row["iteration"]))
    selected_summary = {
        "schema_version": "trim_seed_ratio_point.v1",
        "run_id": summary["run_id"],
        "initial_sample_fraction": task.get("initial_sample_fraction"),
        "report_tolerance": report_tolerance,
        "baseline_test_loss": float(baseline_test_loss),
        "test_loss_threshold": threshold,
        "selected_iteration": int(selected["iteration"]),
        "selected_test_loss": float(selected["test_loss"]),
        "selected_test_delta_u": float(selected["test_delta_u"]),
        "selected_min_k": int(selected["min_k"]),
        "selected_target_k": int(selected["target_k"]),
        "selected_row_count": int(selected["row_count"]),
        "selected_suppressed_row_count": int(selected["suppressed_row_count"]),
        "selected_coverage_fraction": float(selected["coverage_fraction"]),
    }
    write_json(
        Path(summary["run_dir"]) / "paper_seed_ratio_point.json",
        selected_summary,
    )
    return {
        "report_tolerance": report_tolerance,
        "trajectory_point_count": len(rows),
        "report_selected_iteration": selected_summary["selected_iteration"],
        "report_selected_test_loss": selected_summary["selected_test_loss"],
        "report_selected_min_k": selected_summary["selected_min_k"],
        "report_selected_coverage_fraction": selected_summary[
            "selected_coverage_fraction"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Exp-5 initial-sample-ratio sensitivity."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--run-label")
    args = parser.parse_args()

    require_core_parameters("stop_on_utility", "record_iteration_test_metrics")
    output = run_matrix_config(
        args.config,
        expected_kind="exp5_seed_ratio",
        device=args.device,
        run_label=args.run_label,
        after_task=_require_trajectory_artifacts,
    )
    print(output)


if __name__ == "__main__":
    main()
