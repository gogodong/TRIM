"""Run the paper's privacy--utility matrix with TRIM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .._runner import require_core_parameters, run_matrix_config
from ..common import append_jsonl, write_csv_rows


def _require_trajectory_artifacts(_task, summary, _experiment_dir):
    run_dir = Path(summary["run_dir"])
    iterations_path = run_dir / "trim_iterations.json"
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
            f"{missing_test} in {iterations_path}."
        )
    observation_time = summary.get("experiment_observation_time_seconds")
    if observation_time is None:
        raise ValueError(
            "The TRIM run is missing experiment_observation_time_seconds, "
            "which records trajectory retraining time."
        )
    flat_rows = []
    for row in rows:
        flat = {
            "run_id": summary["run_id"],
            "dataset": summary["dataset"],
            "model_name": summary.get("model_name"),
            "seed": summary["random_state"],
            "iteration": int(row["iteration"]),
            "action": row.get("action"),
            "attribute": row.get("attribute"),
            "validation_loss": row.get("validation_loss"),
            "validation_delta_u": row.get("delta_u"),
            "test_loss": row.get("test_loss"),
            "test_delta_u": row.get("test_delta_u"),
            "target_k": row.get("target_k"),
            "min_k": row.get("min_k"),
            "original_leak_k": row["original_leak_k"],
            "original_leak_k_p1": row["original_leak_k_p1"],
            "original_leak_k_p2": row["original_leak_k_p2"],
            "original_leak_k_p3": row["original_leak_k_p3"],
            "original_leak_k_p4": row["original_leak_k_p4"],
            "original_leak_k_p5": row["original_leak_k_p5"],
            "leak_k_p1": row.get("leak_k_p1"),
            "leak_k_p2": row.get("leak_k_p2"),
            "leak_k_p3": row.get("leak_k_p3"),
            "leak_k_p4": row.get("leak_k_p4"),
            "leak_k_p5": row.get("leak_k_p5"),
            "tail_risk_p99": row["tail_risk_p99"],
            "tail_risk_population_size": row["tail_risk_population_size"],
            "tail_risk_semantics": row["tail_risk_semantics"],
            "row_count": row.get("row_count"),
            "suppressed_row_count": row.get("suppressed_row_count"),
            "coverage_fraction": row.get("coverage_fraction"),
            "iter_enumeration_time": row.get("iter_enumeration_time"),
            "iter_selection_time": row.get("iter_selection_time"),
            "iter_privacy_time": row.get("iter_privacy_time"),
            "iter_retraining_time": row.get("iter_retraining_time"),
            "test_observation_seconds": row.get("test_observation_seconds"),
        }
        flat_rows.append(flat)
        append_jsonl(_experiment_dir / "trajectory.jsonl", flat)
    write_csv_rows(run_dir / "trajectory.csv", flat_rows)
    return {"trajectory_point_count": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Exp-1 privacy--utility sweeps.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--run-label")
    args = parser.parse_args()

    require_core_parameters("stop_on_utility", "record_iteration_test_metrics")
    output = run_matrix_config(
        args.config,
        expected_kind="exp1_privacy_utility",
        device=args.device,
        run_label=args.run_label,
        after_task=_require_trajectory_artifacts,
    )
    print(output)


if __name__ == "__main__":
    main()
