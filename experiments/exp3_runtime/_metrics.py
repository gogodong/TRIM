"""Extract the runtime summary from one completed TRIM run."""

from __future__ import annotations

import json
from pathlib import Path

from ..common import write_json


ITERATION_TIME_FIELDS = (
    "iter_enumeration_time",
    "iter_selection_time",
    "iter_privacy_time",
    "iter_retraining_time",
)


def extract_runtime_to_first_feasible(task, summary, _experiment_dir):
    """Return algorithm time through the first test-feasible TRIM release."""
    report_tolerance = task.get("report_tolerance")
    if report_tolerance is None:
        raise ValueError(f"Task {summary['run_id']!r} is missing report_tolerance.")
    report_tolerance = float(report_tolerance)
    run_dir = Path(summary["run_dir"])
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    iteration_payload = json.loads(
        (run_dir / "trim_iterations.json").read_text(encoding="utf-8")
    )
    iteration_rows = iteration_payload.get("iterations")
    if not isinstance(iteration_rows, list) or not iteration_rows:
        raise ValueError(f"No TRIM iterations were recorded for {summary['run_id']!r}.")

    baseline_test_loss = metrics.get("baseline_test_loss")
    if baseline_test_loss is None:
        raise ValueError("metrics.json is missing baseline_test_loss.")
    threshold = float(baseline_test_loss) + report_tolerance
    feasible = []
    for row in iteration_rows:
        if row.get("test_loss") is None:
            raise ValueError(
                "trim_iterations.json is missing per-iteration test_loss, "
                "which is required to select the runtime point."
            )
        if float(row["test_loss"]) <= threshold:
            feasible.append(row)
    if not feasible:
        raise ValueError(
            f"No TRIM iteration meets test loss <= {threshold:.12g} for "
            f"{summary['run_id']!r}."
        )
    selected = min(feasible, key=lambda row: int(row["iteration"]))
    selected_iteration = int(selected["iteration"])

    action_logs = sorted(run_dir.glob("pipeline_selection_actions_*.jsonl"))
    if len(action_logs) != 1:
        raise ValueError(
            f"Expected one action log in {run_dir}; found {len(action_logs)}."
        )
    states = {}
    with action_logs[0].open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL record at {action_logs[0]}:{line_number}."
                ) from exc
            if record.get("event") == "pipeline_iteration_state":
                states[int(record["iteration"])] = record
    expected = set(range(selected_iteration + 1))
    if set(states).intersection(expected) != expected:
        raise ValueError(
            "Action log does not contain every iteration through the selected "
            f"runtime point: expected {sorted(expected)}, got {sorted(states)}."
        )

    cumulative = 0.0
    component_totals = {field: 0.0 for field in ITERATION_TIME_FIELDS}
    for iteration in range(selected_iteration + 1):
        state = states[iteration]
        for field in ITERATION_TIME_FIELDS:
            if state.get(field) is None:
                raise ValueError(
                    f"Action log iteration {iteration} is missing {field!r}. "
                    "The runtime summary requires this field."
                )
            value = float(state[field])
            if value < 0:
                raise ValueError(f"Negative timing field {field!r}: {value}.")
            component_totals[field] += value
            cumulative += value

    timings = metrics.get("timings") or {}
    initial_build = timings.get("initial_retention_class_build_time_seconds")
    candidate_total = timings.get("candidate_enumeration_total_time_seconds")
    if initial_build is None or candidate_total is None:
        raise ValueError(
            "TRIM timings must contain initial_retention_class_build_time_seconds "
            "and candidate_enumeration_total_time_seconds."
        )
    initial_build = float(initial_build)
    if initial_build < 0:
        raise ValueError("Initial retention-class build time cannot be negative.")
    cumulative += initial_build

    runtime = {
        "schema_version": "trim_paper_runtime.v1",
        "run_id": summary["run_id"],
        "run_dir": str(run_dir),
        "report_tolerance": report_tolerance,
        "baseline_test_loss": float(baseline_test_loss),
        "test_loss_threshold": threshold,
        "selected_iteration": selected_iteration,
        "selected_test_loss": float(selected["test_loss"]),
        "algorithm_time_seconds": cumulative,
        "initial_retention_class_build_time_seconds": initial_build,
        **component_totals,
    }
    write_json(run_dir / "paper_runtime.json", runtime)
    return {
        "report_tolerance": report_tolerance,
        "runtime_selected_iteration": selected_iteration,
        "runtime_selected_test_loss": float(selected["test_loss"]),
        "paper_algorithm_time_seconds": cumulative,
    }
