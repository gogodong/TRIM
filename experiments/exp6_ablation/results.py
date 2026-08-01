"""Export per-iteration TRIM observations for Exp-6."""

from __future__ import annotations

import json
import math
from pathlib import Path


_REQUIRED_ITERATION_FIELDS = {
    "iteration",
    "action",
    "original_leak_k",
    "original_leak_k_p1",
    "original_leak_k_p5",
    "min_k",
    "leak_k_p1",
    "leak_k_p5",
    "tail_risk_p99",
    "test_status",
    "test_loss",
    "test_delta_u",
}


def _finite_number(value, *, field, path, iteration):
    if isinstance(value, bool):
        raise ValueError(f"Invalid {field} in {path} iteration {iteration}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {field} in {path} iteration {iteration}: {value!r}."
        ) from exc
    if not math.isfinite(number):
        raise ValueError(
            f"Non-finite {field} in {path} iteration {iteration}: {value!r}."
        )
    return number


def iteration_rows(*, task, summary, experiment_id, experiment_kind, method):
    """Read one TRIM run's chronological observations."""

    run_dir = Path(summary["run_dir"])
    path = run_dir / "trim_iterations.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing TRIM iteration output: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"iterations"}:
        raise ValueError(f"Unexpected TRIM iteration schema: {path}")
    records = payload["iterations"]
    if not isinstance(records, list) or not records:
        raise ValueError(f"Exp-6 run produced no plottable iterations: {path}")

    rows = []
    observed_iterations = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Non-object TRIM iteration in {path}.")
        missing = sorted(_REQUIRED_ITERATION_FIELDS - set(record))
        if missing:
            raise ValueError(f"Missing TRIM iteration fields in {path}: {missing}.")
        iteration = int(record["iteration"])
        observed_iterations.append(iteration)
        if record["test_status"] != "ok":
            raise ValueError(
                f"Iteration {iteration} in {path} has test_status="
                f"{record['test_status']!r}; enable record_iteration_test_metrics."
            )
        numeric = {
            field: _finite_number(
                record[field],
                field=field,
                path=path,
                iteration=iteration,
            )
            for field in (
                "original_leak_k",
                "original_leak_k_p1",
                "original_leak_k_p5",
                "min_k",
                "leak_k_p1",
                "leak_k_p5",
                "tail_risk_p99",
                "test_loss",
                "test_delta_u",
            )
        }
        for field in (
            "original_leak_k",
            "original_leak_k_p1",
            "original_leak_k_p5",
            "min_k",
            "leak_k_p1",
            "leak_k_p5",
        ):
            if numeric[field] <= 0.0:
                raise ValueError(
                    f"Exp-6 privacy field {field} must be positive in "
                    f"{path} iteration {iteration}."
                )
        rows.append({
            "experiment_id": experiment_id,
            "experiment_kind": experiment_kind,
            "method": method,
            "run_id": summary["run_id"],
            "dataset": summary["dataset"],
            "model_name": summary["model_name"],
            "random_state": summary["random_state"],
            "tolerance": summary["tolerance"],
            "iteration": iteration,
            "action": record["action"],
            "attribute": record.get("attribute"),
            "baseline_test_loss": summary["baseline_test_loss"],
            **numeric,
            "release_target_k": record.get("target_k"),
            "release_row_count": record.get("row_count"),
            "release_suppressed_row_count": record.get(
                "suppressed_row_count"
            ),
            "release_coverage_fraction": record.get("coverage_fraction"),
            "tail_risk_population_size": record.get(
                "tail_risk_population_size"
            ),
            "tail_risk_semantics": record.get("tail_risk_semantics"),
            "source_trim_iterations": str(path),
        })
    if observed_iterations != list(range(len(observed_iterations))):
        raise ValueError(
            f"TRIM iterations must be chronological 0..N-1 in {path}; "
            f"found {observed_iterations}."
        )
    return rows


__all__ = ["iteration_rows"]
