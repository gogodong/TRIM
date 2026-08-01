"""Small orchestration helpers shared by the paper-numbered entry points."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Mapping

from prototype.TRIM_prototype_pipeline import run_trim_pipeline

from .common import (
    append_jsonl,
    create_experiment_dir,
    load_yaml_mapping,
    run_core_task,
    write_csv_rows,
    write_json,
)
from .run_matrix_sweep import expand_matrix


AfterTask = Callable[
    [Mapping[str, Any], Mapping[str, Any], Path],
    Mapping[str, Any] | None,
]


def require_core_parameters(*parameter_names: str) -> None:
    """Fail before launching a sweep when its observation hooks are absent."""
    available = set(inspect.signature(run_trim_pipeline).parameters)
    missing = [name for name in parameter_names if name not in available]
    if missing:
        raise RuntimeError(
            "This experiment requires TRIM observation controls that are "
            f"not available in the current implementation: {missing}. "
            "See the Experiments section in the project README."
        )


def run_matrix_config(
    config_path: str | Path,
    *,
    expected_kind: str,
    device: str | None = None,
    run_label: str | None = None,
    after_task: AfterTask | None = None,
) -> Path:
    """Expand one matrix YAML and execute every task through ``run_core_task``."""
    resolved_config_path, config = load_yaml_mapping(config_path)
    experiment_kind = str(config.get("experiment_kind", "")).strip()
    if experiment_kind != expected_kind:
        raise ValueError(
            f"Expected experiment_kind={expected_kind!r}, got {experiment_kind!r}."
        )
    experiment_id = str(config.get("experiment_id", "")).strip()
    if not experiment_id:
        raise ValueError("Experiment config must define experiment_id.")
    results_root = config.get("results_root")
    if results_root is None:
        raise ValueError("Experiment config must define results_root.")

    tasks = expand_matrix(config)
    experiment_dir = create_experiment_dir(
        results_root,
        experiment_id,
        run_label=run_label,
    )
    write_json(
        experiment_dir / "manifest.json",
        {
            "experiment_id": experiment_id,
            "experiment_kind": experiment_kind,
            "source_config": str(resolved_config_path),
            "device_override": device,
            "resolved_runs": tasks,
        },
    )

    summaries = []
    for task in tasks:
        summary = run_core_task(task, experiment_dir=experiment_dir, device=device)
        summary.update(task["axis_values"])
        if after_task is not None:
            patch = after_task(task, summary, experiment_dir)
            if patch:
                summary.update(dict(patch))
        summaries.append(summary)
        append_jsonl(experiment_dir / "run_summaries.jsonl", summary)

    write_csv_rows(experiment_dir / "run_summaries.csv", summaries)
    write_json(
        experiment_dir / "completed.json",
        {
            "experiment_id": experiment_id,
            "experiment_kind": experiment_kind,
            "run_count": len(summaries),
            "run_ids": [summary["run_id"] for summary in summaries],
        },
    )
    return experiment_dir
