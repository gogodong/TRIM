"""Run an explicit list of TRIM tasks from one YAML file."""

from __future__ import annotations

import argparse

from .common import (
    append_jsonl,
    create_experiment_dir,
    load_yaml_mapping,
    merge_mappings,
    run_core_task,
    write_csv_rows,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run explicitly declared TRIM experiment tasks."
    )
    parser.add_argument("--config", required=True, help="Experiment YAML path.")
    parser.add_argument("--device", help="Override every task's configured device.")
    parser.add_argument("--run-label", help="Optional label for the experiment directory.")
    args = parser.parse_args()

    config_path, config = load_yaml_mapping(args.config)
    experiment_id = str(config.get("experiment_id", "")).strip()
    if not experiment_id:
        raise ValueError("Experiment config must define experiment_id.")
    results_root = config.get("results_root")
    if results_root is None:
        raise ValueError("Experiment config must define results_root.")
    declared_runs = config.get("runs")
    if not isinstance(declared_runs, list) or not declared_runs:
        raise ValueError("Experiment config must contain a non-empty runs list.")

    defaults = config.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a YAML mapping.")
    resolved_runs = []
    seen_run_ids = set()
    for declared_run in declared_runs:
        if not isinstance(declared_run, dict):
            raise ValueError("Every runs entry must be a YAML mapping.")
        task = merge_mappings(defaults, declared_run)
        run_id = str(task.get("run_id", "")).strip()
        if not run_id:
            raise ValueError("Every runs entry must define run_id.")
        if run_id in seen_run_ids:
            raise ValueError(f"Duplicate run_id in experiment config: {run_id!r}")
        seen_run_ids.add(run_id)
        resolved_runs.append(task)

    experiment_dir = create_experiment_dir(
        results_root,
        experiment_id,
        run_label=args.run_label,
    )
    write_json(
        experiment_dir / "manifest.json",
        {
            "experiment_id": experiment_id,
            "source_config": str(config_path),
            "device_override": args.device,
            "resolved_runs": resolved_runs,
        },
    )

    summaries = []
    for task in resolved_runs:
        summary = run_core_task(
            task,
            experiment_dir=experiment_dir,
            device=args.device,
        )
        summaries.append(summary)
        append_jsonl(experiment_dir / "run_summaries.jsonl", summary)

    write_csv_rows(experiment_dir / "run_summaries.csv", summaries)
    write_json(
        experiment_dir / "completed.json",
        {
            "experiment_id": experiment_id,
            "run_count": len(summaries),
            "run_ids": [summary["run_id"] for summary in summaries],
        },
    )
    print(experiment_dir)


if __name__ == "__main__":
    main()
