"""Expand named YAML axes and run their Cartesian product through TRIM."""

from __future__ import annotations

import argparse
from itertools import product
import re

from .common import (
    append_jsonl,
    create_experiment_dir,
    load_yaml_mapping,
    merge_mappings,
    run_core_task,
    write_csv_rows,
    write_json,
)


def _safe_token(value) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    if not token:
        raise ValueError(f"Axis label cannot be converted to a run-id token: {value!r}")
    return token


def expand_matrix(config: dict) -> list[dict]:
    base_task = config.get("base_task") or {}
    if not isinstance(base_task, dict):
        raise ValueError("base_task must be a YAML mapping.")
    axes = config.get("axes")
    if not isinstance(axes, list) or not axes:
        raise ValueError("Matrix config must contain a non-empty axes list.")

    normalized_axes = []
    for axis in axes:
        if not isinstance(axis, dict):
            raise ValueError("Every axis must be a YAML mapping.")
        name = _safe_token(axis.get("name", ""))
        values = axis.get("values")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Axis {name!r} must have a non-empty values list.")
        normalized_values = []
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(
                    f"Every value in axis {name!r} must contain label and patch."
                )
            if "label" not in value or "patch" not in value:
                raise ValueError(
                    f"Every value in axis {name!r} must contain label and patch."
                )
            if not isinstance(value["patch"], dict):
                raise ValueError(f"Axis {name!r} value patch must be a mapping.")
            normalized_values.append(
                {
                    "label": _safe_token(value["label"]),
                    "patch": value["patch"],
                }
            )
        normalized_axes.append((name, normalized_values))

    run_prefix = _safe_token(config.get("run_prefix") or config.get("experiment_id"))
    tasks = []
    for combination in product(*(values for _name, values in normalized_axes)):
        task = merge_mappings({}, base_task)
        run_parts = [run_prefix]
        axis_values = {}
        for (axis_name, _values), selected in zip(normalized_axes, combination):
            task = merge_mappings(task, selected["patch"])
            run_parts.extend((axis_name, selected["label"]))
            axis_values[axis_name] = selected["label"]
        task["run_id"] = "_".join(run_parts)
        task["axis_values"] = axis_values
        tasks.append(task)

    run_ids = [task["run_id"] for task in tasks]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Matrix expansion produced duplicate run IDs.")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a declared experiment matrix.")
    parser.add_argument("--config", required=True, help="Matrix YAML path.")
    parser.add_argument("--device", help="Override every task's device.")
    parser.add_argument("--run-label", help="Optional experiment-directory label.")
    args = parser.parse_args()

    config_path, config = load_yaml_mapping(args.config)
    experiment_id = str(config.get("experiment_id", "")).strip()
    if not experiment_id:
        raise ValueError("Matrix config must define experiment_id.")
    if config.get("results_root") is None:
        raise ValueError("Matrix config must define results_root.")
    tasks = expand_matrix(config)
    experiment_dir = create_experiment_dir(
        config["results_root"],
        experiment_id,
        run_label=args.run_label,
    )
    write_json(
        experiment_dir / "manifest.json",
        {
            "experiment_id": experiment_id,
            "source_config": str(config_path),
            "device_override": args.device,
            "resolved_runs": tasks,
        },
    )

    summaries = []
    for task in tasks:
        summary = run_core_task(
            task,
            experiment_dir=experiment_dir,
            device=args.device,
        )
        summary.update(task["axis_values"])
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
