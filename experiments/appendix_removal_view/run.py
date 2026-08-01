"""Run the declared appendix removal-view matrix."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from prototype.dataset_registry import (
    build_data_loader,
    resolve_generalization_tree,
)
from prototype.model_factory import build_model_factory

from ..common import (
    append_jsonl,
    create_experiment_dir,
    load_yaml_mapping,
    resolve_project_path,
    write_csv_rows,
    write_json,
)
from ..run_matrix_sweep import expand_matrix
from .pipeline import run_removal_view


_ALLOWED_PIPELINE_KEYS = {
    "enable_exchange",
    "exchange_pool_size",
    "greedy_ratio_epsilon",
    "max_iterations",
    "nrows",
    "random_state",
    "rank_top_k",
    "selected_attributes",
    "standardizer",
    "test_size",
    "tolerance",
    "val_size",
}


def run_config(
    config_path: str | Path,
    *,
    device: str | None = None,
    run_label: str | None = None,
) -> Path:
    resolved_config_path, config = load_yaml_mapping(config_path)
    if config.get("experiment_kind") != "appendix_removal_view":
        raise ValueError(
            "Expected experiment_kind='appendix_removal_view', got "
            f"{config.get('experiment_kind')!r}."
        )
    experiment_id = str(config.get("experiment_id", "")).strip()
    if not experiment_id:
        raise ValueError("Experiment config must define experiment_id.")
    if config.get("results_root") is None:
        raise ValueError("Experiment config must define results_root.")

    tasks = expand_matrix(config)
    for task in tasks:
        pipeline_options = dict(task.get("pipeline") or {})
        unknown = sorted(set(pipeline_options).difference(_ALLOWED_PIPELINE_KEYS))
        if unknown:
            raise ValueError(
                f"Task {task['run_id']!r} contains unsupported removal-view "
                f"pipeline keys: {unknown}."
            )
        if task.get("task_type") not in {None, "classification"}:
            raise ValueError("Appendix removal-view tasks are classification-only.")

    experiment_dir = create_experiment_dir(
        config["results_root"],
        experiment_id,
        run_label=run_label,
    )
    write_json(
        experiment_dir / "manifest.json",
        {
            "experiment_id": experiment_id,
            "experiment_kind": "appendix_removal_view",
            "source_config": str(resolved_config_path),
            "device_override": device,
            "resolved_runs": tasks,
        },
    )

    summaries = []
    for task in tasks:
        run_id = task["run_id"]
        dataset = str(task.get("dataset", "")).strip()
        if not dataset:
            raise ValueError(f"Task {run_id!r} must define dataset.")
        data_loader = build_data_loader(
            dataset,
            data_path=resolve_project_path(task.get("data_path")),
        )
        tree_path = resolve_generalization_tree(
            dataset,
            tree_path=resolve_project_path(
                task.get("generalization_tree_path")
            ),
        )
        configured_device = device or task.get("device", "cuda")
        dtype = task.get("dtype", "float32")
        allow_cpu_fallback = bool(task.get("allow_cpu_fallback", True))
        pipeline_options = deepcopy(dict(task.get("pipeline") or {}))
        random_state = int(pipeline_options.get("random_state", 42))
        model_config = deepcopy(dict(task.get("model") or {}))
        factories = {
            role: build_model_factory(
                model_config.get(role),
                device=configured_device,
                dtype=dtype,
                allow_cpu_fallback=allow_cpu_fallback,
                random_state=random_state,
                role=role,
            )
            for role in ("downstream", "proxy", "estimator")
        }

        result = run_removal_view(
            data_loader,
            tree_path,
            results_dir=experiment_dir / "runs" / run_id,
            run_id=run_id,
            model_factory=factories["downstream"],
            proxy_model_factory=factories["proxy"],
            estimator_model_factory=factories["estimator"],
            device=configured_device,
            dtype=dtype,
            allow_cpu_fallback=allow_cpu_fallback,
            **pipeline_options,
        )
        write_json(Path(result.run_dir) / "declared_task.json", task)
        summary = {
            "run_id": run_id,
            "dataset": dataset,
            "model_name": task.get("model_name"),
            "task_type": result.task_type,
            "selection_rule": result.selection_rule,
            "random_state": random_state,
            "tolerance": pipeline_options.get("tolerance"),
            "nrows": pipeline_options.get("nrows"),
            "rank_top_k": pipeline_options.get("rank_top_k"),
            "greedy_ratio_epsilon": result.selection_rule_config.get(
                "epsilon"
            ),
            "exchange_pool_size": pipeline_options.get("exchange_pool_size"),
            "enable_exchange": pipeline_options.get("enable_exchange"),
            "termination_condition": result.termination_condition,
            "iteration_count": result.iteration_count,
            "exchange_count": result.exchange_count,
            "active_row_count": len(result.active_row_ids),
            "removed_row_count": len(result.removed_row_ids),
            "original_leak_k": result.original_leak_k,
            "original_bottleneck_count": result.original_bottleneck_count,
            "final_leak_k": result.final_leak_k,
            "final_bottleneck_count": result.final_bottleneck_count,
            "final_leak_k_p1": result.leak_k_p1,
            "final_leak_k_p5": result.leak_k_p5,
            "baseline_val_loss": result.baseline_val_loss,
            "loss_threshold_val": result.loss_threshold_val,
            "final_actual_val_loss": result.final_actual_val_loss,
            "baseline_test_loss": result.baseline_test_loss,
            "loss_threshold_test": result.loss_threshold_test,
            "final_actual_model_loss": result.final_actual_model_loss,
            "validation_utility_constraint_met": (
                result.validation_utility_constraint_met
            ),
            "utility_constraint_met": result.utility_constraint_met,
            "run_dir": result.run_dir,
            **result.timings,
            **task["axis_values"],
        }
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
    return experiment_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the appendix greedy-ratio removal view."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--run-label")
    args = parser.parse_args()
    output = run_config(
        args.config,
        device=args.device,
        run_label=args.run_label,
    )
    print(output)


if __name__ == "__main__":
    main()
