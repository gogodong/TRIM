"""Configuration-driven command-line entry point for TRIM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .TRIM_prototype_pipeline import run_trim_pipeline
from .dataset_registry import (
    PROJECT_ROOT,
    build_data_loader,
    resolve_generalization_tree,
)
from .model_factory import build_model_factory


def main():
    parser = argparse.ArgumentParser(
        description="Run TRIM from a YAML configuration file."
    )
    parser.add_argument("--config", required=True, help="YAML configuration path.")
    parser.add_argument("--dataset", help="Override the configured dataset.")
    parser.add_argument("--data-path", help="Override the dataset file/directory.")
    parser.add_argument("--tree-path", help="Override the generalization tree.")
    parser.add_argument("--results-dir", help="Override the result directory.")
    parser.add_argument("--device", help="Override the configured Torch device.")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("The TRIM configuration must be a YAML mapping.")

    dataset = args.dataset or config.get("dataset")
    if dataset is None:
        raise ValueError("The configuration must define 'dataset'.")

    data_path_value = args.data_path or config.get("data_path")
    if data_path_value is not None:
        data_path = Path(data_path_value).expanduser()
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path
    else:
        data_path = None
    data_loader = build_data_loader(dataset, data_path=data_path)

    tree_path_value = args.tree_path or config.get("generalization_tree_path")
    if tree_path_value is not None:
        tree_path = Path(tree_path_value).expanduser()
        if not tree_path.is_absolute():
            tree_path = PROJECT_ROOT / tree_path
    else:
        tree_path = None
    generalization_tree_path = resolve_generalization_tree(
        dataset,
        tree_path=tree_path,
    )

    results_dir_value = args.results_dir or config.get("results_dir")
    if results_dir_value is not None:
        results_dir = Path(results_dir_value).expanduser()
        if not results_dir.is_absolute():
            results_dir = PROJECT_ROOT / results_dir
    else:
        results_dir = None

    device = args.device or config.get("device", "cuda")
    dtype = config.get("dtype", "float32")
    allow_cpu_fallback = bool(config.get("allow_cpu_fallback", True))
    pipeline_options = dict(config.get("pipeline") or {})
    model_config = config.get("model")
    if model_config is not None and not isinstance(model_config, dict):
        raise ValueError("'model' must be a mapping of model roles.")

    factories = {}
    for role in ("downstream", "proxy", "estimator"):
        spec = model_config.get(role) if model_config is not None else None
        factories[role] = build_model_factory(
            spec,
            device=device,
            dtype=dtype,
            allow_cpu_fallback=allow_cpu_fallback,
            random_state=int(pipeline_options.get("random_state", 42)),
            role=role,
        )

    result = run_trim_pipeline(
        data_loader,
        generalization_tree_path=generalization_tree_path,
        model_config=model_config,
        model_factory=factories["downstream"],
        proxy_model_factory=factories["proxy"],
        estimator_model_factory=factories["estimator"],
        results_dir=results_dir,
        device=device,
        dtype=dtype,
        **pipeline_options,
    )
    print(
        json.dumps(
            {
                "dataset": str(dataset),
                "termination_condition": result.termination_condition,
                "iteration_count": result.iteration_count,
                "selected_row_count": len(result.selected_row_ids),
                "final_leak_k": result.final_leak_k,
                "baseline_val_loss": result.baseline_val_loss,
                "loss_threshold_val": result.loss_threshold_val,
                "final_actual_model_loss": result.final_actual_model_loss,
                "utility_constraint_met": result.utility_constraint_met,
                "run_dir": result.run_dir,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
