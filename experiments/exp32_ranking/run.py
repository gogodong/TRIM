"""Run the paper's Exp-32 MLP-LGA candidate-ranking comparison."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from experiments.common import (
    create_experiment_dir,
    load_yaml_mapping,
    resolve_project_path,
    write_json,
)
from experiments.ranking_models import CandidateExperiment
from prototype.dataset_registry import build_data_loader, resolve_generalization_tree
from prototype.model_factory import build_model_factory

from .pipeline import run_trim_pipeline


def _mlp_factory(spec, *, device, dtype, allow_cpu_fallback, seed, role):
    if not isinstance(spec, dict) or spec.get("family") != "mlp":
        raise ValueError(f"Exp-32 model.{role}.family must be 'mlp'.")
    # The shared implementation reserves its estimator role for Logistic LGA.
    # Exp-32 owns an MLP-LGA pipeline, so it builds that model
    # through the general MLP factory and labels the resulting callable here.
    factory = build_model_factory(
        spec,
        device=device,
        dtype=dtype,
        allow_cpu_fallback=allow_cpu_fallback,
        random_state=seed,
        role="downstream",
    )
    factory.model_role = role
    return factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", help="Override the configured device.")
    args = parser.parse_args()

    config_path, config = load_yaml_mapping(args.config)
    dataset = config.get("dataset")
    if not dataset:
        raise ValueError("Exp-32 config must define dataset.")
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("Exp-32 config must define a non-empty seeds list.")
    if len({int(seed) for seed in seeds}) != len(seeds):
        raise ValueError("Exp-32 seeds must be unique.")

    results_root = config.get("results_root")
    if not results_root:
        raise ValueError("Exp-32 config must define results_root.")
    experiment_dir = create_experiment_dir(
        results_root,
        "exp32_ranking",
        run_label=config.get("run_label"),
    )
    device = args.device or config.get("device", "cuda")
    dtype = config.get("dtype", "float32")
    allow_cpu_fallback = bool(config.get("allow_cpu_fallback", True))
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("Exp-32 config must define model as a mapping.")
    if model.get("proxy") != model.get("downstream"):
        raise ValueError(
            "The Exp-32 configuration requires identical proxy and downstream "
            "MLP specifications so exact retrain rows use the same model."
        )
    pipeline_options = deepcopy(dict(config.get("pipeline") or {}))
    influence = deepcopy(dict(config.get("yang_if") or {}))
    required_if = {"horizontal_last", "vertical_all"}
    missing_if = sorted(required_if - set(influence))
    if missing_if:
        raise ValueError(f"Exp-32 yang_if is missing {missing_if}.")
    oracle_limit = config.get("oracle_candidate_limit_per_type")
    if oracle_limit is None:
        raise ValueError("Exp-32 requires oracle_candidate_limit_per_type.")

    data_path = resolve_project_path(config.get("data_path"))
    tree_value = resolve_project_path(config.get("generalization_tree_path"))
    summaries = []
    for raw_seed in seeds:
        seed = int(raw_seed)
        task_options = deepcopy(pipeline_options)
        task_options["random_state"] = seed
        loader = build_data_loader(dataset, data_path=data_path)
        tree_path = resolve_generalization_tree(dataset, tree_path=tree_value)
        factories = {
            role: _mlp_factory(
                model.get(role),
                device=device,
                dtype=dtype,
                allow_cpu_fallback=allow_cpu_fallback,
                seed=seed,
                role=role,
            )
            for role in ("downstream", "proxy", "estimator")
        }
        output_csv = experiment_dir / f"seed_{seed}_candidates.csv"
        runtime_output_csv = experiment_dir / f"seed_{seed}_runtime.csv"
        candidate_experiment = CandidateExperiment(
            mode="ranking",
            seed=seed,
            output_csv=output_csv,
            runtime_output_csv=runtime_output_csv,
            oracle_candidate_limit_per_type=int(oracle_limit),
            horizontal_last=influence["horizontal_last"],
            vertical_all=influence["vertical_all"],
        )
        result = run_trim_pipeline(
            loader,
            generalization_tree_path=tree_path,
            model_config=model,
            model_factory=factories["downstream"],
            proxy_model_factory=factories["proxy"],
            estimator_model_factory=factories["estimator"],
            results_dir=experiment_dir / "runs" / f"seed_{seed}",
            run_tag=f"exp32_seed_{seed}",
            device=device,
            dtype=dtype,
            candidate_experiment=candidate_experiment,
            **task_options,
        )
        candidate_experiment.write()
        summaries.append({
            "seed": seed,
            "candidate_csv": str(output_csv),
            "runtime_csv": str(runtime_output_csv),
            "run_dir": result.run_dir,
            "iteration_count": result.iteration_count,
            "termination_condition": result.termination_condition,
        })

    write_json(experiment_dir / "manifest.json", {
        "experiment": "exp32_ranking",
        "config": str(config_path),
        "dataset": dataset,
        "device": device,
        "runs": summaries,
    })
    print(experiment_dir)


if __name__ == "__main__":
    main()
