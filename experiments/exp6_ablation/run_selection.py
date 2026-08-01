"""Run the paper's Exp-6b2 selection-rule ablation for TRIM."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math

from prototype.model_factory import build_model_factory

from .._runner import require_core_parameters
from ..common import (
    append_jsonl,
    create_experiment_dir,
    load_yaml_mapping,
    run_core_task,
    write_csv_rows,
    write_json,
)
from ..run_matrix_sweep import expand_matrix
from .policies import (
    SeededUniformShortlist,
    YangIFSelectionScorerFactory,
    privacy_only_score,
    utility_only_score,
    yang_if_utility_privacy_score,
)
from .results import iteration_rows


METHODS = ("TRIM", "Privacy", "Utility", "Random", "IF")


def _finite(value, *, name, minimum=0.0):
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be numeric.")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}.")
    return parsed


def _solver(config, name):
    values = config.get(name)
    if not isinstance(values, dict):
        raise ValueError(f"yang_if.{name} must be a mapping.")
    required = {"damping", "cg_max_iter", "cg_tolerance"}
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"yang_if.{name} is missing {missing}.")
    damping = _finite(values["damping"], name=f"{name}.damping")
    cg_max_iter = int(values["cg_max_iter"])
    if isinstance(values["cg_max_iter"], bool) or cg_max_iter < 1:
        raise ValueError(f"{name}.cg_max_iter must be a positive integer.")
    if float(cg_max_iter) != float(values["cg_max_iter"]):
        raise ValueError(f"{name}.cg_max_iter must be an integer.")
    cg_tolerance = _finite(
        values["cg_tolerance"], name=f"{name}.cg_tolerance"
    )
    return {
        "damping": damping,
        "cg_max_iter": cg_max_iter,
        "cg_tolerance": cg_tolerance,
    }


def _prepare(config_path, config, *, tolerance, yang_if, device_override=None):
    tasks = expand_matrix(config)
    if not tasks:
        raise ValueError(f"Exp-6b2 configuration produced no tasks: {config_path}")
    tolerance = _finite(tolerance, name="tolerance")
    if_model = yang_if.get("model")
    if not isinstance(if_model, dict) or if_model.get("family") != "mlp":
        raise ValueError(
            "yang_if.model must explicitly configure family: mlp."
        )
    horizontal_last = _solver(yang_if, "horizontal_last")
    vertical_all = _solver(yang_if, "vertical_all")

    seen = set()
    shared = None
    prepared = []
    for raw_task in tasks:
        task = deepcopy(raw_task)
        method = task.get("method")
        if method not in METHODS:
            raise ValueError(f"Unknown Exp-6b2 method {method!r}.")
        if "tolerance" in task:
            raise ValueError(
                "Tolerance must not be a task field; supply --tolerance."
            )
        pipeline = task.get("pipeline")
        if not isinstance(pipeline, dict):
            raise ValueError(f"Task {task['run_id']!r} pipeline must be a mapping.")
        if "tolerance" in pipeline:
            raise ValueError(
                "Tolerance must be absent from YAML and supplied by --tolerance."
            )
        forbidden = {
            "fixed_row_candidate_builder",
            "candidate_shortlist",
            "candidate_state_scorer_factory",
            "selection_score",
        }.intersection(pipeline)
        if forbidden:
            raise ValueError(f"Exp-6b2 configuration cannot override hooks: {forbidden}.")
        pipeline["tolerance"] = tolerance
        if pipeline.get("record_iteration_test_metrics") is not True:
            raise ValueError(
                "Exp-6b2 requires pipeline.record_iteration_test_metrics=true."
            )
        if str(task.get("dataset", "")).lower() != "pubcov":
            raise ValueError("The Exp-6b2 configuration must use PubCov.")
        s0 = task.get("initial_sample_fraction")
        if s0 is None or not 0.0 < float(s0) <= 1.0:
            raise ValueError("Every Exp-6b2 method must use one valid shared S0.")
        seed = int(pipeline.get("random_state"))
        combination = (method, seed)
        if combination in seen:
            raise ValueError(f"Duplicate Exp-6b2 method/seed: {combination}.")
        seen.add(combination)

        if method == "Privacy":
            pipeline["selection_score"] = privacy_only_score
        elif method == "Utility":
            pipeline["selection_score"] = utility_only_score
        elif method == "Random":
            pipeline["candidate_shortlist"] = SeededUniformShortlist(seed)
        elif method == "IF":
            scorer_model_factory = build_model_factory(
                if_model,
                device=device_override or task.get("device", "cuda"),
                dtype=task.get("dtype", "float32"),
                allow_cpu_fallback=bool(task.get("allow_cpu_fallback", True)),
                random_state=seed,
                role="downstream",
            )
            pipeline["candidate_state_scorer_factory"] = (
                YangIFSelectionScorerFactory(
                    model_factory=scorer_model_factory,
                    horizontal_last=horizontal_last,
                    vertical_all=vertical_all,
                )
            )
            pipeline["selection_score"] = yang_if_utility_privacy_score

        comparison_task = deepcopy(task)
        comparison_task.pop("run_id", None)
        comparison_task.pop("axis_values", None)
        comparison_task.pop("method", None)
        comparison_pipeline = comparison_task["pipeline"]
        comparison_pipeline.pop("random_state", None)
        comparison_pipeline.pop("tolerance", None)
        comparison_pipeline.pop("candidate_shortlist", None)
        comparison_pipeline.pop("candidate_state_scorer_factory", None)
        comparison_pipeline.pop("selection_score", None)
        current_shared = json.dumps(comparison_task, sort_keys=True)
        if shared is None:
            shared = current_shared
        elif shared != current_shared:
            raise ValueError(
                "Apart from method, seed, and declared selection hooks, every "
                "Exp-6b2 task setting must agree."
            )
        prepared.append(task)

    seeds = {seed for _method, seed in seen}
    expected = {(method, seed) for method in METHODS for seed in seeds}
    if seen != expected:
        raise ValueError("Exp-6b2 requires the complete method x seed matrix.")
    return prepared, horizontal_last, vertical_all


def _manifest_task(task):
    output = deepcopy(task)
    pipeline = output["pipeline"]
    for name in (
        "candidate_shortlist",
        "candidate_state_scorer_factory",
        "selection_score",
    ):
        value = pipeline.get(name)
        if value is not None:
            pipeline[name] = (
                f"{value.__class__.__module__}.{value.__class__.__qualname__}"
                if not hasattr(value, "__qualname__")
                else f"{value.__module__}.{value.__qualname__}"
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tolerance", required=True, type=float)
    parser.add_argument("--device")
    parser.add_argument("--run-label")
    args = parser.parse_args()

    config_path, config = load_yaml_mapping(args.config)
    if config.get("experiment_kind") != "exp6b2_selection":
        raise ValueError("Expected experiment_kind='exp6b2_selection'.")
    if "tolerance" in config:
        raise ValueError("Tolerance must be supplied only by --tolerance.")
    experiment_id = str(config.get("experiment_id", "")).strip()
    if not experiment_id or config.get("results_root") is None:
        raise ValueError("Exp-6b2 requires experiment_id and results_root.")
    yang_if = config.get("yang_if")
    if not isinstance(yang_if, dict):
        raise ValueError("Exp-6b2 configuration must define yang_if.")
    require_core_parameters(
        "candidate_shortlist",
        "candidate_state_scorer_factory",
        "selection_score",
        "record_iteration_test_metrics",
    )
    tasks, horizontal_last, vertical_all = _prepare(
        config_path,
        config,
        tolerance=args.tolerance,
        yang_if=yang_if,
        device_override=args.device,
    )
    experiment_dir = create_experiment_dir(
        config["results_root"], experiment_id, run_label=args.run_label
    )
    write_json(experiment_dir / "manifest.json", {
        "experiment_id": experiment_id,
        "experiment_kind": "exp6b2_selection",
        "source_config": str(config_path),
        "device_override": args.device,
        "reader_supplied_tolerance": float(args.tolerance),
        "selection_rule_definitions": {
            "TRIM": "release_validation_utility_gain / privacy_cost",
            "Privacy": "1 / privacy_cost",
            "Utility": "release_validation_utility_gain",
            "Random": "seeded_uniform_one_from_all_H_and_V_actions",
            "IF": "yang_fixed_original_validation_utility_delta / privacy_cost",
        },
        "yang_if": {
            "model": yang_if["model"],
            "horizontal_last": horizontal_last,
            "vertical_all": vertical_all,
        },
        "resolved_runs": [_manifest_task(task) for task in tasks],
    })

    summaries = []
    raw_rows = []
    for task in tasks:
        summary = run_core_task(task, experiment_dir=experiment_dir, device=args.device)
        method = task["method"]
        summary.update(task["axis_values"])
        summary["method"] = method
        summaries.append(summary)
        append_jsonl(experiment_dir / "run_summaries.jsonl", summary)
        raw_rows.extend(iteration_rows(
            task=task,
            summary=summary,
            experiment_id=experiment_id,
            experiment_kind="exp6b2_selection",
            method=method,
        ))
    write_csv_rows(experiment_dir / "run_summaries.csv", summaries)
    write_csv_rows(experiment_dir / "iterations.csv", raw_rows)
    for row in raw_rows:
        append_jsonl(experiment_dir / "iterations.jsonl", row)
    write_json(experiment_dir / "completed.json", {
        "experiment_id": experiment_id,
        "experiment_kind": "exp6b2_selection",
        "run_count": len(summaries),
        "iteration_row_count": len(raw_rows),
        "run_ids": [summary["run_id"] for summary in summaries],
    })
    print(experiment_dir)


if __name__ == "__main__":
    main()
