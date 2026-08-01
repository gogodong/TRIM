"""Run the paper's Figure 5h candidate-enumeration ablation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math

from prototype import AblationActions, NO_VERTICAL_POSTPROCESSING
from prototype.dataset_registry import resolve_generalization_tree

from .._runner import require_core_parameters
from ..common import (
    append_jsonl,
    create_experiment_dir,
    load_yaml_mapping,
    resolve_project_path,
    run_core_task,
    write_csv_rows,
    write_json,
)
from ..run_matrix_sweep import expand_matrix
from .enumeration import (
    InformationLossBuilder,
    KMeansClusterRingBuilder,
    RandomSplitBuilder,
    SampleLevelBuilder,
)
from .results import iteration_rows


METHODS = ("TRIM", "Sample", "KMeans", "IL", "RandomSplit")


def _integer(value, *, name, minimum=1):
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be an integer.")
    parsed = int(value)
    if float(parsed) != float(value) or parsed < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return parsed


def _mapping(value, *, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping.")
    return value


def _builders(config):
    settings = _mapping(config.get("enumeration"), name="enumeration")
    initial_generalization = str(
        settings.get("initial_generalization", "max")
    )
    kmeans = _mapping(settings.get("kmeans"), name="enumeration.kmeans")
    random_split = _mapping(
        settings.get("random_split"), name="enumeration.random_split"
    )
    information_loss = _mapping(
        settings.get("information_loss"),
        name="enumeration.information_loss",
    )

    base_task = _mapping(config.get("base_task"), name="base_task")
    dataset = str(base_task.get("dataset", "")).strip()
    tree_path = resolve_generalization_tree(
        dataset,
        tree_path=resolve_project_path(
            base_task.get("generalization_tree_path")
        ),
    )
    _, tree_config = load_yaml_mapping(tree_path)
    trees = tree_config.get("trees")
    if not isinstance(trees, dict) or not trees:
        raise ValueError(f"Generalization tree has no non-empty trees map: {tree_path}")

    partner_sample_size = information_loss.get("merge_partner_sample_size")
    if partner_sample_size is not None:
        partner_sample_size = _integer(
            partner_sample_size,
            name="enumeration.information_loss.merge_partner_sample_size",
        )
    return {
        "Sample": SampleLevelBuilder(
            initial_generalization=initial_generalization
        ),
        "KMeans": KMeansClusterRingBuilder(
            n_clusters=_integer(
                kmeans.get("n_clusters"),
                name="enumeration.kmeans.n_clusters",
            ),
            n_rings=_integer(
                kmeans.get("n_rings"),
                name="enumeration.kmeans.n_rings",
            ),
            random_state=_integer(
                kmeans.get("random_state"),
                name="enumeration.kmeans.random_state",
                minimum=0,
            ),
            initial_generalization=initial_generalization,
        ),
        "IL": InformationLossBuilder(
            trees=trees,
            k_ur=_integer(
                information_loss.get("k_ur"),
                name="enumeration.information_loss.k_ur",
            ),
            merge_partner_sample_size=partner_sample_size,
            merge_strategy=str(information_loss.get("merge_strategy", "")),
            random_state=_integer(
                information_loss.get("random_state"),
                name="enumeration.information_loss.random_state",
                minimum=0,
            ),
            initial_generalization=initial_generalization,
        ),
        "RandomSplit": RandomSplitBuilder(
            n_splits=_integer(
                random_split.get("n_splits"),
                name="enumeration.random_split.n_splits",
            ),
            random_state=_integer(
                random_split.get("random_state"),
                name="enumeration.random_split.random_state",
                minimum=0,
            ),
            initial_generalization=initial_generalization,
        ),
    }


def _prepare(config_path, config, *, builders):
    tasks = expand_matrix(config)
    if not tasks:
        raise ValueError(
            f"Figure 5h configuration produced no tasks: {config_path}"
        )
    fixed_actions = AblationActions(
        vertical_postprocessing=NO_VERTICAL_POSTPROCESSING
    )
    seen = set()
    shared = None
    prepared = []
    for raw_task in tasks:
        task = deepcopy(raw_task)
        method = task.get("method")
        if method not in METHODS:
            raise ValueError(f"Unknown Figure 5h method {method!r}.")
        if str(task.get("dataset", "")).lower() != "pubcov":
            raise ValueError("Figure 5h must use PubCov.")
        pipeline = task.get("pipeline")
        if not isinstance(pipeline, dict):
            raise ValueError(f"Task {task['run_id']!r} pipeline must be a mapping.")
        forbidden = {
            "ablation_actions",
            "fixed_row_candidate_builder",
        }.intersection(pipeline)
        if forbidden:
            raise ValueError(
                f"Figure 5h configuration cannot override hooks: {forbidden}."
            )
        tolerance = pipeline.get("tolerance")
        if (
            isinstance(tolerance, bool)
            or tolerance is None
            or not math.isfinite(float(tolerance))
            or float(tolerance) < 0.0
        ):
            raise ValueError("Figure 5h pipeline.tolerance must be non-negative.")
        if pipeline.get("record_iteration_test_metrics") is not True:
            raise ValueError(
                "Figure 5h requires record_iteration_test_metrics=true."
            )
        seed = int(pipeline.get("random_state"))
        combination = (method, seed)
        if combination in seen:
            raise ValueError(f"Duplicate Figure 5h method/seed: {combination}.")
        seen.add(combination)

        initial_fraction = task.get("initial_sample_fraction")
        if method == "TRIM":
            if initial_fraction is None or not 0.0 < float(initial_fraction) <= 1.0:
                raise ValueError("TRIM must declare a valid D0 fraction.")
            pipeline["ablation_actions"] = AblationActions()
        else:
            if initial_fraction is not None:
                raise ValueError(
                    f"{method} uses its largest published group, not D0."
                )
            pipeline["ablation_actions"] = fixed_actions
            pipeline["fixed_row_candidate_builder"] = builders[method]

        comparison = deepcopy(task)
        comparison.pop("run_id", None)
        comparison.pop("axis_values", None)
        comparison.pop("method", None)
        comparison.pop("initial_sample_fraction", None)
        comparison_pipeline = comparison["pipeline"]
        comparison_pipeline.pop("random_state", None)
        comparison_pipeline.pop("ablation_actions", None)
        comparison_pipeline.pop("fixed_row_candidate_builder", None)
        current_shared = json.dumps(comparison, sort_keys=True)
        if shared is None:
            shared = current_shared
        elif current_shared != shared:
            raise ValueError(
                "Apart from method, seed, and the declared enumeration hooks, "
                "every Figure 5h task setting must agree."
            )
        prepared.append(task)

    seeds = {seed for _method, seed in seen}
    expected = {(method, seed) for method in METHODS for seed in seeds}
    if seen != expected:
        raise ValueError("Figure 5h requires the complete method x seed matrix.")
    return prepared


def _manifest_task(task):
    output = deepcopy(task)
    pipeline = output["pipeline"]
    pipeline["ablation_actions"] = pipeline["ablation_actions"].as_dict()
    builder = pipeline.get("fixed_row_candidate_builder")
    if builder is not None:
        pipeline["fixed_row_candidate_builder"] = builder.as_dict()
    return output


def _algorithm_time(summary):
    return sum(
        float(summary.get(field) or 0.0)
        for field in (
            "candidate_enumeration_total_time_seconds",
            "selection_time_seconds",
            "privacy_accounting_time_seconds",
            "backend_retraining_certification_time_seconds",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--run-label")
    args = parser.parse_args()

    config_path, config = load_yaml_mapping(args.config)
    if config.get("experiment_kind") != "exp6a1_enumeration":
        raise ValueError("Expected experiment_kind='exp6a1_enumeration'.")
    experiment_id = str(config.get("experiment_id", "")).strip()
    if not experiment_id or config.get("results_root") is None:
        raise ValueError("Figure 5h requires experiment_id and results_root.")
    builders = _builders(config)
    require_core_parameters(
        "ablation_actions",
        "fixed_row_candidate_builder",
        "record_iteration_test_metrics",
    )
    tasks = _prepare(config_path, config, builders=builders)
    experiment_dir = create_experiment_dir(
        config["results_root"], experiment_id, run_label=args.run_label
    )
    write_json(experiment_dir / "manifest.json", {
        "experiment_id": experiment_id,
        "experiment_kind": "exp6a1_enumeration",
        "figure": "5h",
        "source_config": str(config_path),
        "device_override": args.device,
        "enumeration_definitions": {
            "TRIM": {
                "type": "dynamic_retention_classes",
                "initialization": "D0",
            },
            **{method: builder.as_dict() for method, builder in builders.items()},
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
        summary["experiment_kind"] = "exp6a1_enumeration"
        summary["algorithm_time_seconds"] = _algorithm_time(summary)
        summary["enumeration_definition"] = (
            {
                "type": "dynamic_retention_classes",
                "initialization": "D0",
            }
            if method == "TRIM"
            else builders[method].as_dict()
        )
        summaries.append(summary)
        append_jsonl(experiment_dir / "run_summaries.jsonl", summary)
        raw_rows.extend(iteration_rows(
            task=task,
            summary=summary,
            experiment_id=experiment_id,
            experiment_kind="exp6a1_enumeration",
            method=method,
        ))
    write_csv_rows(experiment_dir / "run_summaries.csv", summaries)
    write_csv_rows(experiment_dir / "iterations.csv", raw_rows)
    for row in raw_rows:
        append_jsonl(experiment_dir / "iterations.jsonl", row)
    write_json(experiment_dir / "completed.json", {
        "experiment_id": experiment_id,
        "experiment_kind": "exp6a1_enumeration",
        "run_count": len(summaries),
        "iteration_row_count": len(raw_rows),
        "run_ids": [summary["run_id"] for summary in summaries],
    })
    print(experiment_dir)


if __name__ == "__main__":
    main()
