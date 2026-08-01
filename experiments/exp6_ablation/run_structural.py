"""Run the paper's Exp-6a2 structural ablation for TRIM."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math

from prototype import (
    ALL_ROWS_AT_MAX_GENERALIZATION,
    EVICT_LOW_K_POSTPROCESSING,
    FULL_SWAP_POSTPROCESSING,
    NO_VERTICAL_POSTPROCESSING,
    RETENTION_CLASS_ACTION,
    S0_INITIALIZATION,
    VERTICAL_REFINEMENT_ACTION,
    AblationActions,
)

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
from .policies import KMeansClusterRingBuilder
from .results import iteration_rows


METHODS = ("TRIM", "TRIM-H", "TRIM-V", "RefineOnly", "RefineEvict")


def _actions(move_out_threshold):
    return {
        "TRIM": AblationActions(),
        "TRIM-H": AblationActions(
            enabled_action_types=(RETENTION_CLASS_ACTION,),
            initialization=S0_INITIALIZATION,
            vertical_postprocessing=FULL_SWAP_POSTPROCESSING,
        ),
        "TRIM-V": AblationActions(
            enabled_action_types=(VERTICAL_REFINEMENT_ACTION,),
            initialization=ALL_ROWS_AT_MAX_GENERALIZATION,
            vertical_postprocessing=FULL_SWAP_POSTPROCESSING,
        ),
        "RefineOnly": AblationActions(
            enabled_action_types=(
                RETENTION_CLASS_ACTION,
                VERTICAL_REFINEMENT_ACTION,
            ),
            initialization=S0_INITIALIZATION,
            vertical_postprocessing=NO_VERTICAL_POSTPROCESSING,
        ),
        "RefineEvict": AblationActions(
            enabled_action_types=(
                RETENTION_CLASS_ACTION,
                VERTICAL_REFINEMENT_ACTION,
            ),
            initialization=S0_INITIALIZATION,
            vertical_postprocessing=EVICT_LOW_K_POSTPROCESSING,
            move_out_threshold=move_out_threshold,
        ),
    }


def _number(value, *, name, integer=False, minimum=0.0):
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be numeric.")
    parsed = int(value) if integer else float(value)
    if not math.isfinite(float(parsed)) or float(parsed) < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}.")
    if integer and float(parsed) != float(value):
        raise ValueError(f"{name} must be an integer.")
    return parsed


def _prepare(config_path, config, *, tolerance, actions):
    tasks = expand_matrix(config)
    if not tasks:
        raise ValueError(f"Exp-6a2 configuration produced no tasks: {config_path}")
    ring = config.get("horizontal_kmeans")
    if not isinstance(ring, dict):
        raise ValueError("Exp-6a2 configuration must define horizontal_kmeans.")
    ring_builder = KMeansClusterRingBuilder(
        n_clusters=_number(
            ring.get("n_clusters"), name="horizontal_kmeans.n_clusters",
            integer=True, minimum=1,
        ),
        n_rings=_number(
            ring.get("n_rings"), name="horizontal_kmeans.n_rings",
            integer=True, minimum=1,
        ),
        random_state=_number(
            ring.get("random_state"), name="horizontal_kmeans.random_state",
            integer=True,
        ),
    )
    tolerance = _number(tolerance, name="tolerance")

    seen = set()
    shared = None
    shared_s0 = None
    prepared = []
    for raw_task in tasks:
        task = deepcopy(raw_task)
        method = task.get("method")
        if method not in actions:
            raise ValueError(f"Unknown Exp-6a2 method {method!r}.")
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
            "ablation_actions",
            "fixed_row_candidate_builder",
            "candidate_shortlist",
            "candidate_state_scorer_factory",
            "selection_score",
        }.intersection(pipeline)
        if forbidden:
            raise ValueError(f"Exp-6a2 configuration cannot override hooks: {forbidden}.")
        pipeline["tolerance"] = tolerance
        if pipeline.get("record_iteration_test_metrics") is not True:
            raise ValueError(
                "Exp-6a2 requires pipeline.record_iteration_test_metrics=true."
            )
        if str(task.get("dataset", "")).lower() != "pubcov":
            raise ValueError("The Exp-6a2 configuration must use PubCov.")
        seed = int(pipeline.get("random_state"))
        combination = (method, seed)
        if combination in seen:
            raise ValueError(f"Duplicate Exp-6a2 method/seed: {combination}.")
        seen.add(combination)

        s0 = task.get("initial_sample_fraction")
        if method == "TRIM-H":
            if s0 is not None:
                raise ValueError("TRIM-H uses its largest ring, not S0.")
            pipeline["fixed_row_candidate_builder"] = ring_builder
        else:
            if s0 is None or not 0.0 < float(s0) <= 1.0:
                raise ValueError(f"{method} must declare a valid shared S0 fraction.")
            if shared_s0 is None:
                shared_s0 = float(s0)
            elif float(s0) != shared_s0:
                raise ValueError("Every S0-based structural method must share S0.")
        pipeline["ablation_actions"] = actions[method]

        comparison_task = deepcopy(task)
        comparison_task.pop("run_id", None)
        comparison_task.pop("axis_values", None)
        comparison_task.pop("method", None)
        comparison_task.pop("initial_sample_fraction", None)
        comparison_pipeline = comparison_task["pipeline"]
        comparison_pipeline.pop("random_state", None)
        comparison_pipeline.pop("tolerance", None)
        comparison_pipeline.pop("ablation_actions", None)
        comparison_pipeline.pop("fixed_row_candidate_builder", None)
        current_shared = json.dumps(comparison_task, sort_keys=True)
        if shared is None:
            shared = current_shared
        elif shared != current_shared:
            raise ValueError(
                "Apart from method, seed, initialization source, and declared "
                "structural hooks, every Exp-6a2 task setting must agree."
            )
        prepared.append(task)

    seeds = {seed for _method, seed in seen}
    expected = {(method, seed) for method in METHODS for seed in seeds}
    if seen != expected:
        raise ValueError("Exp-6a2 requires the complete method x seed matrix.")
    return prepared, ring_builder


def _manifest_task(task):
    output = deepcopy(task)
    pipeline = output["pipeline"]
    pipeline["ablation_actions"] = pipeline["ablation_actions"].as_dict()
    builder = pipeline.get("fixed_row_candidate_builder")
    if builder is not None:
        pipeline["fixed_row_candidate_builder"] = builder.as_dict()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tolerance", required=True, type=float)
    parser.add_argument("--device")
    parser.add_argument("--run-label")
    args = parser.parse_args()

    config_path, config = load_yaml_mapping(args.config)
    if config.get("experiment_kind") != "exp6a2_structural":
        raise ValueError("Expected experiment_kind='exp6a2_structural'.")
    if "tolerance" in config:
        raise ValueError("Tolerance must be supplied only by --tolerance.")
    experiment_id = str(config.get("experiment_id", "")).strip()
    if not experiment_id or config.get("results_root") is None:
        raise ValueError("Exp-6a2 requires experiment_id and results_root.")
    move_out_threshold = _number(
        config.get("refine_evict_threshold"),
        name="refine_evict_threshold",
        integer=True,
        minimum=1,
    )
    actions = _actions(move_out_threshold)
    require_core_parameters(
        "ablation_actions",
        "fixed_row_candidate_builder",
        "record_iteration_test_metrics",
    )
    tasks, ring_builder = _prepare(
        config_path,
        config,
        tolerance=args.tolerance,
        actions=actions,
    )
    experiment_dir = create_experiment_dir(
        config["results_root"], experiment_id, run_label=args.run_label
    )
    write_json(experiment_dir / "manifest.json", {
        "experiment_id": experiment_id,
        "experiment_kind": "exp6a2_structural",
        "source_config": str(config_path),
        "device_override": args.device,
        "reader_supplied_tolerance": float(args.tolerance),
        "refine_evict_threshold": move_out_threshold,
        "horizontal_kmeans": {
            "n_clusters": ring_builder.n_clusters,
            "n_rings": ring_builder.n_rings,
            "random_state": ring_builder.random_state,
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
        summary["structural_definition"] = actions[method].as_dict()
        summaries.append(summary)
        append_jsonl(experiment_dir / "run_summaries.jsonl", summary)
        raw_rows.extend(iteration_rows(
            task=task,
            summary=summary,
            experiment_id=experiment_id,
            experiment_kind="exp6a2_structural",
            method=method,
        ))
    write_csv_rows(experiment_dir / "run_summaries.csv", summaries)
    write_csv_rows(experiment_dir / "iterations.csv", raw_rows)
    for row in raw_rows:
        append_jsonl(experiment_dir / "iterations.jsonl", row)
    write_json(experiment_dir / "completed.json", {
        "experiment_id": experiment_id,
        "experiment_kind": "exp6a2_structural",
        "run_count": len(summaries),
        "iteration_row_count": len(raw_rows),
        "run_ids": [summary["run_id"] for summary in summaries],
    })
    print(experiment_dir)


if __name__ == "__main__":
    main()
