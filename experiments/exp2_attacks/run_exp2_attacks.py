"""Run reconstruction and strong-linkage attacks on declared TRIM releases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from prototype.dataset_registry import (
    build_data_loader,
    resolve_generalization_tree,
)
from prototype.release_artifacts import (
    load_trim_release,
    materialize_trim_release,
)

from experiments.common import (
    append_jsonl,
    create_experiment_dir,
    load_yaml_mapping,
    merge_mappings,
    resolve_project_path,
    run_core_task,
    write_csv_rows,
    write_json,
)

from .attacks import run_reconstruction_attack, run_strong_linkage_attack


def _iteration_numbers(run_dir: Path) -> list[int]:
    path = run_dir / "trim_iterations.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid TRIM iteration file: {path}")
    iterations = payload.get("iterations")
    if not isinstance(iterations, list):
        raise ValueError(f"Invalid TRIM iteration file: {path}")
    if not iterations:
        raise ValueError(f"TRIM iteration file is empty: {path}")
    if any(not isinstance(row, dict) or "iteration" not in row for row in iterations):
        raise ValueError(f"TRIM iteration rows are invalid: {path}")
    numbers = [int(row["iteration"]) for row in iterations]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"Duplicate TRIM iteration numbers in {path}")
    return sorted(numbers)


def _release_points(run_dir: Path, configured) -> list[int | None]:
    if configured == "all_trim_iterations":
        return [*_iteration_numbers(run_dir), None]
    if not isinstance(configured, list) or not configured:
        raise ValueError(
            "release_points must be 'all_trim_iterations' or a non-empty list."
        )
    points: list[int | None] = []
    for value in configured:
        point = None if str(value).lower() == "final" else int(value)
        if point in points:
            raise ValueError(f"Duplicate configured release point: {value!r}")
        points.append(point)
    return points


def _resolve_declared_runs(defaults, declared_runs) -> list[dict]:
    """Merge defaults into each explicitly declared TRIM run."""
    resolved = []
    for index, declared in enumerate(declared_runs):
        if not isinstance(declared, dict):
            raise ValueError(f"runs[{index}] must be a YAML mapping.")
        if "seed" not in declared or declared["seed"] is None:
            raise ValueError(f"runs[{index}] must explicitly declare its seed.")
        task = merge_mappings(defaults, declared)
        if declared.get("trim_run_dir") is not None and "trim_task" not in declared:
            task["trim_task"] = None
        if declared.get("trim_task") is not None and "trim_run_dir" not in declared:
            task["trim_run_dir"] = None
        resolved.append(task)

    run_ids = [str(task.get("run_id", "")).strip() for task in resolved]
    if any(not run_id for run_id in run_ids):
        raise ValueError("Every attack run must define a non-empty run_id.")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Attack run_id values must be unique.")
    invalid_run_ids = [
        run_id
        for run_id in run_ids
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id) is None
    ]
    if invalid_run_ids:
        raise ValueError(
            "Attack run_id values may use only letters, digits, '_' and '-': "
            f"{invalid_run_ids}"
        )
    for run_id, task in zip(run_ids, resolved):
        configured_run_dir = task.get("trim_run_dir")
        trim_task = task.get("trim_task")
        if (configured_run_dir is None) == (trim_task is None):
            raise ValueError(
                f"Run {run_id!r} must define exactly one of trim_run_dir or trim_task."
            )
        if trim_task is not None and not isinstance(trim_task, dict):
            raise ValueError(f"Run {run_id!r} trim_task must be a YAML mapping.")
    seeds = [int(task["seed"]) for task in resolved]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Attack run seeds must be unique.")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate attacks on explicitly declared TRIM releases."
    )
    parser.add_argument("--config", required=True, help="Attack experiment YAML.")
    parser.add_argument("--device", help="Override the attack-model device.")
    parser.add_argument("--run-label", help="Optional output-directory label.")
    args = parser.parse_args()

    config_path, config = load_yaml_mapping(args.config)
    experiment_id = str(config.get("experiment_id", "exp2_attacks"))
    results_root = config.get("results_root")
    if results_root is None:
        raise ValueError("Attack config must define results_root.")
    declared_runs = config.get("runs")
    if not isinstance(declared_runs, list) or not declared_runs:
        raise ValueError("Attack config must contain a non-empty runs list.")
    defaults = config.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a YAML mapping.")

    resolved_runs = _resolve_declared_runs(defaults, declared_runs)
    configured_seeds = [int(task["seed"]) for task in resolved_runs]
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
            "configured_seeds": configured_seeds,
            "resolved_runs": resolved_runs,
            "device_override": args.device,
        },
    )

    flat_rows = []
    for task in resolved_runs:
        run_id = str(task.get("run_id", "")).strip()
        dataset = str(task.get("dataset", "")).strip()
        model_name = str(task.get("model_name", "")).strip()
        if not run_id or not dataset or not model_name:
            raise ValueError("Each run needs run_id, dataset, and model_name.")
        configured_run_dir = task.get("trim_run_dir")
        trim_task = task.get("trim_task")
        if (configured_run_dir is None) == (trim_task is None):
            raise ValueError(
                f"Run {run_id!r} must define exactly one of trim_run_dir or trim_task."
            )
        if trim_task is not None:
            if not isinstance(trim_task, dict):
                raise ValueError(f"Run {run_id!r} trim_task must be a YAML mapping.")
            trim_model_name = str(trim_task.get("model_name", "")).strip()
            if trim_model_name and trim_model_name != model_name:
                raise ValueError(
                    f"Run {run_id!r} model_name={model_name!r} disagrees with "
                    f"trim_task.model_name={trim_model_name!r}."
                )
            core_task = merge_mappings(
                {
                    "run_id": f"{run_id}_trim",
                    "dataset": dataset,
                    "data_path": task.get("data_path"),
                    "device": task.get("device", "cuda"),
                    "dtype": task.get("dtype", "float32"),
                    "allow_cpu_fallback": task.get("allow_cpu_fallback", True),
                },
                trim_task,
            )
            core_summary = run_core_task(
                core_task,
                experiment_dir=experiment_dir,
                device=args.device,
            )
            trim_run_dir = Path(core_summary["run_dir"])
        else:
            trim_run_dir = resolve_project_path(configured_run_dir)
            if trim_run_dir is None:
                raise ValueError(f"Run {run_id!r} has an empty trim_run_dir.")
        loader = build_data_loader(
            dataset,
            data_path=resolve_project_path(task.get("data_path")),
        )
        tree_path = resolve_generalization_tree(
            dataset,
            tree_path=resolve_project_path(task.get("generalization_tree_path")),
        )
        points = _release_points(trim_run_dir, task.get("release_points"))
        device = str(args.device or task.get("device", "cuda"))
        dtype = str(task.get("dtype", "float32"))
        allow_cpu_fallback = bool(task.get("allow_cpu_fallback", True))

        for point in points:
            loaded = load_trim_release(trim_run_dir, iteration=point)
            original_leak_k_value = loaded["release"].get("original_leak_k")
            if isinstance(original_leak_k_value, bool):
                raise ValueError(
                    f"Run {run_id!r} release {point!r} has an invalid "
                    "original_leak_k reference."
                )
            try:
                original_leak_k = int(original_leak_k_value)
                original_leak_k_numeric = float(original_leak_k_value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"Run {run_id!r} release {point!r} is missing a valid "
                    "original_leak_k reference."
                ) from error
            if (
                original_leak_k <= 0
                or original_leak_k_numeric != original_leak_k
            ):
                raise ValueError(
                    f"Run {run_id!r} release {point!r} needs a positive integer "
                    f"original_leak_k; got {original_leak_k_value!r}."
                )
            if "random_state" not in loaded["config"]:
                raise ValueError(
                    f"TRIM run {run_id!r} does not record random_state."
                )
            persisted_seed = int(loaded["config"]["random_state"])
            configured_seed = task.get("seed")
            if configured_seed is not None and int(configured_seed) != persisted_seed:
                raise ValueError(
                    f"Run {run_id!r} seed disagrees with its TRIM output: "
                    f"{configured_seed} != {persisted_seed}."
                )
            materialized = materialize_trim_release(
                loader,
                tree_path,
                loaded,
                evaluation_split=str(task.get("evaluation_split", "test")),
            )
            reconstruction = run_reconstruction_attack(
                materialized,
                loader,
                task.get("reconstruction") or {},
                random_state=persisted_seed,
                device=device,
                dtype=dtype,
                allow_cpu_fallback=allow_cpu_fallback,
            )
            linkage = run_strong_linkage_attack(
                materialized,
                task.get("linkage") or {},
                random_state=persisted_seed,
                device=device,
                dtype=dtype,
                allow_cpu_fallback=allow_cpu_fallback,
            )
            point_label = "final" if point is None else f"iteration_{point:04d}"
            detail = {
                "run_id": run_id,
                "dataset": dataset,
                "model_name": model_name,
                "seed": persisted_seed,
                "artifact_run_dir": str(trim_run_dir),
                "release_point": point_label,
                "evaluation_split": materialized.evaluation_split,
                "release": {
                    "status": materialized.release.get("status"),
                    "iteration": materialized.release.get("iteration"),
                    "target_k": materialized.release.get("target_k"),
                    "min_k": materialized.release.get("min_k"),
                    "original_leak_k": original_leak_k,
                    "row_count": materialized.release.get("row_count"),
                    "suppressed_row_count": materialized.release.get(
                        "suppressed_row_count"
                    ),
                    "coverage_fraction": materialized.release.get(
                        "coverage_fraction"
                    ),
                    "utility_constraint_met": materialized.release.get(
                        "utility_constraint_met"
                    ),
                    "assignment_counts": materialized.release.get(
                        "assignment_counts"
                    ),
                    "snapshot_history": materialized.release.get(
                        "snapshot_history"
                    ),
                    "published_row_semantics": (
                        "assigned training rows materialized at their persisted "
                        "TRIM snapshot levels"
                    ),
                    "evaluation_row_semantics": (
                        f"routed held-out {materialized.evaluation_split} rows"
                    ),
                },
                "reconstruction": reconstruction,
                "linkage": linkage,
            }
            write_json(
                experiment_dir / "details" / run_id / f"{point_label}.json",
                detail,
            )
            release = materialized.release
            reconstruction_error = reconstruction.get(
                "test_reconstruction_error_mean"
            )
            reconstruction_baseline_error = reconstruction.get(
                "baseline_test_reconstruction_error_mean"
            )
            linkage_error = linkage.get("test_linkage_error")
            linkage_baseline_error = linkage.get(
                "baseline_test_linkage_error"
            )
            reconstruction_attributes = reconstruction.get("attribute_results", ())
            row = {
                "run_id": run_id,
                "dataset": dataset,
                "model_name": model_name,
                "seed": persisted_seed,
                "artifact_run_dir": str(trim_run_dir),
                "iteration": point,
                "release_point": point_label,
                "evaluation_split": materialized.evaluation_split,
                "published_row_semantics": (
                    "assigned mixed-level TRIM training rows"
                ),
                "evaluation_row_semantics": (
                    f"routed held-out {materialized.evaluation_split} rows"
                ),
                "target_k": release.get("target_k"),
                "min_k": release.get("min_k"),
                "original_leak_k": original_leak_k,
                "published_row_count": len(materialized.published_frame),
                "suppressed_row_count": len(release.get("suppressed_row_ids", ())),
                "coverage_fraction": release.get("coverage_fraction"),
                "utility_constraint_met": release.get("utility_constraint_met"),
                "release_snapshot_count": len(release.get("snapshot_history", ())),
                "release_assignment_counts": release.get("assignment_counts"),
                "reconstruction_error": reconstruction_error,
                "reconstruction_baseline_error": reconstruction_baseline_error,
                "reconstruction_accuracy_gain_over_baseline": (
                    None
                    if reconstruction_error is None
                    or reconstruction_baseline_error is None
                    else reconstruction_baseline_error - reconstruction_error
                ),
                "linkage_error": linkage_error,
                "linkage_baseline_error": linkage_baseline_error,
                "linkage_accuracy_gain_over_baseline": (
                    None
                    if linkage_error is None or linkage_baseline_error is None
                    else linkage_baseline_error - linkage_error
                ),
                "fold_count": reconstruction.get("fold_count"),
                "fold_index": reconstruction.get("fold_index"),
                "numeric_tolerance": reconstruction.get("numeric_tolerance"),
                "reconstruction_training_row_count": reconstruction.get(
                    "attacker_training_row_count"
                ),
                "reconstruction_evaluation_row_count": reconstruction.get(
                    "evaluation_row_count"
                ),
                "reconstruction_attribute_count": len(reconstruction_attributes),
                "reconstruction_skipped_attribute_count": sum(
                    bool(attribute.get("skipped"))
                    for attribute in reconstruction_attributes
                ),
                "linkage_strategy": linkage.get("strategy"),
                "query_size": linkage.get("query_size"),
                "query_count": linkage.get("query_count"),
                "linkage_training_row_count": linkage.get("training_row_count"),
                "linkage_evaluation_row_count": linkage.get(
                    "evaluation_row_count"
                ),
                "linkage_class_count": linkage.get("class_count"),
                "linkage_test_accuracy": linkage.get("test_accuracy"),
                "linkage_baseline_test_accuracy": linkage.get(
                    "baseline_test_accuracy"
                ),
                "linkage_test_leakage_gap": linkage.get("test_leakage_gap"),
                "linkage_skipped": linkage.get("skipped"),
                "linkage_skip_reason": linkage.get("skip_reason"),
                "a_attributes": ",".join(linkage.get("a_attributes", ())),
                "b_attributes": ",".join(linkage.get("b_attributes", ())),
            }
            flat_rows.append(row)
            append_jsonl(experiment_dir / "results.jsonl", row)

    write_csv_rows(experiment_dir / "results.csv", flat_rows)
    write_json(
        experiment_dir / "completed.json",
        {
            "experiment_id": experiment_id,
            "configured_seeds": configured_seeds,
            "run_count": len(resolved_runs),
            "result_count": len(flat_rows),
        },
    )
    print(experiment_dir)


if __name__ == "__main__":
    main()
