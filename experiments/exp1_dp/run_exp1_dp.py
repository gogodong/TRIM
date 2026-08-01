"""Run the configured DP comparison from one explicit TRIM release."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import (
    append_jsonl,
    create_experiment_dir,
    load_yaml_mapping,
    resolve_project_path,
    write_csv_rows,
    write_json,
)
from prototype.backend_training import fit_standardizer
from prototype.dataset_registry import (
    build_data_loader,
    resolve_generalization_tree,
)
from prototype.greedy_selection import calculate_leakage
from prototype.release_artifacts import (
    load_dataset_split,
    load_trim_release,
    materialize_trim_release,
)

from .mondrian import encode_mondrian, mondrian_partition, partition_records
from .summarize import REQUIRED_METHODS, summarize_rows
from .training import (
    MethodDataset,
    NoiseCalibration,
    _require_opacus,
    calibrate_noise_multiplier,
    person_level_group_privacy_bound,
    train_nonprivate_sgd,
    train_private_or_clipped_sgd,
)


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping.")
    return dict(value)


def _validate_optional_dependencies() -> None:
    _require_opacus()
    try:
        import scipy.stats  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Experiment 1 requires the optional `scipy` package for its "
            "paired Student-t summary. Install the project environment "
            "before running this experiment."
        ) from exc


def _assert_equal_float(actual: Any, expected: Any, *, field: str) -> None:
    if not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"TRIM run {field}={actual!r} does not match the configured "
            f"experiment value {expected!r}."
        )


def _result_row(
    *,
    dataset: MethodDataset,
    outcome,
    training_mode: str,
    seed: int,
    target_k: int,
    training: dict[str, Any],
    x: float | None = None,
    target_epsilon: float | None = None,
    noise_multiplier: float | None = None,
) -> dict[str, Any]:
    actual_epsilon = outcome.actual_epsilon
    person_epsilon = None
    person_delta = None
    if actual_epsilon is not None:
        person_epsilon, person_delta = person_level_group_privacy_bound(
            row_epsilon=actual_epsilon,
            row_delta=float(training["delta"]),
            max_contributions=1,
        )
    return {
        "experiment": f"exp1_dp_k{int(target_k)}",
        "dataset": "income",
        "method": dataset.method,
        "training_mode": training_mode,
        "seed": int(seed),
        "target_k": int(target_k),
        "final_min_k": int(dataset.final_min_k),
        "privacy_constraint_met": bool(dataset.privacy_constraint_met),
        "row_count": int(dataset.row_count),
        "unique_row_count": int(dataset.unique_row_count),
        "published_row_count": int(dataset.published_row_count),
        "row_semantics": dataset.row_semantics,
        "source_detail": dataset.source_detail,
        "evaluation_split": "test",
        "evaluation_protocol": dataset.evaluation_protocol,
        "test_logloss": float(outcome.test_logloss),
        "baseline_logloss": None,
        "delta_logloss": None,
        "optimizer": "sgd",
        "model_hidden_sizes": [int(v) for v in training["hidden_sizes"]],
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "epochs": int(training["epochs"]),
        "requested_batch_size": int(training["batch_size"]),
        "expected_batch_size": int(outcome.expected_batch_size),
        "sample_rate": float(outcome.sample_rate),
        "training_steps": int(outcome.training_steps),
        "poisson_sampling": True,
        "max_grad_norm": (
            None
            if training_mode == "sgd"
            else float(training["max_grad_norm"])
        ),
        "x": None if x is None else float(x),
        "target_epsilon": (
            None if target_epsilon is None else float(target_epsilon)
        ),
        "actual_epsilon": (
            None if actual_epsilon is None else float(actual_epsilon)
        ),
        "noise_multiplier": (
            None if noise_multiplier is None else float(noise_multiplier)
        ),
        "dp_delta": (
            None if training_mode != "dp_sgd" else float(training["delta"])
        ),
        "dp_accountant": (
            None if training_mode != "dp_sgd" else str(training["accountant"])
        ),
        "dp_privacy_unit": (
            None if training_mode != "dp_sgd" else "one training row"
        ),
        "person_level_max_contributions": (
            None if training_mode != "dp_sgd" else 1
        ),
        "person_level_epsilon_upper_bound": person_epsilon,
        "person_level_delta_upper_bound": person_delta,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare matched SGD, clipped-SGD, and DP-SGD on an explicit "
            "configured-K final TRIM release."
        )
    )
    parser.add_argument("--config", required=True, help="Experiment 1 YAML.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="TRIM run directory containing trim_release.json.",
    )
    parser.add_argument("--device", help="Override the configured device.")
    parser.add_argument("--run-label", help="Optional output-directory label.")
    args = parser.parse_args(argv)

    config_path, config = load_yaml_mapping(args.config)
    dataset_config = _mapping(config.get("dataset"), name="dataset")
    release_config = _mapping(config.get("release"), name="release")
    training = _mapping(config.get("training"), name="training")
    summary_config = _mapping(config.get("summary"), name="summary")
    if str(dataset_config.get("key", "")).lower() != "income":
        raise ValueError("The fixed Experiment 1 protocol supports Income only.")
    if release_config.get("target_k") is None:
        raise ValueError("Experiment 1 configuration must declare release.target_k.")
    target_k = int(release_config["target_k"])
    if target_k < 1:
        raise ValueError("release.target_k must be at least 1.")
    methods = list(config.get("methods") or [])
    if methods != list(REQUIRED_METHODS):
        raise ValueError(
            f"Experiment 1 methods must be ordered as {list(REQUIRED_METHODS)}."
        )
    seeds = [int(seed) for seed in training.get("seeds") or []]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError(
            "training.seeds must contain at least two distinct values."
        )
    x_values = [float(value) for value in training.get("x_values") or []]
    if (
        not x_values
        or len(set(x_values)) != len(x_values)
        or any(not math.isfinite(value) or value <= 1.0 for value in x_values)
    ):
        raise ValueError(
            "training.x_values must contain distinct finite values greater than 1."
        )
    if not bool(training.get("poisson_sampling")):
        raise ValueError("Experiment 1 requires matched Poisson sampling.")
    if str(training.get("optimizer", "")).lower() != "sgd":
        raise ValueError("Experiment 1 requires SGD for all three training modes.")
    _validate_optional_dependencies()

    run_dir = Path(args.run_dir).expanduser().resolve()
    loaded_release = load_trim_release(run_dir)
    artifact_config = dict(loaded_release["config"])
    release = dict(loaded_release["release"])
    if int(release["target_k"]) != target_k:
        raise ValueError(
            "The explicit final TRIM release target does not match "
            f"release.target_k={target_k}."
        )
    if int(release.get("min_k", -1)) < target_k:
        raise ValueError(
            f"The explicit final TRIM release has min-K below {target_k}."
        )
    if int(artifact_config.get("release_target_k", -1)) != target_k:
        raise ValueError(
            "The TRIM run release_target_k does not match "
            f"release.target_k={target_k}."
        )
    if int(artifact_config.get("nrows", -1)) != int(
        dataset_config["expected_nrows"]
    ):
        raise ValueError("TRIM run nrows does not match Experiment 1.")
    _assert_equal_float(
        artifact_config.get("val_size"),
        dataset_config["expected_val_size"],
        field="val_size",
    )
    _assert_equal_float(
        artifact_config.get("test_size"),
        dataset_config["expected_test_size"],
        field="test_size",
    )
    if int(artifact_config.get("random_state", -1)) != int(
        dataset_config["expected_split_seed"]
    ):
        raise ValueError("TRIM run split seed does not match Experiment 1.")

    data_loader = build_data_loader(
        "income",
        data_path=resolve_project_path(dataset_config.get("data_path")),
    )
    tree_path = resolve_generalization_tree(
        "income",
        tree_path=resolve_project_path(
            dataset_config.get("generalization_tree_path")
        ),
    )
    materialized = materialize_trim_release(
        data_loader, tree_path, loaded_release, evaluation_split="test"
    )
    split = load_dataset_split(data_loader, artifact_config)
    original_train = data_loader.encode_original(split.X_train_raw)
    original_test = data_loader.encode_original(split.X_test_raw)

    numeric_attributes = tuple(data_loader.numeric_attributes)
    categorical_attributes = tuple(data_loader.categorical_attributes)
    partitions = mondrian_partition(
        split.X_train_raw,
        k=target_k,
        numeric_attributes=numeric_attributes,
        categorical_attributes=categorical_attributes,
    )
    mondrian_train = encode_mondrian(
        split.X_train_raw,
        partitions,
        numeric_attributes=numeric_attributes,
        categorical_attributes=categorical_attributes,
        category_maps=data_loader.category_maps,
    )
    if list(mondrian_train.columns) != list(original_train.columns):
        raise RuntimeError("Mondrian and level-0 model columns do not match.")
    if list(materialized.published_frame.columns) != list(original_train.columns):
        raise RuntimeError("TRIM and level-0 model columns do not match.")

    method_datasets = {
        "original": MethodDataset(
            method="original",
            train_frame=original_train,
            train_y=split.y_train,
            evaluation_frame=original_test,
            evaluation_y=split.y_test,
            final_min_k=int(calculate_leakage(original_train)),
            row_count=len(original_train),
            unique_row_count=len(original_train),
            published_row_count=len(original_train),
            source_detail="recreated level-0 training split",
            row_semantics="all distinct training contributors",
            evaluation_protocol="train_level0__test_level0",
            privacy_constraint_met=(
                int(calculate_leakage(original_train)) >= target_k
            ),
        ),
        "mondrian": MethodDataset(
            method="mondrian",
            train_frame=mondrian_train,
            train_y=split.y_train,
            evaluation_frame=original_test,
            evaluation_y=split.y_test,
            final_min_k=int(calculate_leakage(mondrian_train)),
            row_count=len(mondrian_train),
            unique_row_count=len(mondrian_train),
            published_row_count=len(mondrian_train),
            source_detail="Mondrian full release fitted on the training split",
            row_semantics="all distinct training contributors",
            evaluation_protocol="train_mondrian__test_level0",
            privacy_constraint_met=(
                int(calculate_leakage(mondrian_train)) >= target_k
            ),
        ),
        "trim": MethodDataset(
            method="trim",
            train_frame=materialized.published_frame,
            train_y=materialized.published_y,
            evaluation_frame=materialized.evaluation_frame,
            evaluation_y=materialized.evaluation_y,
            final_min_k=int(calculate_leakage(materialized.published_frame)),
            row_count=len(materialized.published_frame),
            unique_row_count=len(materialized.published_frame),
            published_row_count=len(materialized.published_frame),
            source_detail=f"final TRIM release from {run_dir}",
            row_semantics="assigned distinct training contributors",
            evaluation_protocol="train_trim__test_trim_routed",
            privacy_constraint_met=(
                int(calculate_leakage(materialized.published_frame))
                >= target_k
            ),
        ),
    }
    if not method_datasets["mondrian"].privacy_constraint_met:
        raise RuntimeError(f"Mondrian failed the declared K={target_k} constraint.")
    if not method_datasets["trim"].privacy_constraint_met:
        raise RuntimeError(
            f"The materialized TRIM release failed K={target_k}."
        )

    standardizer = fit_standardizer(original_train, data_loader)
    device = str(args.device or training.get("device", "cuda"))
    allow_cpu_fallback = bool(training.get("allow_cpu_fallback", True))
    dtype = str(training.get("dtype", "float32"))
    classes = np.unique(np.asarray(split.y_train))
    if len(classes) < 2:
        raise ValueError("Experiment 1 requires at least two target classes.")

    experiment_id = str(config.get("experiment_id", f"exp1_dp_k{target_k}"))
    experiment_dir = create_experiment_dir(
        config["results_root"], experiment_id, run_label=args.run_label
    )
    write_json(
        experiment_dir / "manifest.json",
        {
            "experiment_id": experiment_id,
            "source_config": str(config_path),
            "trim_run_dir": str(run_dir),
            "trim_release_file": str(run_dir / "trim_release.json"),
            "trim_release": release,
            "generalization_tree": str(tree_path),
            "device": device,
            "allow_cpu_fallback": allow_cpu_fallback,
            "resolved_config": config,
        },
    )
    write_json(
        experiment_dir / "mondrian_partitions.json",
        {
            "target_k": target_k,
            "partition_count": len(partitions),
            "partitions": partition_records(
                split.X_train_raw,
                partitions,
                numeric_attributes=numeric_attributes,
                categorical_attributes=categorical_attributes,
            ),
        },
    )

    common_training_kwargs = {
        "classes": classes,
        "hidden_sizes": tuple(int(v) for v in training["hidden_sizes"]),
        "epochs": int(training["epochs"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "batch_size": int(training["batch_size"]),
        "device": device,
        "dtype": dtype,
        "allow_cpu_fallback": allow_cpu_fallback,
    }
    private_training_kwargs = {
        **common_training_kwargs,
        "max_grad_norm": float(training["max_grad_norm"]),
        "delta": float(training["delta"]),
        "accountant": str(training["accountant"]),
    }
    epsilon_tolerance = float(training["epsilon_tolerance"])
    calibration_cache: dict[tuple[int, float], NoiseCalibration] = {}
    raw_rows = []
    for seed in seeds:
        seed_rows = []
        for method in REQUIRED_METHODS:
            dataset = method_datasets[method]
            sgd = train_nonprivate_sgd(
                dataset,
                standardizer,
                random_state=seed,
                **common_training_kwargs,
            )
            seed_rows.append(
                _result_row(
                    dataset=dataset,
                    outcome=sgd,
                    training_mode="sgd",
                    seed=seed,
                    target_k=target_k,
                    training=training,
                )
            )
            clipped = train_private_or_clipped_sgd(
                dataset,
                standardizer,
                calibration=NoiseCalibration(0.0, math.inf),
                random_state=seed,
                **private_training_kwargs,
            )
            seed_rows.append(
                _result_row(
                    dataset=dataset,
                    outcome=clipped,
                    training_mode="clipped_sgd",
                    seed=seed,
                    target_k=target_k,
                    training=training,
                    noise_multiplier=0.0,
                )
            )
            for x_value in x_values:
                target_epsilon = math.log(x_value)
                calibration_key = (dataset.row_count, x_value)
                if calibration_key not in calibration_cache:
                    calibration_cache[calibration_key] = calibrate_noise_multiplier(
                        target_epsilon=target_epsilon,
                        dataset_size=dataset.row_count,
                        batch_size=int(training["batch_size"]),
                        epochs=int(training["epochs"]),
                        delta=float(training["delta"]),
                        tolerance=epsilon_tolerance,
                        accountant=str(training["accountant"]),
                    )
                calibration = calibration_cache[calibration_key]
                private = train_private_or_clipped_sgd(
                    dataset,
                    standardizer,
                    calibration=calibration,
                    random_state=seed,
                    **private_training_kwargs,
                )
                if private.actual_epsilon is None or abs(
                    private.actual_epsilon - target_epsilon
                ) > epsilon_tolerance:
                    raise RuntimeError(
                        "Realized DP epsilon is outside the configured tolerance: "
                        f"method={method}, seed={seed}, x={x_value}."
                    )
                seed_rows.append(
                    _result_row(
                        dataset=dataset,
                        outcome=private,
                        training_mode="dp_sgd",
                        seed=seed,
                        target_k=target_k,
                        training=training,
                        x=x_value,
                        target_epsilon=target_epsilon,
                        noise_multiplier=calibration.noise_multiplier,
                    )
                )

        baseline_candidates = [
            row
            for row in seed_rows
            if row["method"] == "original" and row["training_mode"] == "sgd"
        ]
        if len(baseline_candidates) != 1:
            raise RuntimeError("Could not identify the seed-matched original SGD baseline.")
        baseline = float(baseline_candidates[0]["test_logloss"])
        for row in seed_rows:
            row["baseline_logloss"] = baseline
            row["delta_logloss"] = float(row["test_logloss"]) - baseline
            raw_rows.append(row)
            append_jsonl(experiment_dir / "raw_results.jsonl", row)

    write_json(experiment_dir / "raw_results.json", raw_rows)
    write_csv_rows(experiment_dir / "raw_results.csv", raw_rows)
    summary = summarize_rows(
        raw_rows,
        noninferiority_margin=float(summary_config["noninferiority_margin"]),
        confidence_level=float(summary_config["confidence_level"]),
        expected_seeds=seeds,
        expected_x_values=x_values,
    )
    write_json(experiment_dir / "paired_summary.json", summary)
    flat_summary_rows = []
    for row in summary["dp_summaries"]:
        flat = {key: value for key, value in row.items() if key != "paired_vs_mondrian"}
        comparison = row.get("paired_vs_mondrian") or {}
        flat.update({f"vs_mondrian_{key}": value for key, value in comparison.items()})
        flat_summary_rows.append(flat)
    write_csv_rows(experiment_dir / "paired_summary.csv", flat_summary_rows)
    write_csv_rows(
        experiment_dir / "training_mode_summary.csv", summary["mode_summaries"]
    )
    write_json(
        experiment_dir / "completed.json",
        {
            "experiment_id": experiment_id,
            "raw_result_count": len(raw_rows),
            "seed_count": len(seeds),
            "trim_run_dir": str(run_dir),
        },
    )
    print(experiment_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
