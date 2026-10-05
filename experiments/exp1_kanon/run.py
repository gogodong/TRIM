"""Run a validation-only Income pilot or a fixed-variant four-dataset sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from experiments.common import (
    append_jsonl, create_experiment_dir, load_yaml_mapping, resolve_project_path,
    write_csv_rows, write_json,
)
from experiments.run_matrix_sweep import expand_matrix
from prototype.TRIM_prototype_pipeline import _loaded_input_sha256
from prototype.backend_training import fit_standardizer
from prototype.dataloader import load_generalization_rules_from_file
from prototype.dataset_registry import build_data_loader, resolve_dataset, resolve_generalization_tree
from prototype.gpu_math import classification_log_loss_tensor, to_device_tensor
from prototype.model_factory import build_model_factory
from prototype.privacy_metrics import individual_tail_risk_stats, original_equivalence_class_sizes
from prototype.release_artifacts import load_dataset_split

from .mondrian import EVALUATION_ROUTING, HierarchySchema
from .artifacts import MondrianReleaseCache


def k_grid(n_train, *, extras=(12, 235), explicit=None):
    """Powers of two through N_train/2, additional comparison Ks, and N_train."""
    if n_train < 2:
        raise ValueError("The sweep needs at least two training records.")
    if explicit is not None:
        values = list(explicit)
    else:
        values = [n_train]
        value = 2
        while value <= n_train // 2:
            values.append(value)
            value *= 2
        values.extend(value for value in extras if 2 <= value <= n_train)
    if not values or any(isinstance(value, bool) or int(value) != value or not 2 <= value <= n_train for value in values):
        raise ValueError("Sweep Ks must be integers in [2, N_train]; raw reference is separate.")
    return sorted({int(value) for value in values})


def implementation_sha256():
    """Invalidate pilot decisions if the anonymizer or shared protocol changes."""
    from prototype import backend_training, dataloader, gpu_math, gpu_mlp, gpu_xgboost, model_factory, privacy_metrics, release_artifacts
    from . import artifacts, mondrian

    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    for module in (artifacts, mondrian, backend_training, dataloader, gpu_math, gpu_mlp, gpu_xgboost, model_factory, privacy_metrics, release_artifacts):
        digest.update(Path(module.__file__).read_bytes())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("pilot", "full"))
    parser.add_argument("--selection", help="Recorded choice from the validation-only pilot.")
    parser.add_argument("--device")
    parser.add_argument("--run-label")
    parser.add_argument("--k", action="append", type=int, help="Explicit K; repeat to replace the grid.")
    args = parser.parse_args(argv)
    config_path, config = load_yaml_mapping(args.config)
    if config.get("experiment_kind") != "exp1_kanon":
        raise ValueError("Expected experiment_kind: exp1_kanon.")
    matrix_path, matrix_config = load_yaml_mapping(resolve_project_path(config["matrix_config"]))
    if matrix_config.get("experiment_kind") != "exp1_privacy_utility":
        raise ValueError("matrix_config must declare the shared Exp-1 matrix.")
    mode = args.mode or config.get("mode", "pilot")
    if mode not in {"pilot", "full"}:
        raise ValueError("mode must be pilot or full.")
    routing_review_threshold = float(config.get("routing_review_threshold", 0.01))
    if not 0 <= routing_review_threshold <= 1:
        raise ValueError("routing_review_threshold must be in [0, 1].")
    code_hash = implementation_sha256()
    selection = None
    if mode == "full":
        if args.selection is None:
            raise ValueError("Full sweep requires --selection from the validation-only pilot.")
        selection = json.loads(Path(args.selection).expanduser().resolve().read_text(encoding="utf-8"))
        if (
            selection.get("schema_version") != "kanon_selection.v1"
            or selection.get("variant") not in {"median", "infogain"}
            or selection.get("evaluation_split") != "validation"
            or selection.get("implementation_sha256") != code_hash
            or not selection.get("rationale")
            or not selection.get("income_protocol_signatures")
            or selection.get("selection_scope") != "one_variant_for_all_datasets_models_seeds_and_Ks"
        ):
            raise ValueError("Selection must record a validation-only decision for this implementation.")
        pilot_dir = Path(selection["pilot_dir"])
        pilot_manifest = json.loads((pilot_dir / "manifest.json").read_text(encoding="utf-8"))
        pilot_hash = hashlib.sha256((pilot_dir / "raw_results.csv").read_bytes()).hexdigest()
        if (
            pilot_manifest.get("mode") != "pilot" or pilot_manifest.get("status") != "complete"
            or pilot_manifest.get("implementation_sha256") != code_hash
            or pilot_hash != selection.get("pilot_results_sha256")
            or pilot_manifest.get("income_protocol_signatures") != selection["income_protocol_signatures"]
        ):
            raise ValueError("Recorded pilot evidence changed or is incomplete.")
        variants = [selection["variant"]]
    else:
        if args.selection:
            raise ValueError("Pilot mode evaluates both variants and does not consume a selection.")
        variants = ["median", "infogain"]
    tasks = expand_matrix(matrix_config)
    if mode == "pilot":
        tasks = [task for task in tasks if resolve_dataset(task["dataset"]).key == "income"]
    if not tasks:
        raise ValueError("No Income tasks were found for the pilot.")
    experiment_dir = create_experiment_dir(config["results_root"], "exp1_kanon", run_label=args.run_label)
    manifest = {
        "schema_version": "kanon_sweep.v1", "status": "running", "mode": mode,
        "config_path": str(config_path), "matrix_path": str(matrix_path),
        "configuration": config, "matrix_configuration": matrix_config,
        "implementation_sha256": code_hash, "variants": variants,
        "selection": selection, "tasks": tasks, "income_protocol_signatures": [],
        "tail_risk_population": "training_split",
        "original_privacy_population": "training_split",
        "release_storage": "shared_compact_json_gzip_with_per_model_references",
        "evaluation_routing": EVALUATION_ROUTING,
        "trim_evaluation_difference": "KAnon widens failing leaf QIs individually; TRIM falls back whole records across released snapshots.",
        "routing_review_threshold": routing_review_threshold,
    }
    write_json(experiment_dir / "manifest.json", manifest)
    tree_cache = MondrianReleaseCache(experiment_dir / "shared_releases")
    raw_rows, plot_rows = [], []
    try:
        for task in tasks:
            dataset = resolve_dataset(task["dataset"]).key
            split_config = dict(task.get("pipeline") or {})
            seed = int(split_config.get("random_state", 42))
            family = task["model"]["downstream"]["family"]
            if family not in {"mlp", "xgboost"}:
                raise ValueError("Exp-1 KAnon supports MLP and XGBoost downstream models.")
            device = args.device or task.get("device", "cuda")
            dtype = task.get("dtype", "float32")
            cpu_fallback = bool(task.get("allow_cpu_fallback", True))
            loader = build_data_loader(dataset, data_path=resolve_project_path(task.get("data_path")))
            split = load_dataset_split(loader, split_config)
            original_X, original_y = loader.X_raw, loader.y
            classes = np.unique(split.y_train.to_numpy())
            if len(classes) < 2:
                raise ValueError("The downstream classification reference needs at least two training classes.")
            tree_path = resolve_generalization_tree(
                dataset, resolve_project_path(task.get("generalization_tree_path")),
            )
            tree_hash = hashlib.sha256(tree_path.read_bytes()).hexdigest()
            generalization = load_generalization_rules_from_file(tree_path, loader, generalization_level=0)
            schema = HierarchySchema(generalization)
            input_hash = _loaded_input_sha256(original_X, original_y)
            training_hash = _loaded_input_sha256(split.X_train_raw, split.y_train)
            model_spec = dict(task["model"]["downstream"])
            if model_spec.get("warm_start"):
                raise ValueError("Every K point must train a fresh model; warm_start must be false.")
            protocol = {
                "loaded_input_sha256": input_hash, "tree_sha256": tree_hash,
                "model": model_spec, "device": device, "dtype": dtype,
                "allow_cpu_fallback": cpu_fallback,
                "split": {key: split_config.get(key, default) for key, default in (
                    ("nrows", None), ("val_size", 0.15), ("test_size", 0.15),
                )},
                "encoding": "shared_leaf_space_with_level_or_shared_mlp_means",
                "evaluation_routing": EVALUATION_ROUTING,
                "privacy_population": "training_split",
                "original_privacy_encoding": "raw_qis",
            }
            signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
            if dataset == "income":
                if mode == "full" and signature not in selection["income_protocol_signatures"]:
                    raise ValueError("Income data/tree/model/split settings differ from the selection pilot.")
                manifest["income_protocol_signatures"].append(signature)
            task_dir = experiment_dir / task["run_id"]
            task_dir.mkdir()
            task_manifest = {
                "task": task, "protocol": protocol, "protocol_sha256": signature,
                "seed": seed, "preprocessing": getattr(loader, "preprocessing_metadata", {}),
                "qi_attributes": list(loader.qi_attributes),
                "train_row_ids": split.X_train_raw.index.tolist(),
                "validation_row_ids": split.X_val_raw.index.tolist(),
                "test_row_ids": split.X_test_raw.index.tolist(),
            }
            write_json(task_dir / "config.json", task_manifest)
            original_sizes = original_equivalence_class_sizes(split.X_train_raw, loader.qi_attributes)
            original_k = int(original_sizes.min())
            original_tail = individual_tail_risk_stats(split.X_train_raw.index, original_sizes)["tail_risk_p99"]
            level0_train = generalization.encode(split.X_train_raw)
            standardizer = fit_standardizer(level0_train, loader)
            # Match TRIM's tensor dtype before standardization for MLP.
            def model_input(frame, groups=None):
                encoded = (
                    schema.encode(frame, groups, family=family) if groups is not None
                    else generalization.encode_xgboost_leaf_space(frame).toarray() if family == "xgboost"
                    else generalization.encode(frame)
                )
                if family == "xgboost":
                    return np.ascontiguousarray(encoded, dtype=np.float32)
                return standardizer.transform(to_device_tensor(
                    encoded, device=device, dtype=dtype, allow_cpu_fallback=cpu_fallback,
                ))

            factory = build_model_factory(
                model_spec, device=device, dtype=dtype, allow_cpu_fallback=cpu_fallback,
                random_state=seed, role="downstream",
            )
            def losses(train_input, validation_input, test_input=None):
                model = factory()
                model.set_classes(classes)
                model.fit(train_input, split.y_train)
                output = {}
                for name, labels, encoded in [
                    ("validation", split.y_val, validation_input),
                    ("test", split.y_test, test_input),
                ]:
                    if encoded is None:
                        continue
                    probabilities = model.predict_proba_tensor(encoded)
                    target = to_device_tensor(labels, device=probabilities.device, dtype=probabilities.dtype)
                    output[name] = float(classification_log_loss_tensor(
                        target, probabilities, classes=classes, n_classes=len(classes),
                    ).detach().cpu())
                    if not np.isfinite(output[name]):
                        raise ValueError(f"{name} loss is nonfinite; check model inputs and training.")
                return output

            reference = losses(
                model_input(split.X_train_raw), model_input(split.X_val_raw),
                model_input(split.X_test_raw) if mode == "full" else None,
            )
            write_json(task_dir / "reference.json", {
                **reference, "original_leak_k": original_k, "original_tail_risk_p99": original_tail,
                "train_size": len(split.X_train_raw), "loaded_size": len(original_X),
                "privacy_population": "training_split", "privacy_population_size": len(split.X_train_raw),
                "standardizer_mean": getattr(standardizer, "mean", np.array([])).tolist(),
                "standardizer_std": getattr(standardizer, "std", np.array([])).tolist(),
            })
            sweep = config.get("sweep") or {}
            ks = k_grid(len(split.X_train_raw), extras=sweep.get("extra_k", [12, 235]), explicit=args.k or sweep.get("k_values"))
            for variant in variants:
                for k in ks:
                    fitted, release_reference = tree_cache.get_or_fit(
                        schema, split.X_train_raw, split.y_train, identity={
                            "dataset": dataset, "seed": seed, "variant": variant, "k": k,
                            "training_data_sha256": training_hash, "tree_sha256": tree_hash,
                            "implementation_sha256": code_hash,
                        },
                    )
                    point_dir = task_dir / variant / f"k_{k}"
                    write_json(point_dir / "release.json", {
                        **release_reference, "protocol_sha256": signature,
                        "shared_release_path": os.path.relpath(release_reference["shared_release_path"], point_dir),
                        "tree_sha256": tree_hash, "loaded_input_sha256": input_hash,
                    })
                    sizes = fitted.per_record_k(split.X_train_raw.index)
                    tail = individual_tail_risk_stats(split.X_train_raw.index, sizes)
                    val_groups = fitted.transform_groups(split.X_val_raw)
                    test_groups = fitted.transform_groups(split.X_test_raw) if mode == "full" else None
                    val_routing = fitted.routing_stats(
                        val_groups, len(split.X_val_raw), review_threshold=routing_review_threshold,
                    )
                    test_routing = fitted.routing_stats(
                        test_groups, len(split.X_test_raw), review_threshold=routing_review_threshold,
                    ) if test_groups is not None else None
                    write_json(point_dir / "routing_diagnostics.json", {
                        "policy": EVALUATION_ROUTING,
                        "validation": val_routing, "test": test_routing,
                    })
                    started = time.perf_counter()
                    outcome = losses(
                        model_input(split.X_train_raw, fitted.partitions),
                        model_input(split.X_val_raw, val_groups),
                        model_input(split.X_test_raw, test_groups) if test_groups is not None else None,
                    )
                    row = {
                        "experiment": "exp1_privacy_utility", "method": "KAnon",
                        "dataset": dataset, "model": task.get("model_name", family),
                        "seed": seed, "point": k, "target_k": k, "variant": variant,
                        "evaluation_split": "validation" if mode == "pilot" else "test",
                        "validation_loss": outcome["validation"],
                        "validation_delta_u": outcome["validation"] - reference["validation"],
                        "baseline_validation_loss": reference["validation"],
                        "test_loss": outcome.get("test"), "baseline_test_loss": reference.get("test"),
                        "test_delta_u": outcome["test"] - reference["test"] if mode == "full" else None,
                        "min_k": int(sizes.min()), "original_leak_k": original_k,
                        "privacy_delta_h": -float(np.log(sizes.min() / original_k)),
                        **tail, "original_tail_risk_p99": original_tail,
                        "row_count": len(sizes), "coverage_fraction": 1.0,
                        "suppressed_row_count": 0, "partition_count": len(fitted.partitions),
                        **{
                            f"{split_name}_{field}": routing[field] if routing is not None else None
                            for split_name, routing in [("validation", val_routing), ("test", test_routing)]
                            for field in (
                                "leaf_count", "internal_count", "root_count", "fallback_count",
                                "fallback_fraction", "review_needed",
                                "leaf_unmodified_count", "leaf_widened_count", "widened_record_count",
                                "widened_record_fraction", "widened_qi_count", "root_widened_record_count",
                                "whole_record_ancestor_fallback_count", "adjusted_record_count", "adjusted_record_fraction",
                            )
                        },
                        "routing_review_threshold": routing_review_threshold,
                        "evaluation_routing": EVALUATION_ROUTING,
                        "partition_seconds": release_reference["partition_seconds"],
                        "tree_cache_hit": release_reference["cache_hit"],
                        "tree_cache_seconds": release_reference["tree_cache_seconds"],
                        "original_partition_seconds": release_reference["original_partition_seconds"],
                        "shared_release_bytes": release_reference["shared_release_bytes"],
                        "shared_release_path": release_reference["shared_release_path"],
                        "model_evaluation_seconds": time.perf_counter() - started,
                        "protocol_sha256": signature, "release_path": str(point_dir / "release.json"),
                    }
                    write_json(point_dir / "metrics.json", row)
                    append_jsonl(experiment_dir / "raw_results.jsonl", row)
                    raw_rows.append(row)
                    if mode == "full":
                        for metric, value, reference_value in [
                            ("min_k", row["min_k"], original_k),
                            ("tail_risk_p99", tail["tail_risk_p99"], original_tail),
                        ]:
                            plot_rows.append({
                                **row, "x_metric": "test_delta_u", "x_value": row["test_delta_u"],
                                "y_metric": metric, "y_value": value, "y_reference": reference_value,
                            })
                    print(f'{task["run_id"]} {variant} K={k}: validation_delta={row["validation_delta_u"]:.6g}', flush=True)
                    for split_name, routing in [("validation", val_routing), ("test", test_routing)]:
                        if routing is not None and routing["review_needed"]:
                            print(
                                f'Routing review needed: {split_name} whole-record fallback={routing["fallback_fraction"]:.2%}, '
                                f'per-QI widening={routing["widened_record_fraction"]:.2%} '
                                f'(internal={routing["internal_count"]}, root={routing["root_count"]}) '
                                f'for {task["run_id"]} {variant} K={k}.',
                                flush=True,
                            )
            # Save incrementally so completed tasks remain reviewable after interruption.
            write_csv_rows(experiment_dir / "raw_results.csv", raw_rows)
            if plot_rows:
                write_csv_rows(experiment_dir / "baseline_long.csv", plot_rows)
        manifest["status"] = "complete"
        manifest["income_protocol_signatures"] = sorted(set(manifest["income_protocol_signatures"]))
        manifest["raw_point_count"] = len(raw_rows)
        write_json(experiment_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update({"status": "failed", "error": str(exc)})
        write_json(experiment_dir / "manifest.json", manifest)
        raise
    print(experiment_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
