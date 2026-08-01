"""Shared utilities for paper experiment runners."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping
from uuid import uuid4

import yaml

from prototype.TRIM_prototype_pipeline import run_trim_pipeline
from prototype.dataset_registry import (
    PROJECT_ROOT,
    build_data_loader,
    resolve_generalization_tree,
)
from prototype.model_factory import build_model_factory


_CORE_RESERVED_OPTIONS = {
    "data_loader",
    "generalization_tree_path",
    "model_config",
    "model_factory",
    "proxy_model_factory",
    "estimator_model_factory",
    "results_dir",
    "device",
    "dtype",
    "task_type",
    "learning_rate",
    "initial_sample_fraction",
}


def load_yaml_mapping(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Read a YAML mapping and return its absolute path and contents."""
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a YAML mapping: {config_path}")
    return config_path, config


def resolve_project_path(value: str | Path | None) -> Path | None:
    """Resolve a user path relative to the project root."""
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def merge_mappings(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recursively merge experiment defaults with one task override."""
    merged = deepcopy(dict(base or {}))
    for key, value in dict(override or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_mappings(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def create_experiment_dir(
    results_root: str | Path,
    experiment_id: str,
    *,
    run_label: str | None = None,
) -> Path:
    """Create a unique experiment directory without overwriting prior results."""
    root = resolve_project_path(results_root)
    if root is None:
        raise ValueError("results_root is required for TRIM experiments.")
    safe_experiment = re.sub(r"[^A-Za-z0-9_-]+", "_", experiment_id).strip("_")
    if not safe_experiment:
        raise ValueError("experiment_id must contain a letter or number.")
    safe_label = ""
    if run_label:
        normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", str(run_label)).strip("_")
        safe_label = f"_{normalized}" if normalized else ""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment_dir = root / f"{timestamp}_{uuid4().hex[:8]}_{safe_experiment}{safe_label}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, default=str))
        handle.write("\n")


def write_csv_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError(f"Cannot write an empty experiment summary: {path}")
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def run_core_task(
    task: Mapping[str, Any],
    *,
    experiment_dir: str | Path,
    device: str | None = None,
) -> dict[str, Any]:
    """Run one declared task through the shared TRIM implementation."""
    task = deepcopy(dict(task))
    run_id = str(task.get("run_id", "")).strip()
    if not run_id:
        raise ValueError("Every experiment task must define a non-empty run_id.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError(
            f"run_id must use only letters, digits, '_' and '-': {run_id!r}"
        )
    dataset = str(task.get("dataset", "")).strip()
    if not dataset:
        raise ValueError(f"Experiment task {run_id!r} must define dataset.")

    data_loader = build_data_loader(
        dataset,
        data_path=resolve_project_path(task.get("data_path")),
    )
    tree_path = resolve_generalization_tree(
        dataset,
        tree_path=resolve_project_path(task.get("generalization_tree_path")),
    )

    configured_device = device or task.get("device", "cuda")
    dtype = task.get("dtype", "float32")
    allow_cpu_fallback = bool(task.get("allow_cpu_fallback", True))
    pipeline_options = deepcopy(dict(task.get("pipeline") or {}))
    forbidden = sorted(_CORE_RESERVED_OPTIONS.intersection(pipeline_options))
    if forbidden:
        raise ValueError(
            f"Task {run_id!r} puts reserved keys under pipeline: {forbidden}"
        )
    initial_sample_fraction = task.get("initial_sample_fraction")
    if initial_sample_fraction is not None:
        fraction = float(initial_sample_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"Task {run_id!r} initial_sample_fraction must be in (0, 1]."
            )
        if pipeline_options.get("initial_sample_size") is not None:
            raise ValueError(
                f"Task {run_id!r} cannot set both initial_sample_fraction "
                "and pipeline.initial_sample_size."
            )
        pipeline_options["initial_sample_fraction"] = fraction

    configured_model = task.get("model") or {}
    if not isinstance(configured_model, Mapping):
        raise ValueError(f"Task {run_id!r} model must be a YAML mapping.")
    model_config = deepcopy(dict(configured_model))
    unknown_model_roles = sorted(
        set(model_config).difference({"downstream", "proxy", "estimator"})
    )
    if unknown_model_roles:
        raise ValueError(
            f"Task {run_id!r} has unknown model roles: {unknown_model_roles}"
        )
    random_state = int(pipeline_options.get("random_state", 42))
    factories = {}
    for role in ("downstream", "proxy", "estimator"):
        factories[role] = build_model_factory(
            model_config.get(role),
            device=configured_device,
            dtype=dtype,
            allow_cpu_fallback=allow_cpu_fallback,
            random_state=random_state,
            role=role,
        )

    run_results_root = Path(experiment_dir) / "runs" / run_id
    run_results_root.mkdir(parents=True, exist_ok=False)
    if str(configured_device).startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    wrapper_started_at = time.perf_counter()
    result = run_trim_pipeline(
        data_loader,
        generalization_tree_path=tree_path,
        model_config=model_config,
        model_factory=factories["downstream"],
        proxy_model_factory=factories["proxy"],
        estimator_model_factory=factories["estimator"],
        results_dir=run_results_root,
        run_tag=run_id,
        device=configured_device,
        dtype=dtype,
        **pipeline_options,
    )
    if str(configured_device).startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    wrapper_wall_time = time.perf_counter() - wrapper_started_at
    return {
        "run_id": run_id,
        "dataset": dataset,
        "model_name": task.get("model_name"),
        "random_state": random_state,
        "tolerance": pipeline_options.get("tolerance"),
        "nrows": pipeline_options.get("nrows"),
        "initial_sample_fraction": initial_sample_fraction,
        "initial_sample_size": len(result.initial_row_ids),
        "rank_top_k": pipeline_options.get("rank_top_k"),
        "termination_condition": result.termination_condition,
        "iteration_count": result.iteration_count,
        "selected_row_count": len(result.selected_row_ids),
        "original_leak_k": result.original_leak_k,
        "original_leak_k_p1": result.original_leak_k_p1,
        "original_leak_k_p2": result.original_leak_k_p2,
        "original_leak_k_p3": result.original_leak_k_p3,
        "original_leak_k_p4": result.original_leak_k_p4,
        "original_leak_k_p5": result.original_leak_k_p5,
        "final_leak_k": result.final_leak_k,
        "final_leak_k_p5": result.leak_k_p5,
        "tail_risk_p99": result.tail_risk_p99,
        "baseline_val_loss": result.baseline_val_loss,
        "loss_threshold_val": result.loss_threshold_val,
        "baseline_test_loss": result.baseline_test_loss,
        "loss_threshold_test": result.loss_threshold_test,
        "final_actual_model_loss": result.final_actual_model_loss,
        "utility_constraint_met": result.utility_constraint_met,
        "validation_utility_constraint_met": (
            result.validation_utility_constraint_met
        ),
        "release_target_k": result.release_target_k,
        "release_suppressed_row_count": len(result.release_suppressed_row_ids),
        "run_dir": result.run_dir,
        "wrapper_wall_time_seconds": wrapper_wall_time,
        **{key: value for key, value in result.timings.items()},
    }
