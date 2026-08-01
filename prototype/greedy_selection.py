"""Shared state and logging helpers used by TRIM."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import yaml


def calculate_leakage(original_encode):
    """Return the minimum equivalence-class size of a non-empty encoding."""
    if original_encode.empty:
        raise ValueError("original_encode must contain at least one record.")
    return original_encode.value_counts(sort=False).min()


@dataclass
class SelectionState:
    predicted_model_loss: float | None = None
    actual_model_loss: float | None = None
    is_actual_model_loss_trusted: bool = False
    current_generalization: Any = None
    selected_row_ids: list[Any] = field(default_factory=list)
    current_train_encode: Any = None
    current_train_tensor: Any = None
    action_next_states_by_iteration: list[Any] = field(default_factory=list)
    action_log_path: str | None = None


def _write_action_log(
    *,
    action_log_path,
    iteration,
    current_state,
    candidates,
    best_candidate,
):
    if action_log_path is None:
        return

    current_generalization_level = (
        dict(current_state.current_generalization.generalization_level)
        if current_state.current_generalization is not None
        else None
    )

    def to_json_scalar(value):
        if hasattr(value, "item"):
            return value.item()
        return value

    def to_json_key(value):
        if isinstance(value, tuple):
            return [to_json_scalar(item) for item in value]
        return to_json_scalar(value)

    with Path(action_log_path).open("a", encoding="utf-8") as log_file:
        for candidate in candidates:
            next_state = candidate["state"]
            next_generalization_level = (
                dict(next_state.current_generalization.generalization_level)
                if next_state.current_generalization is not None
                else None
            )
            candidate_id = (
                candidate.get("retention_class_key")
                if candidate["action"] == "retention_class"
                else candidate.get("attribute")
            )
            record = {
                "event": "candidate_evaluation",
                "schema_version": "trim_candidate_evaluation.v1",
                "iteration": iteration,
                "action": candidate["action"],
                "candidate_type": candidate["action"],
                "candidate_id": to_json_key(candidate_id),
                "selected": candidate is best_candidate,
                "retention_class_key": to_json_key(
                    candidate.get("retention_class_key")
                ),
                "attribute": candidate.get("attribute"),
                "lga_score": to_json_scalar(candidate.get("lga_score")),
                "rank_method": candidate.get("rank_method"),
                "rank_score": to_json_scalar(candidate.get("rank_score")),
                "rank_position": candidate.get("rank_position"),
                "in_rank_top_k": candidate.get("in_rank_top_k"),
                "exact_rank_score": to_json_scalar(
                    candidate.get("exact_rank_score")
                ),
                "exact_rank_position": candidate.get("exact_rank_position"),
                "in_exact_top_k": candidate.get("in_exact_top_k"),
                "current_k": to_json_scalar(candidate["current_k"]),
                "next_k": to_json_scalar(candidate["next_k"]),
                "utility_gain": to_json_scalar(candidate.get("utility_gain")),
                "privacy_cost": to_json_scalar(candidate.get("privacy_cost")),
                "score": to_json_scalar(candidate["score"]),
                "candidate_state_score": to_json_scalar(
                    candidate.get("candidate_state_score")
                ),
                "candidate_published_min_k": to_json_scalar(
                    candidate.get("candidate_published_min_k")
                ),
                "release_target_k": to_json_scalar(
                    candidate.get("release_target_k")
                ),
                "release_validation_loss": to_json_scalar(
                    candidate.get("release_validation_loss")
                ),
                "release_row_count": to_json_scalar(
                    candidate.get("release_row_count")
                ),
                "release_suppressed_row_count": to_json_scalar(
                    candidate.get("release_suppressed_row_count")
                ),
                "release_coverage_fraction": to_json_scalar(
                    candidate.get("release_coverage_fraction")
                ),
                "current_predicted_model_loss": to_json_scalar(
                    current_state.predicted_model_loss
                ),
                "next_predicted_model_loss": to_json_scalar(
                    next_state.predicted_model_loss
                ),
                "current_selected_row_count": len(current_state.selected_row_ids),
                "next_selected_row_count": len(next_state.selected_row_ids),
                "current_generalization_level": current_generalization_level,
                "next_generalization_level": next_generalization_level,
                "current_train_encode_rows": len(current_state.current_train_encode),
                "next_train_encode_rows": len(next_state.current_train_encode),
            }
            log_file.write(json.dumps(record, default=str, sort_keys=True))
            log_file.write("\n")


def _load_max_generalization_level(generalization_tree_path, attributes):
    data = yaml.safe_load(
        Path(generalization_tree_path).read_text(encoding="utf-8-sig")
    )
    trees = data.get("trees", {})
    return {
        attribute: max(
            (
                int(node.get("height_from_leaf", 0))
                for node in trees.get(attribute, {}).get("nodes", [])
            ),
            default=0,
        )
        for attribute in attributes
    }
