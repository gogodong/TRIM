"""Attacks evaluated directly on a materialized TRIM release."""

from __future__ import annotations

import random
from typing import Any, Mapping

import numpy as np
import pandas as pd

from prototype.release_artifacts import MaterializedTRIMRelease

from .attack_models import build_attack_model


MISSING_VALUE = "__MISSING__"
DOWNSTREAM_LABEL_COLUMN = "__downstream_label__"


def _value_key(value: Any) -> Any:
    value = value.item() if hasattr(value, "item") else value
    if pd.isna(value):
        return MISSING_VALUE
    return value


def _categorical_target(series: pd.Series) -> pd.Series:
    return pd.Series(
        [_value_key(value) for value in series.tolist()],
        index=series.index,
        dtype="object",
    )


def _sorted_classes(series: pd.Series) -> list[Any]:
    return sorted(
        pd.unique(series).tolist(),
        key=lambda value: (type(value).__name__, str(value)),
    )


def _json_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _fold(row_ids, *, fold_count: int, fold_index: int, random_state: int) -> list[Any]:
    fold_count = int(fold_count)
    fold_index = int(fold_index)
    if fold_count < 1 or not 0 <= fold_index < fold_count:
        raise ValueError("fold_count/fold_index do not identify a valid fold.")
    shuffled = list(row_ids)
    random.Random(int(random_state)).shuffle(shuffled)
    fold_sizes = [len(shuffled) // fold_count] * fold_count
    for index in range(len(shuffled) % fold_count):
        fold_sizes[index] += 1
    start = sum(fold_sizes[:fold_index])
    selected = shuffled[start : start + fold_sizes[fold_index]]
    if not selected:
        raise ValueError("The configured attacker training fold is empty.")
    return selected


def _accuracy(y_true, prediction) -> float | None:
    true_values = np.asarray(list(y_true), dtype=object)
    predicted_values = np.asarray(list(prediction), dtype=object)
    if true_values.size == 0:
        return None
    if true_values.shape != predicted_values.shape:
        raise ValueError("Prediction length does not match the target length.")
    return float(np.mean(true_values == predicted_values))


def _numeric_accuracy(y_true, prediction, tolerance: float) -> float | None:
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(prediction, dtype=float)
    if true_values.size == 0:
        return None
    if true_values.shape != predicted_values.shape:
        raise ValueError("Prediction length does not match the target length.")
    finite = np.isfinite(true_values) & np.isfinite(predicted_values)
    hits = np.zeros(true_values.shape, dtype=bool)
    hits[finite] = np.abs(true_values[finite] - predicted_values[finite]) <= tolerance
    return float(np.mean(hits))


def _mean(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _input_frames(
    materialized: MaterializedTRIMRelease,
    *,
    include_label: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_X = materialized.published_frame.copy()
    test_X = materialized.evaluation_frame.copy()
    if list(train_X.columns) != list(test_X.columns):
        raise ValueError("Published and evaluation release columns disagree.")
    if not train_X.index.is_unique or not test_X.index.is_unique:
        raise ValueError("Attack inputs require unique published/evaluation row IDs.")
    if set(train_X.index).intersection(test_X.index):
        raise ValueError("Published training rows and held-out evaluation rows overlap.")
    if not train_X.index.equals(materialized.published_raw.index):
        raise ValueError("Published encoded/raw row order disagrees.")
    if not test_X.index.equals(materialized.evaluation_raw.index):
        raise ValueError("Evaluation encoded/raw row order disagrees.")
    if include_label:
        train_X[DOWNSTREAM_LABEL_COLUMN] = pd.Series(
            np.asarray(materialized.published_y), index=train_X.index
        )
        test_X[DOWNSTREAM_LABEL_COLUMN] = pd.Series(
            np.asarray(materialized.evaluation_y), index=test_X.index
        )
    return train_X, test_X


def run_reconstruction_attack(
    materialized: MaterializedTRIMRelease,
    data_loader,
    config: Mapping[str, Any],
    *,
    random_state: int,
    device: str,
    dtype: str,
    allow_cpu_fallback: bool,
) -> dict[str, Any]:
    """Infer each original attribute from its released mixed-level encoding."""
    attack_config = dict(config)
    required_options = (
        "fold_count",
        "fold_index",
        "include_label",
        "numeric_tolerance",
        "model",
    )
    missing_options = [name for name in required_options if name not in attack_config]
    if missing_options:
        raise ValueError(
            "Reconstruction config is missing required options: "
            f"{missing_options}"
        )
    fold_count = int(attack_config.pop("fold_count"))
    fold_index = int(attack_config.pop("fold_index"))
    include_label = bool(attack_config.pop("include_label"))
    numeric_tolerance = float(attack_config.pop("numeric_tolerance"))
    attributes = attack_config.pop("attributes", None)
    model_spec = attack_config.pop("model")
    if attack_config:
        raise ValueError(f"Unknown reconstruction options: {sorted(attack_config)}")
    if numeric_tolerance < 0.0:
        raise ValueError("numeric_tolerance must be non-negative.")
    if not isinstance(model_spec, Mapping):
        raise ValueError("reconstruction.model must be a YAML mapping.")
    configured_model_spec = dict(model_spec)

    train_input, test_input = _input_frames(
        materialized,
        include_label=include_label,
    )
    knowledge_ids = _fold(
        train_input.index,
        fold_count=fold_count,
        fold_index=fold_index,
        random_state=random_state,
    )
    attributes = list(
        attributes
        or getattr(
            data_loader,
            "qi_attributes",
            getattr(data_loader, "feature_columns", ()),
        )
    )
    if not attributes:
        raise ValueError("At least one reconstruction attribute is required.")
    if len(attributes) != len(set(attributes)):
        raise ValueError("Reconstruction attributes must not contain duplicates.")
    missing = [name for name in attributes if name not in materialized.published_raw.columns]
    if missing:
        raise KeyError(f"Reconstruction attributes are absent from raw data: {missing}")
    numeric_attributes = set(getattr(data_loader, "numeric_attributes", ()))

    results = []
    for offset, attribute in enumerate(attributes):
        is_numeric = attribute in numeric_attributes
        target_type = "numeric" if is_numeric else "categorical"
        if is_numeric:
            train_target = pd.to_numeric(
                materialized.published_raw.loc[knowledge_ids, attribute],
                errors="coerce",
            )
            test_target = pd.to_numeric(
                materialized.evaluation_raw.loc[test_input.index, attribute],
                errors="coerce",
            )
            train_valid = train_target.notna()
            test_valid = test_target.notna()
            classes = None
        else:
            train_target = _categorical_target(
                materialized.published_raw.loc[knowledge_ids, attribute]
            )
            test_target = _categorical_target(
                materialized.evaluation_raw.loc[test_input.index, attribute]
            )
            train_valid = pd.Series(True, index=train_target.index)
            test_valid = pd.Series(True, index=test_target.index)
            classes = _sorted_classes(train_target)

        if int(train_valid.sum()) == 0 or int(test_valid.sum()) == 0:
            results.append(
                {
                    "attribute": attribute,
                    "target_type": target_type,
                    "numeric_tolerance": numeric_tolerance if is_numeric else None,
                    "train_row_count": int(train_valid.sum()),
                    "test_row_count": int(test_valid.sum()),
                    "class_count": None if is_numeric else len(classes),
                    "classes": (
                        None
                        if is_numeric
                        else [_json_scalar(value) for value in classes]
                    ),
                    "train_accuracy": None,
                    "test_accuracy": None,
                    "baseline_test_accuracy": None,
                    "train_reconstruction_error": None,
                    "test_reconstruction_error": None,
                    "baseline_test_reconstruction_error": None,
                    "skipped": True,
                    "skip_reason": "no finite training or evaluation targets",
                }
            )
            continue
        if not is_numeric and len(classes) < 2:
            results.append(
                {
                    "attribute": attribute,
                    "target_type": target_type,
                    "numeric_tolerance": None,
                    "train_row_count": int(train_valid.sum()),
                    "test_row_count": int(test_valid.sum()),
                    "class_count": len(classes),
                    "classes": [_json_scalar(value) for value in classes],
                    "train_accuracy": None,
                    "test_accuracy": None,
                    "baseline_test_accuracy": None,
                    "train_reconstruction_error": None,
                    "test_reconstruction_error": None,
                    "baseline_test_reconstruction_error": None,
                    "skipped": True,
                    "skip_reason": "attacker training fold has fewer than two classes",
                }
            )
            continue

        model = build_attack_model(
            target_type,
            classes=classes,
            spec=model_spec,
            device=device,
            dtype=dtype,
            allow_cpu_fallback=allow_cpu_fallback,
            random_state=int(random_state) + offset,
        )
        train_ids = train_target.index[train_valid]
        test_ids = test_target.index[test_valid]
        model.fit(train_input.loc[train_ids], train_target.loc[train_ids])
        train_prediction = model.predict(train_input.loc[train_ids])
        test_prediction = model.predict(test_input.loc[test_ids])
        if is_numeric:
            baseline_prediction = float(train_target.loc[train_ids].mean())
            train_accuracy = _numeric_accuracy(
                train_target.loc[train_ids], train_prediction, numeric_tolerance
            )
            test_accuracy = _numeric_accuracy(
                test_target.loc[test_ids], test_prediction, numeric_tolerance
            )
            baseline_accuracy = _numeric_accuracy(
                test_target.loc[test_ids],
                np.repeat(baseline_prediction, len(test_ids)),
                numeric_tolerance,
            )
        else:
            baseline_prediction = train_target.loc[train_ids].mode().iloc[0]
            train_accuracy = _accuracy(train_target.loc[train_ids], train_prediction)
            test_accuracy = _accuracy(test_target.loc[test_ids], test_prediction)
            baseline_accuracy = _accuracy(
                test_target.loc[test_ids],
                np.repeat(baseline_prediction, len(test_ids)),
            )
        results.append(
            {
                "attribute": attribute,
                "target_type": target_type,
                "numeric_tolerance": numeric_tolerance if is_numeric else None,
                "train_row_count": len(train_ids),
                "test_row_count": len(test_ids),
                "class_count": None if is_numeric else len(classes),
                "classes": (
                    None
                    if is_numeric
                    else [_json_scalar(value) for value in classes]
                ),
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "baseline_test_accuracy": baseline_accuracy,
                "train_reconstruction_error": 1.0 - train_accuracy,
                "test_reconstruction_error": 1.0 - test_accuracy,
                "baseline_test_reconstruction_error": 1.0 - baseline_accuracy,
                "skipped": False,
                "skip_reason": None,
            }
        )

    evaluated = [row for row in results if not row.get("skipped")]
    return {
        "attack": "attribute_reconstruction",
        "input_representation": "persisted mixed-level TRIM encoding",
        "training_row_semantics": (
            "one deterministic fold of assigned published training rows; "
            "raw attribute values are attacker knowledge"
        ),
        "evaluation_row_semantics": (
            f"routed held-out {materialized.evaluation_split} rows with raw targets"
        ),
        "evaluation_split": materialized.evaluation_split,
        "fold_count": fold_count,
        "fold_index": fold_index,
        "include_label": include_label,
        "numeric_tolerance": numeric_tolerance,
        "model_spec": configured_model_spec,
        "attacker_training_row_count": len(knowledge_ids),
        "evaluation_row_count": len(test_input),
        "attribute_count": len(results),
        "evaluated_attribute_count": len(evaluated),
        "skipped_attribute_count": len(results) - len(evaluated),
        "train_reconstruction_error_mean": _mean(
            row["train_reconstruction_error"] for row in evaluated
        ),
        "test_reconstruction_error_mean": _mean(
            row["test_reconstruction_error"] for row in evaluated
        ),
        "baseline_test_reconstruction_error_mean": _mean(
            row["baseline_test_reconstruction_error"] for row in evaluated
        ),
        "attribute_results": results,
    }


def _columns_for_attributes(frame: pd.DataFrame, attributes: list[str]) -> list[str]:
    columns = []
    for attribute in attributes:
        matches = [
            column
            for column in frame.columns
            if column == attribute
            or str(column).startswith(f"{attribute}=")
            or str(column).startswith(f"{attribute}_cat=")
            or str(column).startswith(f"{attribute}__")
        ]
        if not matches:
            raise KeyError(f"No released columns encode linkage attribute {attribute!r}.")
        columns.extend(matches)
    if len(columns) != len(set(columns)):
        raise ValueError("Linkage A attributes resolve to overlapping released columns.")
    return columns


def run_strong_linkage_attack(
    materialized: MaterializedTRIMRelease,
    config: Mapping[str, Any],
    *,
    random_state: int,
    device: str,
    dtype: str,
    allow_cpu_fallback: bool,
) -> dict[str, Any]:
    """Predict the raw B attribute from released A attributes with an MLP."""
    attack_config = dict(config)
    required_options = (
        "strategy",
        "query_size",
        "a_attributes",
        "b_attributes",
        "model",
    )
    missing_options = [name for name in required_options if name not in attack_config]
    if missing_options:
        raise ValueError(
            f"Linkage config is missing required options: {missing_options}"
        )
    strategy = str(attack_config.pop("strategy"))
    if strategy != "mlp_raw_b":
        raise ValueError("Only linkage strategy='mlp_raw_b' is included.")
    query_size = int(attack_config.pop("query_size"))
    a_attributes = list(attack_config.pop("a_attributes"))
    b_attributes = list(attack_config.pop("b_attributes"))
    model_spec = attack_config.pop("model")
    if attack_config:
        raise ValueError(f"Unknown linkage options: {sorted(attack_config)}")
    if query_size <= 0:
        raise ValueError("linkage.query_size must be positive.")
    if not isinstance(model_spec, Mapping):
        raise ValueError("linkage.model must be a YAML mapping.")
    configured_model_spec = dict(model_spec)
    if not a_attributes:
        raise ValueError("linkage.a_attributes must be non-empty.")
    if len(b_attributes) != 1:
        raise ValueError("Strong linkage requires exactly one B attribute.")
    if len(set(a_attributes)) != len(a_attributes):
        raise ValueError("linkage.a_attributes must not contain duplicates.")
    if b_attributes[0] in set(a_attributes):
        raise ValueError("Strong linkage A and B attributes must be disjoint.")

    b_attribute = b_attributes[0]
    if b_attribute not in materialized.published_raw.columns:
        raise KeyError(f"B attribute {b_attribute!r} is absent from raw data.")
    a_columns = _columns_for_attributes(materialized.published_frame, a_attributes)
    train_X = materialized.published_frame.loc[:, a_columns]
    test_X = materialized.evaluation_frame.loc[:, a_columns]
    if set(train_X.index).intersection(test_X.index):
        raise ValueError("Linkage training and held-out evaluation rows overlap.")
    train_target = _categorical_target(
        materialized.published_raw.loc[train_X.index, b_attribute]
    )
    test_target = _categorical_target(
        materialized.evaluation_raw.loc[test_X.index, b_attribute]
    )
    classes = _sorted_classes(train_target)
    query_ids = list(test_X.index)
    random.Random(int(random_state)).shuffle(query_ids)
    query_ids = query_ids[: min(query_size, len(query_ids))]
    if len(classes) < 2:
        return {
            "strategy": strategy,
            "model_spec": configured_model_spec,
            "skipped": True,
            "skip_reason": "published release has fewer than two B classes",
            "a_attributes": a_attributes,
            "b_attributes": b_attributes,
            "a_columns": a_columns,
            "query_size": query_size,
            "query_count": len(query_ids),
            "training_row_count": len(train_X),
            "evaluation_row_count": len(test_X),
            "class_count": len(classes),
            "classes": [_json_scalar(value) for value in classes],
            "test_accuracy": None,
            "baseline_test_accuracy": None,
            "test_linkage_error": None,
            "baseline_test_linkage_error": None,
            "test_leakage_gap": None,
            "training_row_semantics": (
                "all assigned published training rows with raw B knowledge"
            ),
            "evaluation_row_semantics": (
                f"routed held-out {materialized.evaluation_split} query rows"
            ),
            "evaluation_split": materialized.evaluation_split,
        }
    if not query_ids:
        raise ValueError("The routed linkage evaluation split is empty.")
    model = build_attack_model(
        "categorical",
        classes=classes,
        spec=model_spec,
        device=device,
        dtype=dtype,
        allow_cpu_fallback=allow_cpu_fallback,
        random_state=random_state,
    )
    model.fit(train_X, train_target)
    prediction = model.predict(test_X.loc[query_ids])
    baseline_prediction = train_target.mode().iloc[0]
    accuracy = _accuracy(test_target.loc[query_ids], prediction)
    baseline_accuracy = _accuracy(
        test_target.loc[query_ids],
        np.repeat(baseline_prediction, len(query_ids)),
    )
    return {
        "strategy": strategy,
        "model_spec": configured_model_spec,
        "a_attributes": a_attributes,
        "b_attributes": b_attributes,
        "a_columns": a_columns,
        "query_size": query_size,
        "query_count": len(query_ids),
        "training_row_count": len(train_X),
        "evaluation_row_count": len(test_X),
        "class_count": len(classes),
        "classes": [_json_scalar(value) for value in classes],
        "training_row_semantics": (
            "all assigned published training rows with raw B knowledge"
        ),
        "evaluation_row_semantics": (
            f"routed held-out {materialized.evaluation_split} query rows"
        ),
        "evaluation_split": materialized.evaluation_split,
        "test_accuracy": accuracy,
        "baseline_test_accuracy": baseline_accuracy,
        "test_linkage_error": 1.0 - accuracy,
        "baseline_test_linkage_error": 1.0 - baseline_accuracy,
        "test_leakage_gap": accuracy - baseline_accuracy,
        "skipped": False,
        "skip_reason": None,
    }
