from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
}


def resolve_dtype(dtype: str | torch.dtype = "float64") -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return DTYPE_BY_NAME[str(dtype).lower()]
    except KeyError as exc:
        raise ValueError("dtype must be float32 or float64.") from exc


def resolve_device(
    device: str | torch.device = "cuda",
    *,
    allow_cpu_fallback: bool = True,
) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        if allow_cpu_fallback:
            return torch.device("cpu")
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return resolved


def to_device_tensor(
    values: Any,
    *,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype = "float64",
    copy: bool = False,
    allow_cpu_fallback: bool = True,
) -> torch.Tensor:
    resolved_device = resolve_device(device, allow_cpu_fallback=allow_cpu_fallback)
    resolved_dtype = resolve_dtype(dtype)
    if isinstance(values, torch.Tensor):
        return values.to(device=resolved_device, dtype=resolved_dtype, copy=copy).contiguous()
    array = np.asarray(values, dtype=np.float64 if resolved_dtype == torch.float64 else np.float32)
    return torch.as_tensor(array, device=resolved_device, dtype=resolved_dtype).contiguous()


def tensor_to_numpy(values: torch.Tensor) -> np.ndarray:
    return values.detach().cpu().numpy()


def _class_indices_tensor(
    y_true: torch.Tensor,
    classes: Any | None = None,
) -> torch.Tensor:
    y_values = y_true.reshape(-1)
    if classes is None:
        return y_values.to(dtype=torch.long)

    classes_array = np.asarray(classes)
    if classes_array.ndim != 1 or classes_array.size == 0:
        raise ValueError("classes must be a non-empty 1D sequence.")
    classes_tensor = torch.as_tensor(classes_array, device=y_values.device)
    if y_values.dtype.is_floating_point:
        classes_tensor = classes_tensor.to(dtype=y_values.dtype)
    elif classes_tensor.dtype != y_values.dtype:
        classes_tensor = classes_tensor.to(dtype=y_values.dtype)
    matches = y_values.unsqueeze(1) == classes_tensor.unsqueeze(0)
    if not bool(matches.any(dim=1).all().detach().cpu()):
        raise ValueError("y_true contains labels that are not present in classes.")
    return matches.to(dtype=torch.long).argmax(dim=1)


def binary_log_loss_tensor(
    y_true: torch.Tensor,
    positive_proba: torch.Tensor,
    *,
    eps: float = 1e-15,
) -> torch.Tensor:
    y_true = y_true.to(device=positive_proba.device, dtype=positive_proba.dtype).reshape(-1)
    positive_proba = positive_proba.reshape(-1)
    dtype_eps = torch.finfo(positive_proba.dtype).eps
    effective_eps = max(float(eps), float(dtype_eps))
    positive_proba = positive_proba.clamp(
        min=effective_eps,
        max=1.0 - effective_eps,
    )
    return -(
        y_true * torch.log(positive_proba)
        + (1.0 - y_true) * torch.log(1.0 - positive_proba)
    ).mean()


def multiclass_log_loss_tensor(
    y_true: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    classes: Any | None = None,
    eps: float = 1e-15,
) -> torch.Tensor:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2D matrix for multiclass log loss.")
    y_indices = _class_indices_tensor(
        y_true.to(device=probabilities.device),
        classes=classes,
    )
    if y_indices.shape[0] != probabilities.shape[0]:
        raise ValueError("y_true and probabilities must contain the same number of rows.")
    class_count = probabilities.shape[1]
    if int(y_indices.min().detach().cpu()) < 0 or int(y_indices.max().detach().cpu()) >= class_count:
        raise ValueError("y_true contains class indices outside probability columns.")
    dtype_eps = torch.finfo(probabilities.dtype).eps
    effective_eps = max(float(eps), float(dtype_eps))
    clipped = probabilities.clamp(min=effective_eps, max=1.0)
    row_positions = torch.arange(
        clipped.shape[0],
        device=clipped.device,
        dtype=torch.long,
    )
    true_class_proba = clipped[row_positions, y_indices]
    return -torch.log(true_class_proba).mean()


def classification_log_loss_tensor(
    y_true: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    classes: Any | None = None,
    n_classes: int | None = None,
    eps: float = 1e-15,
) -> torch.Tensor:
    if probabilities.ndim == 0:
        if classes is not None:
            y_true = _class_indices_tensor(
                y_true.to(device=probabilities.device),
                classes=classes,
            ).to(device=probabilities.device, dtype=probabilities.dtype)
        return binary_log_loss_tensor(y_true, probabilities.reshape(1), eps=eps)
    if probabilities.ndim == 1:
        if classes is not None:
            y_true = _class_indices_tensor(
                y_true.to(device=probabilities.device),
                classes=classes,
            ).to(device=probabilities.device, dtype=probabilities.dtype)
        return binary_log_loss_tensor(y_true, probabilities, eps=eps)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 1D vector or 2D matrix.")

    class_count = (
        int(n_classes)
        if n_classes is not None
        else (len(classes) if classes is not None else probabilities.shape[1])
    )
    if class_count <= 2:
        positive_proba = (
            probabilities[:, 1]
            if probabilities.shape[1] > 1
            else probabilities.reshape(-1)
        )
        if classes is not None:
            y_true = _class_indices_tensor(
                y_true.to(device=probabilities.device),
                classes=classes,
            ).to(device=probabilities.device, dtype=probabilities.dtype)
        return binary_log_loss_tensor(y_true, positive_proba, eps=eps)
    return multiclass_log_loss_tensor(
        y_true,
        probabilities,
        classes=classes,
        eps=eps,
    )


def binary_log_loss(
    y_true: Any,
    probabilities: Any,
    *,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype = "float64",
) -> float:
    proba_tensor = to_device_tensor(probabilities, device=device, dtype=dtype)
    if proba_tensor.ndim == 2:
        proba_tensor = proba_tensor[:, 1]
    y_tensor = to_device_tensor(y_true, device=proba_tensor.device, dtype=proba_tensor.dtype)
    return float(binary_log_loss_tensor(y_tensor, proba_tensor).detach().cpu())


def classification_log_loss(
    y_true: Any,
    probabilities: Any,
    *,
    classes: Any | None = None,
    n_classes: int | None = None,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype = "float64",
) -> float:
    proba_tensor = to_device_tensor(probabilities, device=device, dtype=dtype)
    y_tensor = to_device_tensor(
        y_true,
        device=proba_tensor.device,
        dtype=proba_tensor.dtype,
    )
    return float(
        classification_log_loss_tensor(
            y_tensor,
            proba_tensor,
            classes=classes,
            n_classes=n_classes,
        ).detach().cpu()
    )


def binary_accuracy_score_tensor(
    y_true: torch.Tensor,
    positive_proba: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    y_true = y_true.to(device=positive_proba.device, dtype=positive_proba.dtype).reshape(-1)
    predictions = (positive_proba.reshape(-1) >= threshold).to(dtype=positive_proba.dtype)
    return (predictions == y_true).to(dtype=positive_proba.dtype).mean()


def classification_accuracy_score_tensor(
    y_true: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    classes: Any | None = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    if probabilities.ndim == 1 or (
        probabilities.ndim == 2 and probabilities.shape[1] <= 2
    ):
        positive_proba = (
            probabilities[:, 1]
            if probabilities.ndim == 2 and probabilities.shape[1] > 1
            else probabilities.reshape(-1)
        )
        if classes is not None:
            y_true = _class_indices_tensor(
                y_true.to(device=positive_proba.device),
                classes=classes,
            ).to(device=positive_proba.device, dtype=positive_proba.dtype)
        return binary_accuracy_score_tensor(
            y_true,
            positive_proba,
            threshold=threshold,
        )
    y_indices = _class_indices_tensor(
        y_true.to(device=probabilities.device),
        classes=classes,
    )
    predictions = probabilities.argmax(dim=1)
    return (predictions == y_indices).to(dtype=probabilities.dtype).mean()


def binary_roc_auc_score_tensor(
    y_true: torch.Tensor,
    positive_score: torch.Tensor,
) -> torch.Tensor:
    y_true = y_true.to(device=positive_score.device, dtype=torch.bool).reshape(-1)
    positive_score = positive_score.reshape(-1)
    positive_count = int(y_true.sum().detach().cpu())
    negative_count = int((~y_true).sum().detach().cpu())
    if positive_count == 0 or negative_count == 0:
        return torch.tensor(float("nan"), device=positive_score.device, dtype=positive_score.dtype)

    sorted_scores, order = torch.sort(positive_score)
    sorted_y = y_true[order]
    ranks = torch.arange(
        1,
        positive_score.numel() + 1,
        device=positive_score.device,
        dtype=positive_score.dtype,
    )

    # Average ranks for ties to match the Mann-Whitney AUC definition.
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    start = 0
    for count in counts.detach().cpu().tolist():
        if count > 1:
            stop = start + count
            ranks[start:stop] = ranks[start:stop].mean()
        start += count

    positive_rank_sum = ranks[sorted_y].sum()
    auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)
    return auc


def binary_roc_auc_score(
    y_true: Any,
    probabilities: Any,
    *,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype = "float64",
) -> float:
    proba_tensor = to_device_tensor(probabilities, device=device, dtype=dtype)
    if proba_tensor.ndim == 2:
        proba_tensor = proba_tensor[:, 1]
    y_tensor = to_device_tensor(y_true, device=proba_tensor.device, dtype=proba_tensor.dtype)
    return float(binary_roc_auc_score_tensor(y_tensor, proba_tensor).detach().cpu())


def with_bias_column(X: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            X,
            torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype),
        ],
        dim=1,
    )


def logistic_gradient_and_hessian(
    X: torch.Tensor,
    y: torch.Tensor,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if X.ndim != 2:
        raise ValueError("X must be a 2D tensor.")
    y = y.to(device=X.device, dtype=X.dtype).reshape(-1)
    theta = theta.to(device=X.device, dtype=X.dtype).reshape(-1)
    if y.shape[0] != X.shape[0]:
        raise ValueError("y must have the same number of rows as X.")
    if theta.shape[0] != X.shape[1] + 1:
        raise ValueError("theta must contain one weight per feature plus intercept.")

    logits = X @ theta[:-1] + theta[-1]
    probabilities = torch.sigmoid(logits)
    X_bias = with_bias_column(X)
    gradient = ((probabilities - y).unsqueeze(1) * X_bias).mean(dim=0)
    weights = probabilities * (1.0 - probabilities)
    hessian = (X_bias.T * weights) @ X_bias / X.shape[0]
    return gradient, hessian


def logistic_gradient(
    X: torch.Tensor,
    y: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    X = to_device_tensor(X, device=X.device, dtype=X.dtype, copy=False)
    if X.ndim != 2:
        raise ValueError("X must be a 2D tensor.")
    y = y.to(device=X.device, dtype=X.dtype).reshape(-1)
    theta = theta.to(device=X.device, dtype=X.dtype).reshape(-1)
    if y.shape[0] != X.shape[0]:
        raise ValueError("y must have the same number of rows as X.")
    if theta.shape[0] != X.shape[1] + 1:
        raise ValueError("theta must contain one weight per feature plus intercept.")

    logits = X @ theta[:-1] + theta[-1]
    probabilities = torch.sigmoid(logits)
    X_bias = with_bias_column(X)
    return ((probabilities - y).unsqueeze(1) * X_bias).mean(dim=0)


def _multiclass_parameters_from_theta(
    X: torch.Tensor,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if X.ndim != 2:
        raise ValueError("X must be a 2D tensor.")
    feature_count = X.shape[1]
    parameter_width = feature_count + 1
    theta = theta.to(device=X.device, dtype=X.dtype).reshape(-1)
    if theta.numel() % parameter_width != 0:
        raise ValueError(
            "theta must contain one feature vector plus intercept per class."
        )
    class_count = theta.numel() // parameter_width
    if class_count < 2:
        raise ValueError("multiclass theta must contain at least two classes.")
    theta_by_class = theta.reshape(class_count, parameter_width)
    return theta_by_class[:, :feature_count], theta_by_class[:, feature_count]


def multiclass_logistic_gradient_and_hessian(
    X: torch.Tensor,
    y: torch.Tensor,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    coef, intercept = _multiclass_parameters_from_theta(X, theta)
    class_count, feature_count = coef.shape
    y_indices = y.to(device=X.device, dtype=torch.long).reshape(-1)
    if y_indices.shape[0] != X.shape[0]:
        raise ValueError("y must have the same number of rows as X.")
    if int(y_indices.min().detach().cpu()) < 0 or int(y_indices.max().detach().cpu()) >= class_count:
        raise ValueError("y contains class indices outside theta classes.")

    logits = X @ coef.T + intercept
    probabilities = torch.softmax(logits, dim=1)
    targets = torch.nn.functional.one_hot(
        y_indices,
        num_classes=class_count,
    ).to(device=X.device, dtype=X.dtype)
    X_bias = with_bias_column(X)
    residuals = probabilities - targets
    gradient_by_class = residuals.T @ X_bias / X.shape[0]

    parameter_width = feature_count + 1
    hessian = torch.empty(
        (class_count, parameter_width, class_count, parameter_width),
        device=X.device,
        dtype=X.dtype,
    )
    for left_class in range(class_count):
        for right_class in range(class_count):
            if left_class == right_class:
                weights = probabilities[:, left_class] * (
                    1.0 - probabilities[:, right_class]
                )
            else:
                weights = -probabilities[:, left_class] * probabilities[:, right_class]
            hessian[left_class, :, right_class, :] = (
                (X_bias.T * weights) @ X_bias / X.shape[0]
            )
    return (
        gradient_by_class.reshape(-1),
        hessian.reshape(class_count * parameter_width, class_count * parameter_width),
    )


def multiclass_logistic_gradient(
    X: torch.Tensor,
    y: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    gradient, _ = multiclass_logistic_gradient_and_hessian(X, y, theta)
    return gradient


def damped_newton_delta(
    gradient: torch.Tensor,
    hessian: torch.Tensor,
    *,
    damping: float,
) -> torch.Tensor:
    if damping < 0:
        raise ValueError("damping must be non-negative.")
    eye = torch.eye(hessian.shape[0], device=hessian.device, dtype=hessian.dtype)
    damped_hessian = hessian + damping * eye
    try:
        return -torch.linalg.solve(damped_hessian, gradient)
    except RuntimeError:
        solution = torch.linalg.lstsq(damped_hessian, gradient.unsqueeze(1)).solution
        return -solution.squeeze(1)


def validation_delta_from_stats(
    parameter_delta: torch.Tensor,
    val_gradient: torch.Tensor,
    val_hessian: torch.Tensor,
    *,
    include_linear_term: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    linear = (
        val_gradient @ parameter_delta
        if include_linear_term
        else torch.zeros((), device=parameter_delta.device, dtype=parameter_delta.dtype)
    )
    quadratic = 0.5 * parameter_delta @ val_hessian @ parameter_delta
    return linear, quadratic


def model_theta_tensor(
    model: Any,
    *,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype = "float64",
) -> torch.Tensor:
    if hasattr(model, "coef_tensor_"):
        coef = model.coef_tensor_.detach()
        device = coef.device
        dtype = coef.dtype
    else:
        coef = to_device_tensor(getattr(model, "coef_"), device=device, dtype=dtype)
        if coef.ndim == 2 and coef.shape[0] == 1:
            coef = coef[0]
    if hasattr(model, "intercept_tensor_"):
        intercept = model.intercept_tensor_.detach().reshape(-1).to(device=coef.device, dtype=coef.dtype)
    else:
        intercept = to_device_tensor(
            getattr(model, "intercept_", [0.0]),
            device=coef.device,
            dtype=coef.dtype,
        ).reshape(-1)
    if coef.ndim == 2 and coef.shape[0] > 1:
        if intercept.numel() != coef.shape[0]:
            raise ValueError("multiclass intercept must contain one value per class.")
        return torch.cat(
            [
                torch.cat(
                    [
                        coef[class_index].reshape(-1),
                        intercept[class_index].reshape(1),
                    ]
                )
                for class_index in range(coef.shape[0])
            ]
        )
    if coef.ndim == 2:
        coef = coef[0]
    intercept = intercept[:1]
    return torch.cat([coef.reshape(-1), intercept])


def build_soft_interpolation_tensor(
    original: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    target = target.to(device=original.device, dtype=original.dtype)
    return original + alpha * (target - original)


def gpu_memory_summary(device: str | torch.device = "cuda") -> dict[str, float | str]:
    resolved = resolve_device(device)
    if resolved.type != "cuda":
        return {"device": str(resolved), "allocated_gb": 0.0, "reserved_gb": 0.0}
    return {
        "device": torch.cuda.get_device_name(resolved),
        "allocated_gb": torch.cuda.memory_allocated(resolved) / (1024 ** 3),
        "reserved_gb": torch.cuda.memory_reserved(resolved) / (1024 ** 3),
    }
