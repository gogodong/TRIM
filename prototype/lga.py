import numpy as np
import torch

try:
    from .gpu_math import (
        _class_indices_tensor,
        logistic_gradient,
        multiclass_logistic_gradient,
        model_theta_tensor,
        to_device_tensor,
    )
except ImportError:  # pragma: no cover - direct script/test import compatibility
    from gpu_math import (
        _class_indices_tensor,
        logistic_gradient,
        multiclass_logistic_gradient,
        model_theta_tensor,
        to_device_tensor,
    )


def cal_lga(
    train_original_encode,
    val_original_encode,
    train_y,
    val_y,
    train_current_encode,
    train_next_encode,
    backend_model,
    *,
    device="cuda",
    dtype="float64",
    changed_row_positions=None,
    train_next_changed_encode=None,
    precomputed_val_gradient=None,
    precomputed_train_current_gradient=None,
    precomputed_train_original_tensor=None,
    precomputed_val_original_tensor=None,
    precomputed_train_current_tensor=None,
    precomputed_train_y_tensor=None,
    precomputed_val_y_tensor=None,
    precomputed_theta=None,
):
    """Calculate g_val(original) dot (g_train(next) - g_train(current))."""
    if precomputed_theta is None:
        theta = model_theta_tensor(backend_model, device=device, dtype=dtype)
    elif isinstance(precomputed_theta, torch.Tensor):
        theta = precomputed_theta.detach().reshape(-1).contiguous()
    else:
        theta = to_device_tensor(
            precomputed_theta,
            device=device,
            dtype=dtype,
        ).reshape(-1)
    tensor_device = theta.device
    tensor_dtype = theta.dtype

    X_train_original = (
        to_device_tensor(
            precomputed_train_original_tensor,
            device=tensor_device,
            dtype=tensor_dtype,
        )
        if precomputed_train_original_tensor is not None
        else to_device_tensor(
            train_original_encode,
            device=tensor_device,
            dtype=tensor_dtype,
        )
    )
    X_val_original = (
        to_device_tensor(
            precomputed_val_original_tensor,
            device=tensor_device,
            dtype=tensor_dtype,
        )
        if precomputed_val_original_tensor is not None
        else to_device_tensor(
            val_original_encode,
            device=tensor_device,
            dtype=tensor_dtype,
        )
    )
    if precomputed_train_current_tensor is not None:
        X_train_current = to_device_tensor(
            precomputed_train_current_tensor,
            device=tensor_device,
            dtype=tensor_dtype,
        )
    else:
        X_train_current = (
            X_train_original
            if train_current_encode is None
            or (np.isscalar(train_current_encode) and train_current_encode == 0)
            else to_device_tensor(
                train_current_encode,
                device=tensor_device,
                dtype=tensor_dtype,
            )
        )
    if changed_row_positions is None:
        X_train_next = (
            X_train_original
            if train_next_encode is None
            or (np.isscalar(train_next_encode) and train_next_encode == 0)
            else to_device_tensor(
                train_next_encode,
                device=tensor_device,
                dtype=tensor_dtype,
            )
        )
    else:
        X_train_next = None
    y_train = to_device_tensor(
        precomputed_train_y_tensor
        if precomputed_train_y_tensor is not None
        else train_y,
        device=tensor_device,
        dtype=tensor_dtype,
    ).reshape(-1)
    y_val = to_device_tensor(
        precomputed_val_y_tensor
        if precomputed_val_y_tensor is not None
        else val_y,
        device=tensor_device,
        dtype=tensor_dtype,
    ).reshape(-1)
    classes = getattr(backend_model, "classes_", None)
    class_count = len(classes) if classes is not None else 2
    use_multiclass = class_count > 2
    if use_multiclass:
        y_train = _class_indices_tensor(y_train, classes=classes)
        y_val = _class_indices_tensor(y_val, classes=classes)

    if X_train_original.ndim != 2:
        raise ValueError("train_original_encode must be a 2D matrix.")
    if X_val_original.ndim != 2:
        raise ValueError("val_original_encode must be a 2D matrix.")
    if X_train_current.ndim != 2:
        raise ValueError("train_current_encode must be a 2D matrix.")
    if changed_row_positions is None and X_train_next.ndim != 2:
        raise ValueError("train_next_encode must be a 2D matrix.")
    if y_train.ndim != 1 or y_val.ndim != 1:
        raise ValueError("train_y and val_y must be 1D vectors.")
    if X_train_current.shape[0] != y_train.shape[0]:
        raise ValueError("train_current_encode and train_y must have the same rows.")
    if changed_row_positions is None and X_train_next.shape[0] != y_train.shape[0]:
        raise ValueError("train_next_encode and train_y must have the same rows.")
    if X_val_original.shape[0] != y_val.shape[0]:
        raise ValueError("val_original_encode and val_y must have the same rows.")

    expected_theta_count = (
        class_count * (X_train_original.shape[1] + 1)
        if use_multiclass
        else X_train_original.shape[1] + 1
    )
    if theta.shape[0] != expected_theta_count:
        raise ValueError(
            "backend_model parameters must match the encoded feature count and "
            "classification shape."
        )

    gradient_func = multiclass_logistic_gradient if use_multiclass else logistic_gradient

    if precomputed_val_gradient is None:
        g_val = gradient_func(X_val_original, y_val, theta)
    else:
        g_val = to_device_tensor(
            precomputed_val_gradient,
            device=tensor_device,
            dtype=tensor_dtype,
        ).reshape(-1)
        if g_val.shape != theta.shape:
            raise ValueError(
                "precomputed_val_gradient must match the backend_model "
                "parameter shape."
            )
    if changed_row_positions is not None:
        if train_next_changed_encode is None:
            raise ValueError(
                "train_next_changed_encode is required when changed_row_positions "
                "is provided."
            )
        if isinstance(changed_row_positions, torch.Tensor):
            changed_positions = changed_row_positions.to(
                device=tensor_device,
                dtype=torch.long,
            ).reshape(-1)
        else:
            changed_positions = torch.as_tensor(
                list(changed_row_positions),
                device=tensor_device,
                dtype=torch.long,
            ).reshape(-1)
        if changed_positions.numel() == 0:
            return 0.0
        X_train_next_changed = to_device_tensor(
            train_next_changed_encode,
            device=tensor_device,
            dtype=tensor_dtype,
        )
        if X_train_next_changed.ndim != 2:
            raise ValueError("train_next_changed_encode must be a 2D matrix.")
        if X_train_next_changed.shape[0] != changed_positions.numel():
            raise ValueError(
                "train_next_changed_encode must have one row per changed position."
            )
        if X_train_next_changed.shape[1] != X_train_current.shape[1]:
            raise ValueError(
                "train_next_changed_encode must have the same number of columns "
                "as train_current_encode."
            )
        y_changed = y_train[changed_positions]
        if precomputed_train_current_gradient is None:
            g_train_current = gradient_func(
                X_train_current[changed_positions],
                y_changed,
                theta,
            )
        else:
            g_train_current = to_device_tensor(
                precomputed_train_current_gradient,
                device=tensor_device,
                dtype=tensor_dtype,
            ).reshape(-1)
            if g_train_current.shape != theta.shape:
                raise ValueError(
                    "precomputed_train_current_gradient must match the "
                    "backend_model parameter shape."
                )
        g_train_next = gradient_func(
            X_train_next_changed,
            y_changed,
            theta,
        )
        gradient_delta = (
            g_train_next - g_train_current
        ) * (changed_positions.numel() / X_train_current.shape[0])
        return float((g_val @ gradient_delta).detach().cpu())

    g_train_current = gradient_func(X_train_current, y_train, theta)
    g_train_next = gradient_func(X_train_next, y_train, theta)

    return float((g_val @ (g_train_next - g_train_current)).detach().cpu())


def cal_lga_batch(
    train_original_encode,
    val_original_encode,
    train_y,
    val_y,
    train_current_encode,
    backend_model,
    *,
    candidate_changed_positions,
    candidate_next_changed_encodes,
    candidate_admitted_labels=None,
    candidate_current_changed_encodes=None,
    device="cuda",
    dtype="float64",
    precomputed_val_gradient=None,
    precomputed_train_original_tensor=None,
    precomputed_val_original_tensor=None,
    precomputed_train_current_tensor=None,
    precomputed_train_y_tensor=None,
    precomputed_val_y_tensor=None,
    precomputed_theta=None,
    batch_row_limit=50000,
):
    """Calculate changed-row or row-admission LGA scores in batches."""
    if len(candidate_changed_positions) != len(candidate_next_changed_encodes):
        raise ValueError(
            "candidate_changed_positions and candidate_next_changed_encodes "
            "must contain the same number of candidates."
        )
    if candidate_current_changed_encodes is not None and (
        len(candidate_current_changed_encodes) != len(candidate_changed_positions)
    ):
        raise ValueError(
            "candidate_current_changed_encodes must contain one matrix per candidate."
        )
    if candidate_admitted_labels is not None and (
        len(candidate_admitted_labels) != len(candidate_changed_positions)
    ):
        raise ValueError(
            "candidate_admitted_labels must contain one vector per candidate."
        )
    if (
        candidate_admitted_labels is not None
        and candidate_current_changed_encodes is not None
    ):
        raise ValueError(
            "candidate_current_changed_encodes cannot be used for row admissions."
        )
    if not candidate_changed_positions:
        return []

    if precomputed_theta is None:
        theta = model_theta_tensor(backend_model, device=device, dtype=dtype)
    elif isinstance(precomputed_theta, torch.Tensor):
        theta = precomputed_theta.detach().reshape(-1).contiguous()
    else:
        theta = to_device_tensor(
            precomputed_theta,
            device=device,
            dtype=dtype,
        ).reshape(-1)
    tensor_device = theta.device
    tensor_dtype = theta.dtype

    X_train_original = (
        to_device_tensor(
            precomputed_train_original_tensor,
            device=tensor_device,
            dtype=tensor_dtype,
        )
        if precomputed_train_original_tensor is not None
        else to_device_tensor(
            train_original_encode,
            device=tensor_device,
            dtype=tensor_dtype,
        )
    )
    X_val_original = (
        to_device_tensor(
            precomputed_val_original_tensor,
            device=tensor_device,
            dtype=tensor_dtype,
        )
        if precomputed_val_original_tensor is not None
        else to_device_tensor(
            val_original_encode,
            device=tensor_device,
            dtype=tensor_dtype,
        )
    )
    if precomputed_train_current_tensor is not None:
        X_train_current = to_device_tensor(
            precomputed_train_current_tensor,
            device=tensor_device,
            dtype=tensor_dtype,
        )
    else:
        X_train_current = (
            X_train_original
            if train_current_encode is None
            or (np.isscalar(train_current_encode) and train_current_encode == 0)
            else to_device_tensor(
                train_current_encode,
                device=tensor_device,
                dtype=tensor_dtype,
            )
        )
    y_train = to_device_tensor(
        precomputed_train_y_tensor
        if precomputed_train_y_tensor is not None
        else train_y,
        device=tensor_device,
        dtype=tensor_dtype,
    ).reshape(-1)
    y_val = to_device_tensor(
        precomputed_val_y_tensor
        if precomputed_val_y_tensor is not None
        else val_y,
        device=tensor_device,
        dtype=tensor_dtype,
    ).reshape(-1)
    classes = getattr(backend_model, "classes_", None)
    class_count = len(classes) if classes is not None else 2
    use_multiclass = class_count > 2
    if use_multiclass:
        y_train = _class_indices_tensor(y_train, classes=classes)
        y_val = _class_indices_tensor(y_val, classes=classes)

    if X_train_original.ndim != 2:
        raise ValueError("train_original_encode must be a 2D matrix.")
    if X_val_original.ndim != 2:
        raise ValueError("val_original_encode must be a 2D matrix.")
    if X_train_current.ndim != 2:
        raise ValueError("train_current_encode must be a 2D matrix.")
    if y_train.ndim != 1 or y_val.ndim != 1:
        raise ValueError("train_y and val_y must be 1D vectors.")
    if X_train_current.shape[0] != y_train.shape[0]:
        raise ValueError("train_current_encode and train_y must have the same rows.")
    if X_val_original.shape[0] != y_val.shape[0]:
        raise ValueError("val_original_encode and val_y must have the same rows.")

    expected_theta_count = (
        class_count * (X_train_original.shape[1] + 1)
        if use_multiclass
        else X_train_original.shape[1] + 1
    )
    if theta.shape[0] != expected_theta_count:
        raise ValueError(
            "backend_model parameters must match the encoded feature count and "
            "classification shape."
        )

    gradient_func = multiclass_logistic_gradient if use_multiclass else logistic_gradient
    if precomputed_val_gradient is None:
        g_val = gradient_func(X_val_original, y_val, theta)
    else:
        g_val = to_device_tensor(
            precomputed_val_gradient,
            device=tensor_device,
            dtype=tensor_dtype,
        ).reshape(-1)
        if g_val.shape != theta.shape:
            raise ValueError(
                "precomputed_val_gradient must match the backend_model "
                "parameter shape."
            )

    candidate_positions = []
    candidate_next = []
    candidate_current = []
    candidate_admitted_y = []
    for candidate_index, changed_row_positions in enumerate(candidate_changed_positions):
        if isinstance(changed_row_positions, torch.Tensor):
            changed_positions = changed_row_positions.to(
                device=tensor_device,
                dtype=torch.long,
            ).reshape(-1)
        else:
            changed_positions = torch.as_tensor(
                list(changed_row_positions),
                device=tensor_device,
                dtype=torch.long,
            ).reshape(-1)
        X_next_changed = to_device_tensor(
            candidate_next_changed_encodes[candidate_index],
            device=tensor_device,
            dtype=tensor_dtype,
        )
        if X_next_changed.ndim != 2:
            raise ValueError("candidate_next_changed_encodes must contain 2D matrices.")
        if X_next_changed.shape[0] != changed_positions.numel():
            raise ValueError(
                "each candidate_next_changed_encode must have one row per "
                "changed position."
            )
        if X_next_changed.shape[1] != X_train_current.shape[1]:
            raise ValueError(
                "candidate_next_changed_encodes must have the same number of "
                "columns as train_current_encode."
            )
        if candidate_admitted_labels is not None:
            admitted_y = to_device_tensor(
                candidate_admitted_labels[candidate_index],
                device=tensor_device,
                dtype=tensor_dtype,
            ).reshape(-1)
            if use_multiclass:
                admitted_y = _class_indices_tensor(admitted_y, classes=classes)
            if admitted_y.shape[0] != changed_positions.numel():
                raise ValueError(
                    "each candidate_admitted_labels vector must have one label "
                    "per admitted row."
                )
            X_current_changed = None
        elif candidate_current_changed_encodes is None:
            admitted_y = None
            X_current_changed = X_train_current[changed_positions]
        else:
            admitted_y = None
            X_current_changed = to_device_tensor(
                candidate_current_changed_encodes[candidate_index],
                device=tensor_device,
                dtype=tensor_dtype,
            )
            if X_current_changed.ndim != 2:
                raise ValueError(
                    "candidate_current_changed_encodes must contain 2D matrices."
                )
            if X_current_changed.shape[0] != changed_positions.numel():
                raise ValueError(
                    "each candidate_current_changed_encode must have one row per "
                    "changed position."
                )
            if X_current_changed.shape[1] != X_train_current.shape[1]:
                raise ValueError(
                    "candidate_current_changed_encodes must have the same number "
                    "of columns as train_current_encode."
                )
        candidate_positions.append(changed_positions)
        candidate_next.append(X_next_changed)
        candidate_current.append(X_current_changed)
        candidate_admitted_y.append(admitted_y)

    def per_row_gradients(X, y):
        X_bias = torch.cat(
            [
                X,
                torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype),
            ],
            dim=1,
        )
        if use_multiclass:
            parameter_width = X.shape[1] + 1
            theta_by_class = theta.reshape(class_count, parameter_width)
            logits = X @ theta_by_class[:, : X.shape[1]].T + theta_by_class[:, -1]
            probabilities = torch.softmax(logits, dim=1)
            targets = torch.nn.functional.one_hot(
                y.to(dtype=torch.long),
                num_classes=class_count,
            ).to(device=X.device, dtype=X.dtype)
            return ((probabilities - targets).unsqueeze(2) * X_bias.unsqueeze(1)).reshape(
                X.shape[0],
                -1,
            )
        logits = X @ theta[:-1] + theta[-1]
        probabilities = torch.sigmoid(logits)
        return (probabilities - y).unsqueeze(1) * X_bias

    if batch_row_limit is None or int(batch_row_limit) <= 0:
        effective_row_limit = sum(
            int(positions.numel())
            for positions in candidate_positions
        )
    else:
        effective_row_limit = int(batch_row_limit)

    admission_mode = candidate_admitted_labels is not None
    if admission_mode:
        current_train_gradient = gradient_func(
            X_train_current,
            y_train,
            theta,
        )
        current_train_row_count = X_train_current.shape[0]

    scores = [0.0] * len(candidate_positions)
    start = 0
    while start < len(candidate_positions):
        end = start
        row_count = 0
        while end < len(candidate_positions):
            candidate_rows = int(candidate_positions[end].numel())
            if row_count > 0 and row_count + candidate_rows > effective_row_limit:
                break
            row_count += candidate_rows
            end += 1
            if row_count >= effective_row_limit:
                break

        non_empty_indices = [
            candidate_index
            for candidate_index in range(start, end)
            if candidate_positions[candidate_index].numel() > 0
        ]
        if non_empty_indices:
            counts = torch.tensor(
                [
                    int(candidate_positions[candidate_index].numel())
                    for candidate_index in non_empty_indices
                ],
                device=tensor_device,
                dtype=torch.long,
            )
            X_next_all = torch.cat(
                [
                    candidate_next[candidate_index]
                    for candidate_index in non_empty_indices
                ],
                dim=0,
            )
            if admission_mode:
                y_changed_all = torch.cat(
                    [
                        candidate_admitted_y[candidate_index]
                        for candidate_index in non_empty_indices
                    ],
                    dim=0,
                )
            else:
                X_current_all = torch.cat(
                    [
                        candidate_current[candidate_index]
                        for candidate_index in non_empty_indices
                    ],
                    dim=0,
                )
                changed_positions_all = torch.cat(
                    [
                        candidate_positions[candidate_index]
                        for candidate_index in non_empty_indices
                    ],
                    dim=0,
                )
                y_changed_all = y_train[changed_positions_all]
            row_to_candidate = torch.repeat_interleave(
                torch.arange(
                    len(non_empty_indices),
                    device=tensor_device,
                    dtype=torch.long,
                ),
                counts,
            )
            next_row_gradients = per_row_gradients(X_next_all, y_changed_all)
            next_gradient_sums = torch.zeros(
                (len(non_empty_indices), theta.shape[0]),
                device=tensor_device,
                dtype=tensor_dtype,
            )
            next_gradient_sums.index_add_(0, row_to_candidate, next_row_gradients)
            counts_float = counts.to(dtype=tensor_dtype).unsqueeze(1)
            if admission_mode:
                next_gradient = (
                    current_train_row_count * current_train_gradient.unsqueeze(0)
                    + next_gradient_sums
                ) / (current_train_row_count + counts_float)
                gradient_delta = (
                    next_gradient - current_train_gradient.unsqueeze(0)
                )
            else:
                current_row_gradients = per_row_gradients(
                    X_current_all,
                    y_changed_all,
                )
                current_gradient_sums = torch.zeros_like(next_gradient_sums)
                current_gradient_sums.index_add_(
                    0,
                    row_to_candidate,
                    current_row_gradients,
                )
                gradient_delta = (
                    (next_gradient_sums / counts_float)
                    - (current_gradient_sums / counts_float)
                ) * (counts_float / X_train_current.shape[0])
            chunk_scores = gradient_delta @ g_val
            chunk_score_values = chunk_scores.detach().cpu().tolist()
            for score_index, candidate_index in enumerate(non_empty_indices):
                scores[candidate_index] = float(chunk_score_values[score_index])
        start = end

    return scores
