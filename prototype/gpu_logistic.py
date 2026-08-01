from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .gpu_math import (
        binary_accuracy_score_tensor,
        classification_accuracy_score_tensor,
        resolve_device,
        resolve_dtype,
        tensor_to_numpy,
        to_device_tensor,
    )
except ImportError:  # pragma: no cover - direct script/test import compatibility
    from gpu_math import (
        binary_accuracy_score_tensor,
        classification_accuracy_score_tensor,
        resolve_device,
        resolve_dtype,
        tensor_to_numpy,
        to_device_tensor,
    )


def _labels_to_numpy(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return tensor_to_numpy(values)
    return np.asarray(values)


def _label_key(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


class TorchLogisticRegression:
    """Logistic regression classifier trained with full-batch PyTorch LBFGS."""

    def __init__(
        self,
        *,
        max_iter: int = 500,
        tol: float = 1e-9,
        warm_start: bool = False,
        fit_intercept: bool = True,
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype = "float64",
        allow_cpu_fallback: bool = True,
        lbfgs_history_size: int = 100,
        line_search_fn: str | None = "strong_wolfe",
        track_path: bool = False,
        path_checkpoint_count: int = 16,
        n_classes: int | None = None,
        classes: Any | None = None,
    ):
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if path_checkpoint_count < 2:
            raise ValueError("path_checkpoint_count must be at least 2.")
        if not fit_intercept:
            raise ValueError("TorchLogisticRegression currently requires fit_intercept=True.")
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.warm_start = bool(warm_start)
        self.fit_intercept = bool(fit_intercept)
        self.device = resolve_device(device, allow_cpu_fallback=allow_cpu_fallback)
        self.dtype = resolve_dtype(dtype)
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        self.lbfgs_history_size = int(lbfgs_history_size)
        self.line_search_fn = line_search_fn
        self.track_path = bool(track_path)
        self.path_checkpoint_count = int(path_checkpoint_count)
        self.n_classes = int(n_classes) if n_classes is not None else None

        self.coef_tensor_: torch.Tensor | None = None
        self.intercept_tensor_: torch.Tensor | None = None
        self.train_tensor_: torch.Tensor | None = None
        self.train_label_tensor_: torch.Tensor | None = None
        self.path_thetas_: torch.Tensor | None = None
        self.n_iter_ = np.asarray([0], dtype=int)
        self.classes_ = None
        self._class_to_index_: dict[Any, int] = {}
        self._classes_are_default_binary = False
        if classes is not None:
            self.set_classes(classes)
        elif self.n_classes is not None:
            if self.n_classes < 2:
                raise ValueError("n_classes must be at least 2.")
            self.set_classes(np.arange(self.n_classes, dtype=int))
        else:
            self.classes_ = np.asarray([0, 1], dtype=int)
            self._class_to_index_ = {0: 0, 1: 1}
            self._classes_are_default_binary = True
        self.coef_ = None
        self.intercept_ = None

    def set_classes(self, classes):
        classes_array = np.asarray(list(classes))
        if classes_array.ndim != 1 or classes_array.size < 2:
            raise ValueError("classes must contain at least two labels.")
        if len(np.unique(classes_array)) != classes_array.size:
            raise ValueError("classes must be unique.")
        self.classes_ = classes_array
        self.n_classes = int(classes_array.size)
        self._class_to_index_ = {
            _label_key(label): index
            for index, label in enumerate(classes_array.tolist())
        }
        self._classes_are_default_binary = False
        return self

    def _class_count(self) -> int:
        return int(len(self.classes_)) if self.classes_ is not None else 0

    def _is_multiclass(self) -> bool:
        return self._class_count() > 2

    def _classification_target_indices(self, y: Any, *, device: torch.device) -> torch.Tensor:
        y_np = _labels_to_numpy(y).reshape(-1)
        unique = np.unique(y_np)
        if self._classes_are_default_binary and len(unique) > 2:
            self.set_classes(unique)
        elif self.classes_ is None:
            self.set_classes(unique if len(unique) > 2 else [0, 1])
        missing = [
            _label_key(label)
            for label in unique.tolist()
            if _label_key(label) not in self._class_to_index_
        ]
        if missing:
            raise ValueError(f"y contains labels not present in classes: {missing}")
        indices = np.asarray(
            [self._class_to_index_[_label_key(label)] for label in y_np.tolist()],
            dtype=np.int64,
        )
        return torch.as_tensor(indices, device=device, dtype=torch.long)

    def _flatten_parameters(
        self,
        coef: torch.Tensor,
        intercept: torch.Tensor,
    ) -> torch.Tensor:
        if coef.ndim == 2:
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
        return torch.cat([coef.reshape(-1), intercept.reshape(1)])

    def _ensure_parameters(self, feature_count: int, *, reset: bool = False) -> None:
        if self._is_multiclass():
            expected_coef_shape = (self._class_count(), feature_count)
            expected_intercept_shape = (self._class_count(),)
            should_reset = (
                reset
                or self.coef_tensor_ is None
                or self.intercept_tensor_ is None
                or tuple(self.coef_tensor_.shape) != expected_coef_shape
                or tuple(self.intercept_tensor_.shape) != expected_intercept_shape
            )
            if not should_reset:
                self.coef_tensor_ = self.coef_tensor_.to(device=self.device, dtype=self.dtype)
                self.intercept_tensor_ = self.intercept_tensor_.to(device=self.device, dtype=self.dtype)
                return
            self.coef_tensor_ = torch.zeros(
                expected_coef_shape,
                device=self.device,
                dtype=self.dtype,
            )
            self.intercept_tensor_ = torch.zeros(
                expected_intercept_shape,
                device=self.device,
                dtype=self.dtype,
            )
            self._sync_numpy_parameters()
            return

        should_reset = (
            reset
            or self.coef_tensor_ is None
            or self.intercept_tensor_ is None
            or self.coef_tensor_.numel() != feature_count
        )
        if not should_reset:
            self.coef_tensor_ = self.coef_tensor_.to(device=self.device, dtype=self.dtype)
            self.intercept_tensor_ = self.intercept_tensor_.to(device=self.device, dtype=self.dtype)
            return

        self.coef_tensor_ = torch.zeros(feature_count, device=self.device, dtype=self.dtype)
        self.intercept_tensor_ = torch.zeros((), device=self.device, dtype=self.dtype)
        self._sync_numpy_parameters()

    def _sync_numpy_parameters(self) -> None:
        if self.coef_tensor_ is None or self.intercept_tensor_ is None:
            self.coef_ = None
            self.intercept_ = None
            return
        if self.coef_tensor_.ndim == 2:
            self.coef_ = tensor_to_numpy(self.coef_tensor_)
            self.intercept_ = tensor_to_numpy(self.intercept_tensor_.reshape(-1))
            return
        self.coef_ = tensor_to_numpy(self.coef_tensor_).reshape(1, -1)
        self.intercept_ = tensor_to_numpy(self.intercept_tensor_.reshape(1))

    def fit(self, X: Any, y: Any):
        X_tensor = to_device_tensor(
            X,
            device=self.device,
            dtype=self.dtype,
            allow_cpu_fallback=self.allow_cpu_fallback,
        )
        y_indices = self._classification_target_indices(y, device=X_tensor.device)
        if X_tensor.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        if y_indices.shape[0] != X_tensor.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")

        self.device = X_tensor.device
        self.dtype = X_tensor.dtype
        self.train_tensor_ = X_tensor
        self.train_label_tensor_ = y_indices
        self._ensure_parameters(X_tensor.shape[1], reset=not self.warm_start)

        coef = torch.nn.Parameter(self.coef_tensor_.detach().clone())
        intercept = torch.nn.Parameter(self.intercept_tensor_.detach().clone())
        self.path_thetas_ = None

        if self.track_path:
            path_snapshots = [self._flatten_parameters(coef.detach(), intercept.detach()).clone()]
            optimizer = torch.optim.LBFGS(
                [coef, intercept],
                lr=1.0,
                max_iter=1,
                tolerance_grad=self.tol,
                tolerance_change=self.tol,
                history_size=self.lbfgs_history_size,
                line_search_fn=self.line_search_fn,
            )

            def closure():
                optimizer.zero_grad(set_to_none=True)
                if self._is_multiclass():
                    logits = X_tensor @ coef.T + intercept
                    loss = F.cross_entropy(logits, y_indices)
                else:
                    logits = X_tensor @ coef + intercept
                    loss = F.binary_cross_entropy_with_logits(
                        logits,
                        y_indices.to(dtype=X_tensor.dtype),
                    )
                loss.backward()
                return loss

            completed_steps = 0
            for _ in range(self.max_iter):
                optimizer.step(closure)
                completed_steps += 1
                snapshot = self._flatten_parameters(
                    coef.detach(),
                    intercept.detach(),
                ).clone()
                if torch.equal(snapshot, path_snapshots[-1]):
                    break
                path_snapshots.append(snapshot)

            state = optimizer.state.get(coef, {})
            self.n_iter_ = np.asarray(
                [int(state.get("n_iter", completed_steps))],
                dtype=int,
            )
            self.coef_tensor_ = coef.detach().clone()
            self.intercept_tensor_ = intercept.detach().clone()
            if len(path_snapshots) > self.path_checkpoint_count:
                indices = torch.linspace(
                    0,
                    len(path_snapshots) - 1,
                    steps=self.path_checkpoint_count,
                    device=self.device,
                ).round().to(dtype=torch.long)
                self.path_thetas_ = torch.stack(
                    [
                        path_snapshots[int(index.detach().cpu())]
                        for index in indices
                    ]
                )
            else:
                self.path_thetas_ = torch.stack(path_snapshots)
            self._sync_numpy_parameters()
            return self

        optimizer = torch.optim.LBFGS(
            [coef, intercept],
            lr=1.0,
            max_iter=self.max_iter,
            tolerance_grad=self.tol,
            tolerance_change=self.tol,
            history_size=self.lbfgs_history_size,
            line_search_fn=self.line_search_fn,
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            if self._is_multiclass():
                logits = X_tensor @ coef.T + intercept
                loss = F.cross_entropy(logits, y_indices)
            else:
                logits = X_tensor @ coef + intercept
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    y_indices.to(dtype=X_tensor.dtype),
                )
            loss.backward()
            return loss

        optimizer.step(closure)
        state = optimizer.state.get(coef, {})
        self.n_iter_ = np.asarray([int(state.get("n_iter", self.max_iter))], dtype=int)
        self.coef_tensor_ = coef.detach().clone()
        self.intercept_tensor_ = intercept.detach().clone()
        self._sync_numpy_parameters()
        return self

    def decision_function_tensor(self, X: Any) -> torch.Tensor:
        X_tensor = to_device_tensor(
            X,
            device=self.device,
            dtype=self.dtype,
            allow_cpu_fallback=self.allow_cpu_fallback,
        )
        if X_tensor.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        self._ensure_parameters(X_tensor.shape[1], reset=False)
        if self._is_multiclass():
            return X_tensor @ self.coef_tensor_.T + self.intercept_tensor_
        return X_tensor @ self.coef_tensor_ + self.intercept_tensor_

    def predict_proba_tensor(self, X: Any) -> torch.Tensor:
        logits = self.decision_function_tensor(X)
        if self._is_multiclass():
            return torch.softmax(logits, dim=1)
        positive = torch.sigmoid(logits)
        return torch.stack([1.0 - positive, positive], dim=1)

    def predict_proba(self, X: Any) -> np.ndarray:
        return tensor_to_numpy(self.predict_proba_tensor(X))

    def predict_tensor(self, X: Any) -> torch.Tensor:
        probabilities = self.predict_proba_tensor(X)
        if self._is_multiclass():
            class_indices = probabilities.argmax(dim=1)
        else:
            class_indices = (probabilities[:, 1] >= 0.5).to(dtype=torch.long)
        classes_tensor = torch.as_tensor(self.classes_, device=probabilities.device)
        return classes_tensor.index_select(0, class_indices)

    def predict(self, X: Any) -> np.ndarray:
        return tensor_to_numpy(self.predict_tensor(X))

    def score_tensor(self, X: Any, y: Any) -> torch.Tensor:
        X_tensor = to_device_tensor(
            X,
            device=self.device,
            dtype=self.dtype,
            allow_cpu_fallback=self.allow_cpu_fallback,
        )
        y_tensor = to_device_tensor(
            y,
            device=X_tensor.device,
            dtype=X_tensor.dtype,
            allow_cpu_fallback=self.allow_cpu_fallback,
        )
        probabilities = self.predict_proba_tensor(X_tensor)
        if self._is_multiclass():
            return classification_accuracy_score_tensor(
                y_tensor,
                probabilities,
                classes=self.classes_,
            )
        return binary_accuracy_score_tensor(y_tensor, probabilities[:, 1])

    def score(self, X: Any, y: Any) -> float:
        return float(self.score_tensor(X, y).detach().cpu())
