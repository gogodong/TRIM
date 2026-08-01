from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    from .gpu_math import resolve_device, resolve_dtype, tensor_to_numpy, to_device_tensor
except ImportError:  # pragma: no cover - direct script/test import compatibility
    from gpu_math import resolve_device, resolve_dtype, tensor_to_numpy, to_device_tensor


def _labels_to_numpy(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return tensor_to_numpy(values)
    return np.asarray(values)


def _label_key(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


class TorchMLP:
    """Small MLP trained with full-batch PyTorch Adam.

    It supports the proxy and downstream roles. The default LGA estimator
    requires coefficient and intercept tensors, which this model does not
    expose.
    """

    def __init__(
        self,
        *,
        hidden_sizes: Sequence[int] = (32, 32),
        epochs: int = 100,
        lr: float = 1e-2,
        weight_decay: float = 1e-3,
        warm_start: bool = False,
        device: str | torch.device = "cuda",
        dtype: str | torch.dtype = "float32",
        allow_cpu_fallback: bool = True,
        random_state: int = 42,
        n_classes: int | None = None,
        classes: Sequence[Any] | None = None,
    ):
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer.")
        self.hidden_sizes = tuple(int(units) for units in hidden_sizes)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.warm_start = bool(warm_start)
        self.device = resolve_device(device, allow_cpu_fallback=allow_cpu_fallback)
        self.dtype = resolve_dtype(dtype)
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        self.random_state = int(random_state)
        self.n_classes = int(n_classes) if n_classes is not None else None

        self.network_: nn.Module | None = None
        self.feature_count_: int | None = None
        self.output_count_: int | None = None
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

    def set_classes(self, classes: Sequence[Any]):
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
        if self.classes_ is None:
            return 0
        return int(len(self.classes_))

    def _is_multiclass(self) -> bool:
        return self._class_count() > 2

    def _output_count(self) -> int:
        if not self._is_multiclass():
            return 1
        return self._class_count()

    def _build_network(self, feature_count: int, output_count: int) -> nn.Module:
        layers: list[nn.Module] = []
        previous = feature_count
        for units in self.hidden_sizes:
            layers.append(nn.Linear(previous, units, device=self.device, dtype=self.dtype))
            layers.append(nn.ReLU())
            previous = units
        layers.append(nn.Linear(previous, output_count, device=self.device, dtype=self.dtype))
        return nn.Sequential(*layers)

    def _ensure_network(self, feature_count: int) -> nn.Module:
        output_count = self._output_count()
        reset = (
            self.network_ is None
            or self.feature_count_ != feature_count
            or self.output_count_ != output_count
            or not self.warm_start
        )
        if not reset and self.network_ is not None:
            self.network_ = self.network_.to(device=self.device, dtype=self.dtype)
            return self.network_
        torch.manual_seed(self.random_state)
        self.network_ = self._build_network(feature_count, output_count)
        self.feature_count_ = feature_count
        self.output_count_ = output_count
        return self.network_

    def _classification_target_indices(self, y: Any, *, device: torch.device) -> torch.Tensor:
        y_np = _labels_to_numpy(y).reshape(-1)
        unique = np.unique(y_np)
        if self._classes_are_default_binary and len(unique) > 2:
            self.set_classes(unique)
        elif self.classes_ is None:
            self.set_classes(unique if len(unique) > 2 else [0, 1])
        class_to_index = self._class_to_index_
        missing = [
            _label_key(label)
            for label in unique.tolist()
            if _label_key(label) not in class_to_index
        ]
        if missing:
            raise ValueError(f"y contains labels not present in classes: {missing}")
        indices = np.asarray(
            [class_to_index[_label_key(label)] for label in y_np.tolist()],
            dtype=np.int64,
        )
        return torch.as_tensor(indices, device=device, dtype=torch.long)

    def fit(self, X: Any, y: Any):
        X_tensor = to_device_tensor(
            X,
            device=self.device,
            dtype=self.dtype,
            allow_cpu_fallback=self.allow_cpu_fallback,
        )
        if X_tensor.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        y_target = self._classification_target_indices(
            y,
            device=X_tensor.device,
        )
        if y_target.shape[0] != X_tensor.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")

        self.device = X_tensor.device
        self.dtype = X_tensor.dtype
        network = self._ensure_network(X_tensor.shape[1])
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss() if self._is_multiclass() else nn.BCEWithLogitsLoss()
        network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            outputs = network(X_tensor)
            if self._is_multiclass():
                loss = loss_fn(outputs, y_target)
            else:
                loss = loss_fn(outputs.squeeze(1), y_target.to(dtype=X_tensor.dtype))
            loss.backward()
            optimizer.step()
        return self

    def decision_function_tensor(self, X: Any) -> torch.Tensor:
        if self.network_ is None:
            raise RuntimeError("TorchMLP must be fit before predicting.")
        X_tensor = to_device_tensor(
            X,
            device=self.device,
            dtype=self.dtype,
            allow_cpu_fallback=self.allow_cpu_fallback,
        )
        if X_tensor.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        if X_tensor.shape[1] != self.feature_count_:
            raise ValueError(
                "X has the wrong feature count: "
                f"expected {self.feature_count_}, got {X_tensor.shape[1]}."
            )
        self.network_ = self.network_.to(device=self.device, dtype=self.dtype)
        self.network_.eval()
        with torch.no_grad():
            outputs = self.network_(X_tensor)
            if outputs.shape[1] == 1:
                return outputs.squeeze(1)
            return outputs

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
