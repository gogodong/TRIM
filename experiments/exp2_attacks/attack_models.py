"""Attack-only MLP models.

TRIM supports classification only. Numeric
attribute reconstruction is an attacker task, so its regressor lives here and
is used only by the attack implementation.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from prototype.gpu_math import resolve_device, resolve_dtype, tensor_to_numpy, to_device_tensor
from prototype.gpu_mlp import TorchMLP


def _label_key(value: Any) -> Any:
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, float) and np.isnan(value):
        return "__MISSING__"
    return value


class CategoricalAttackMLP:
    """Encode arbitrary raw labels before delegating to the core Torch MLP."""

    def __init__(
        self,
        classes: Sequence[Any],
        *,
        hidden_sizes=(32, 32),
        epochs=300,
        lr=1e-2,
        weight_decay=1e-3,
        device="cuda",
        dtype="float32",
        allow_cpu_fallback=True,
        random_state=42,
    ):
        self.classes_ = np.asarray(list(classes), dtype=object)
        if self.classes_.size < 2:
            raise ValueError("CategoricalAttackMLP needs at least two classes.")
        self._class_to_index = {
            _label_key(value): index
            for index, value in enumerate(self.classes_.tolist())
        }
        self.model = TorchMLP(
            hidden_sizes=hidden_sizes,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            warm_start=False,
            device=device,
            dtype=dtype,
            allow_cpu_fallback=allow_cpu_fallback,
            random_state=random_state,
            classes=np.arange(self.classes_.size, dtype=int),
        )

    def fit(self, X, y):
        encoded = np.asarray(
            [self._class_to_index[_label_key(value)] for value in list(y)],
            dtype=int,
        )
        self.model.fit(X, encoded)
        return self

    def predict(self, X) -> np.ndarray:
        indices = np.asarray(self.model.predict(X), dtype=int)
        return self.classes_[indices]


class NumericAttackMLP:
    """Small full-batch MLP regressor used only by the reconstruction attack."""

    def __init__(
        self,
        *,
        hidden_sizes=(32, 32),
        epochs=300,
        lr=1e-2,
        weight_decay=1e-3,
        device="cuda",
        dtype="float32",
        allow_cpu_fallback=True,
        random_state=42,
    ):
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer.")
        self.epochs = int(epochs)
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.device = resolve_device(device, allow_cpu_fallback=allow_cpu_fallback)
        self.dtype = resolve_dtype(dtype)
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        self.random_state = int(random_state)
        self.network_: nn.Module | None = None
        self.feature_count_: int | None = None
        self.target_mean_: float | None = None
        self.target_scale_: float | None = None

    def _build_network(self, feature_count: int) -> nn.Module:
        layers: list[nn.Module] = []
        previous = feature_count
        for width in self.hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous, width, device=self.device, dtype=self.dtype),
                    nn.ReLU(),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, 1, device=self.device, dtype=self.dtype))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        X_tensor = to_device_tensor(
            X,
            device=self.device,
            dtype=self.dtype,
            allow_cpu_fallback=self.allow_cpu_fallback,
        )
        if X_tensor.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        y_values = np.asarray(y, dtype=float).reshape(-1)
        if y_values.shape[0] != X_tensor.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")
        if not np.isfinite(y_values).all():
            raise ValueError("Numeric attacker targets must be finite.")
        self.target_mean_ = float(np.mean(y_values))
        target_std = float(np.std(y_values))
        self.target_scale_ = target_std if target_std > 0.0 else 1.0
        y_normalized = (y_values - self.target_mean_) / self.target_scale_
        y_tensor = torch.as_tensor(
            y_normalized,
            device=X_tensor.device,
            dtype=X_tensor.dtype,
        )

        torch.manual_seed(self.random_state)
        self.device = X_tensor.device
        self.dtype = X_tensor.dtype
        self.feature_count_ = int(X_tensor.shape[1])
        self.network_ = self._build_network(self.feature_count_)
        optimizer = torch.optim.Adam(
            self.network_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        loss_fn = nn.MSELoss()
        self.network_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            prediction = self.network_(X_tensor).squeeze(1)
            loss = loss_fn(prediction, y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X) -> np.ndarray:
        if self.network_ is None or self.feature_count_ is None:
            raise RuntimeError("NumericAttackMLP must be fit before predicting.")
        X_tensor = to_device_tensor(
            X,
            device=self.device,
            dtype=self.dtype,
            allow_cpu_fallback=self.allow_cpu_fallback,
        )
        if X_tensor.ndim != 2 or X_tensor.shape[1] != self.feature_count_:
            raise ValueError("X has the wrong shape for the fitted attacker.")
        self.network_.eval()
        with torch.no_grad():
            normalized = self.network_(X_tensor).squeeze(1)
        return (
            tensor_to_numpy(normalized) * float(self.target_scale_)
            + float(self.target_mean_)
        )


def build_attack_model(
    target_type: str,
    *,
    classes: Sequence[Any] | None,
    spec: Mapping[str, Any] | None,
    device: str,
    dtype: str,
    allow_cpu_fallback: bool,
    random_state: int,
):
    """Build an explicitly configured attack model."""
    if not isinstance(spec, Mapping):
        raise ValueError("Attack model config must be a YAML mapping.")
    model_spec = dict(spec)
    required_options = ("family", "hidden_sizes", "epochs", "lr", "weight_decay")
    missing_options = [name for name in required_options if name not in model_spec]
    if missing_options:
        raise ValueError(
            f"Attack model config is missing required options: {missing_options}"
        )
    family = str(model_spec.pop("family"))
    if family != "mlp":
        raise ValueError(f"Attack model family must be 'mlp'; got {family!r}.")
    hidden_sizes = tuple(model_spec.pop("hidden_sizes"))
    epochs = int(model_spec.pop("epochs"))
    common = {
        "hidden_sizes": hidden_sizes,
        "epochs": epochs,
        "lr": float(model_spec.pop("lr")),
        "weight_decay": float(model_spec.pop("weight_decay")),
        "device": device,
        "dtype": dtype,
        "allow_cpu_fallback": allow_cpu_fallback,
        "random_state": random_state,
    }
    if model_spec:
        raise ValueError(f"Unknown attack model options: {sorted(model_spec)}")
    if target_type == "numeric":
        return NumericAttackMLP(**common)
    if target_type == "categorical":
        if classes is None:
            raise ValueError("Categorical attack models require explicit classes.")
        return CategoricalAttackMLP(classes, **common)
    raise ValueError(f"Unknown attack target type: {target_type!r}")
