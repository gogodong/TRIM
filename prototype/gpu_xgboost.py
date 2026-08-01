from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

try:
    from .gpu_math import resolve_device, resolve_dtype, tensor_to_numpy, to_device_tensor
except ImportError:  # pragma: no cover - direct script/test import compatibility
    from gpu_math import resolve_device, resolve_dtype, tensor_to_numpy, to_device_tensor


def _to_numpy(values: Any) -> Any:
    """Coerce model input to a contiguous ndarray or preserve CSR input.

    This wrapper supplies NumPy or CSR inputs and lets XGBoost move them to its
    configured device.
    """
    if isinstance(values, torch.Tensor):
        return tensor_to_numpy(values)
    if isinstance(values, np.ndarray):
        return np.ascontiguousarray(values)
    try:
        from scipy import sparse  # type: ignore

        if sparse.issparse(values):
            values = values.tocsr()
            # A maximally generalized leaf-space row can give every feature a
            # positive weight. XGBoost's GPU histogram path treats a fully
            # populated CSR as dense but still follows its sparse missing-bin
            # bookkeeping, which can trigger a device assertion. Dense storage
            # is also smaller once every entry is present.
            if values.nnz == values.shape[0] * values.shape[1]:
                return np.ascontiguousarray(values.toarray())
            return values
    except ImportError:
        pass
    try:
        import pandas as pd  # type: ignore

        if isinstance(values, (pd.DataFrame, pd.Series)):
            return np.ascontiguousarray(values.to_numpy())
    except ImportError:
        pass
    return np.ascontiguousarray(np.asarray(values))


def _labels_to_numpy(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return tensor_to_numpy(values)
    return np.asarray(values)


def _label_key(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


class XGBoostGPUClassifier:
    """XGBoost classifier wrapper for the TRIM pipeline.

    The wrapper implements the fitting and probability-prediction interfaces
    required by the proxy and downstream roles. The default LGA estimator
    requires coefficient and intercept tensors, which this model does not
    expose.

    `device="cuda"` runs the histogram tree method on GPU, and `device="cuda:N"`
    selects a specific GPU ordinal. Set `device="cpu"` for CPU execution.
    `warm_start` is accepted for interface compatibility and is a no-op because
    XGBoost's scikit-learn API refits from scratch on each `.fit` call.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_lambda: float = 1.0,
        min_child_weight: float = 1.0,
        tree_method: str = "hist",
        device: str | torch.device = "cuda",
        eval_metric: str = "logloss",
        random_state: int = 42,
        n_jobs: int = 1,
        warm_start: bool = False,
        allow_cpu_fallback: bool = True,
        n_classes: int | None = None,
        classes: Sequence[Any] | None = None,
    ):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_lambda = float(reg_lambda)
        self.min_child_weight = float(min_child_weight)
        self.tree_method = tree_method
        resolved = resolve_device(device, allow_cpu_fallback=allow_cpu_fallback)
        self.device = str(resolved)
        self.eval_metric = eval_metric
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.warm_start = bool(warm_start)
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        self.n_classes = int(n_classes) if n_classes is not None else None

        self.model_ = None
        self.feature_count_: int | None = None
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
        # Output tensor device for predict_proba_tensor. Defaults to the
        # resolved XGBoost device; adopt the model dtype convention float32.
        self._output_device = torch.device(self.device)
        self._output_dtype = torch.float32

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
        return int(len(self.classes_)) if self.classes_ is not None else 0

    def _is_multiclass(self) -> bool:
        return self._class_count() > 2

    def _classification_target_indices(self, y: Any) -> np.ndarray:
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
        return np.asarray(
            [self._class_to_index_[_label_key(label)] for label in y_np.tolist()],
            dtype=np.int64,
        )

    def _build_estimator(self):
        from xgboost import XGBClassifier

        objective = "multi:softprob" if self._is_multiclass() else "binary:logistic"
        eval_metric = (
            "mlogloss"
            if self._is_multiclass() and self.eval_metric == "logloss"
            else self.eval_metric
        )
        kwargs = {}
        if self._is_multiclass():
            kwargs["num_class"] = self._class_count()
        return XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            tree_method=self.tree_method,
            device=self.device,
            objective=objective,
            eval_metric=eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **kwargs,
        )

    def fit(self, X: Any, y: Any):
        X_np = _to_numpy(X)
        y_np = _labels_to_numpy(y).reshape(-1)
        if X_np.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        if y_np.shape[0] != X_np.shape[0]:
            raise ValueError("X and y must contain the same number of rows.")
        # XGBoost requires float features and integer labels.
        if not np.issubdtype(X_np.dtype, np.floating):
            X_np = X_np.astype(np.float32)
        else:
            X_np = X_np.astype(np.float32, copy=False)
        y_indices = self._classification_target_indices(y_np)
        sample_weight = None
        missing_class_indices = sorted(
            set(range(self._class_count())) - set(np.unique(y_indices).tolist())
        )
        if missing_class_indices:
            if X_np.shape[0] == 0:
                raise ValueError("X must contain at least one row.")
            try:
                from scipy import sparse  # type: ignore
            except ImportError:
                sparse = None
            if sparse is not None and sparse.issparse(X_np):
                X_np = sparse.vstack(
                    [X_np] + [X_np[:1]] * len(missing_class_indices),
                    format="csr",
                )
            else:
                X_np = np.vstack(
                    [
                        X_np,
                        np.repeat(X_np[:1], len(missing_class_indices), axis=0),
                    ]
                )
            y_indices = np.concatenate(
                [
                    y_indices,
                    np.asarray(missing_class_indices, dtype=np.int64),
                ]
            )
            sample_weight = np.concatenate(
                [
                    np.ones(X_np.shape[0] - len(missing_class_indices), dtype=np.float32),
                    np.zeros(len(missing_class_indices), dtype=np.float32),
                ]
            )

        self.feature_count_ = X_np.shape[1]
        self.model_ = self._build_estimator()
        self.model_.fit(X_np, y_indices, sample_weight=sample_weight)
        return self

    def predict_proba_tensor(self, X: Any) -> torch.Tensor:
        if self.model_ is None:
            raise RuntimeError("XGBoostGPUClassifier must be fit before predicting.")
        X_np = _to_numpy(X)
        if X_np.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        if X_np.shape[1] != self.feature_count_:
            raise ValueError(
                "X has the wrong feature count: "
                f"expected {self.feature_count_}, got {X_np.shape[1]}."
            )
        if not np.issubdtype(X_np.dtype, np.floating):
            X_np = X_np.astype(np.float32)
        else:
            X_np = X_np.astype(np.float32, copy=False)
        proba = self.model_.predict_proba(X_np)
        if proba.ndim == 1:
            proba = np.stack([1.0 - proba, proba], axis=1)
        return torch.as_tensor(
            proba, device=self._output_device, dtype=self._output_dtype
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.predict_proba_tensor(X).detach().cpu().numpy()

    def predict_tensor(self, X: Any) -> torch.Tensor:
        probabilities = self.predict_proba_tensor(X)
        if self._is_multiclass():
            class_indices = probabilities.argmax(dim=1)
        else:
            class_indices = (probabilities[:, 1] >= 0.5).to(dtype=torch.long)
        classes_tensor = torch.as_tensor(self.classes_, device=probabilities.device)
        return classes_tensor.index_select(0, class_indices)

    def predict(self, X: Any) -> np.ndarray:
        return self.predict_tensor(X).detach().cpu().numpy()
