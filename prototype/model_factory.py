"""Configuration-driven registry for the three model roles.

Maps a model specification from the configuration's ``model`` section to a
zero-argument callable that builds a fresh model instance. Three roles are
resolved:

- ``estimator`` performs first-stage LGA ranking.
- ``proxy`` performs second-stage candidate certification.
- ``downstream`` performs final utility evaluation.

An omitted role returns ``None`` so the pipeline can apply its fallback. The
default LGA implementation reads coefficient and intercept tensors, so its
estimator must use a compatible flat linear model.
"""
from __future__ import annotations

from typing import Any, Callable

# Registered model families. The default LGA estimator requires a flat linear
# parameterization because it reads coefficient and intercept tensors.
_LINEAR_FAMILIES = {"torch_logistic"}
_DOWNSTREAM_FAMILIES = {"torch_logistic", "mlp", "xgboost"}


def _torch_logistic_factory(spec: dict, *, device, dtype, allow_cpu_fallback, random_state):
    try:
        from .gpu_logistic import TorchLogisticRegression
    except ImportError:  # pragma: no cover - direct script compatibility
        from gpu_logistic import TorchLogisticRegression

    return lambda: TorchLogisticRegression(
        max_iter=int(spec.get("max_iter", 300)),
        warm_start=bool(spec.get("warm_start", True)),
        device=device,
        dtype=dtype,
        allow_cpu_fallback=allow_cpu_fallback,
        lbfgs_history_size=int(spec.get("lbfgs_history_size", 100)),
        line_search_fn=spec.get("line_search_fn", "strong_wolfe"),
    )


def _mlp_factory(spec: dict, *, device, dtype, allow_cpu_fallback, random_state):
    try:
        from .gpu_mlp import TorchMLP
    except ImportError:  # pragma: no cover - direct script compatibility
        from gpu_mlp import TorchMLP

    hidden = spec.get("hidden_sizes")
    if hidden is None:
        hidden = (32, 32)
    return lambda: TorchMLP(
        hidden_sizes=tuple(int(h) for h in hidden),
        epochs=int(spec.get("epochs", spec.get("max_iter", 50))),
        lr=float(spec.get("lr", 1e-2)),
        weight_decay=float(spec.get("weight_decay", 1e-3)),
        warm_start=bool(spec.get("warm_start", False)),
        device=device,
        dtype=dtype,
        allow_cpu_fallback=allow_cpu_fallback,
        random_state=int(spec.get("random_state", random_state)),
    )


def _xgboost_factory(spec: dict, *, device, dtype, allow_cpu_fallback, random_state):
    try:
        from .gpu_xgboost import XGBoostGPUClassifier
    except ImportError:  # pragma: no cover - direct script compatibility
        from gpu_xgboost import XGBoostGPUClassifier

    # XGBoost resolves its own device and selects CPU when
    # allow_cpu_fallback is set and CUDA is unavailable.
    return lambda: XGBoostGPUClassifier(
        n_estimators=int(spec.get("n_estimators", 100)),
        max_depth=int(spec.get("max_depth", 3)),
        learning_rate=float(spec.get("learning_rate", 0.1)),
        subsample=float(spec.get("subsample", 1.0)),
        colsample_bytree=float(spec.get("colsample_bytree", 1.0)),
        reg_lambda=float(spec.get("reg_lambda", 1.0)),
        min_child_weight=float(spec.get("min_child_weight", 1.0)),
        tree_method=spec.get("tree_method", "hist"),
        device=spec.get("device", device),
        eval_metric=spec.get("eval_metric", "logloss"),
        random_state=int(spec.get("random_state", random_state)),
        n_jobs=int(spec.get("n_jobs", 1)),
        warm_start=bool(spec.get("warm_start", False)),
        allow_cpu_fallback=allow_cpu_fallback,
    )


_BUILDERS = {
    "torch_logistic": _torch_logistic_factory,
    "mlp": _mlp_factory,
    "xgboost": _xgboost_factory,
}


def build_model_factory(
    spec: Any,
    *,
    device: str = "cuda",
    dtype: str = "float64",
    allow_cpu_fallback: bool = True,
    random_state: int = 42,
    role: str = "downstream",
) -> Callable | None:
    """Build a zero-arg model factory from a config spec dict.

    Returns None when `spec` is None or empty, signalling
    ``run_trim_pipeline`` to apply its fallback for that role. ``role`` is used
    to validate model compatibility.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = {"family": spec}
    if not isinstance(spec, dict) or not spec:
        return None
    family = spec.get("family")
    if family is None:
        raise ValueError(f"model spec is missing 'family': {spec!r}")
    configured_task_type = spec.get("task_type")
    if configured_task_type not in {None, "classification"}:
        raise ValueError(
            "The TRIM model registry supports classification only; "
            f"got task_type={configured_task_type!r}."
        )
    if role == "estimator" and family not in _LINEAR_FAMILIES:
        raise ValueError(
            f"estimator model family must be a flat linear family "
            f"({_LINEAR_FAMILIES}); got {family!r}. The LGA/Hessian math reads "
            "coef_tensor_/intercept_tensor_."
        )
    if family not in _BUILDERS:
        raise ValueError(
            f"unknown model family {family!r}; known families: "
            f"{sorted(_BUILDERS)}"
        )
    if role == "downstream" and family not in _DOWNSTREAM_FAMILIES:
        # Keep downstream compatibility explicit as the registry grows.
        raise ValueError(
            f"downstream model family {family!r} is not supported as a "
            f"downstream model; supported: {sorted(_DOWNSTREAM_FAMILIES)}"
        )
    factory = _BUILDERS[family](
        spec,
        device=device,
        dtype=dtype,
        allow_cpu_fallback=allow_cpu_fallback,
        random_state=random_state,
    )
    factory.model_family = family
    factory.model_role = role
    return factory


def build_factories(
    config,
    *,
    device: str | None = None,
    dtype: str | None = None,
    allow_cpu_fallback: bool | None = None,
    random_state: int | None = None,
):
    """Resolve model factories from an object with a ``model`` attribute.

    Reads `config.model` (a dict with optional `downstream`/`proxy`/`estimator`
    sub-dicts). Missing roles return ``None`` so ``run_trim_pipeline`` can apply
    its fallbacks.
    """
    device = device if device is not None else getattr(config, "device", "cuda")
    dtype = dtype if dtype is not None else getattr(config, "dtype", "float64")
    allow_cpu_fallback = (
        allow_cpu_fallback
        if allow_cpu_fallback is not None
        else getattr(config, "allow_cpu_fallback", True)
    )
    random_state = (
        random_state if random_state is not None else getattr(config, "random_state", 42)
    )

    model_spec = getattr(config, "model", None)
    if not model_spec:
        return None, None, None

    downstream_spec = model_spec.get("downstream") if isinstance(model_spec, dict) else None
    proxy_spec = model_spec.get("proxy") if isinstance(model_spec, dict) else None
    estimator_spec = model_spec.get("estimator") if isinstance(model_spec, dict) else None

    downstream = build_model_factory(
        downstream_spec,
        device=device,
        dtype=dtype,
        allow_cpu_fallback=allow_cpu_fallback,
        random_state=random_state,
        role="downstream",
    )
    proxy = build_model_factory(
        proxy_spec,
        device=device,
        dtype=dtype,
        allow_cpu_fallback=allow_cpu_fallback,
        random_state=random_state,
        role="proxy",
    )
    estimator = build_model_factory(
        estimator_spec,
        device=device,
        dtype=dtype,
        allow_cpu_fallback=allow_cpu_fallback,
        random_state=random_state,
        role="estimator",
    )
    return downstream, proxy, estimator
