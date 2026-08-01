"""Matched minibatch training and privacy accounting for Experiment 1."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

from prototype.gpu_math import resolve_device, resolve_dtype, to_device_tensor


@dataclass
class MethodDataset:
    """One published representation and its representation-matched test panel."""

    method: str
    train_frame: Any
    train_y: Any
    evaluation_frame: Any
    evaluation_y: Any
    final_min_k: int
    row_count: int
    unique_row_count: int
    published_row_count: int
    source_detail: str
    row_semantics: str
    evaluation_protocol: str
    privacy_constraint_met: bool

    def __post_init__(self) -> None:
        if len(self.train_frame) != len(self.train_y):
            raise ValueError(f"{self.method}: training X/y lengths disagree.")
        if len(self.evaluation_frame) != len(self.evaluation_y):
            raise ValueError(f"{self.method}: evaluation X/y lengths disagree.")
        if int(self.row_count) != len(self.train_frame):
            raise ValueError(f"{self.method}: row_count disagrees with training data.")
        if list(self.train_frame.columns) != list(self.evaluation_frame.columns):
            raise ValueError(
                f"{self.method}: train and evaluation feature columns disagree."
            )


@dataclass(frozen=True)
class NoiseCalibration:
    noise_multiplier: float
    actual_epsilon: float


@dataclass(frozen=True)
class TrainingOutcome:
    test_logloss: float
    sample_rate: float
    expected_batch_size: int
    training_steps: int
    actual_epsilon: float | None


def _require_opacus():
    try:
        from opacus import PrivacyEngine
        from opacus.accountants.utils import create_accountant
        from opacus.data_loader import DPDataLoader
    except ImportError as exc:
        raise RuntimeError(
            "Experiment 1 requires the optional `opacus` package. It is used "
            "for the matched Poisson-SGD loader, the clipping-only control, "
            "DP-SGD, and privacy accounting. Install the project "
            "environment before running this experiment."
        ) from exc
    return PrivacyEngine, create_accountant, DPDataLoader


def effective_batch_size_for_dataset(
    *, dataset_size: int, batch_size: int
) -> int:
    dataset_size = int(dataset_size)
    batch_size = int(batch_size)
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return min(dataset_size, batch_size)


def dp_accounting_schedule(
    *, dataset_size: int, batch_size: int, epochs: int
) -> tuple[float, int, int]:
    """Return Opacus's Poisson sample rate, steps, and expected batch size."""
    dataset_size = int(dataset_size)
    epochs = int(epochs)
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    effective_batch_size = effective_batch_size_for_dataset(
        dataset_size=dataset_size, batch_size=batch_size
    )
    steps_per_epoch = int(math.ceil(dataset_size / effective_batch_size))
    sample_rate = 1.0 / float(steps_per_epoch)
    expected_batch_size = int(dataset_size * sample_rate)
    if expected_batch_size <= 0:
        raise RuntimeError("The expected Poisson minibatch size is zero.")
    return sample_rate, steps_per_epoch * epochs, expected_batch_size


def epsilon_for_noise_multiplier(
    noise_multiplier: float,
    *,
    dataset_size: int,
    batch_size: int,
    epochs: int,
    delta: float,
    accountant: str = "prv",
) -> float:
    """Compute row-level epsilon for the exact training schedule."""
    _PrivacyEngine, create_accountant, _DPDataLoader = _require_opacus()
    if not 0.0 < float(delta) < 1.0:
        raise ValueError("delta must be in (0, 1).")
    if float(noise_multiplier) <= 0.0:
        return math.inf
    sample_rate, steps, _expected_batch_size = dp_accounting_schedule(
        dataset_size=dataset_size,
        batch_size=batch_size,
        epochs=epochs,
    )
    privacy_accountant = create_accountant(mechanism=str(accountant))
    for _ in range(steps):
        privacy_accountant.step(
            noise_multiplier=float(noise_multiplier),
            sample_rate=sample_rate,
        )
    return float(privacy_accountant.get_epsilon(delta=float(delta)))


def calibrate_noise_multiplier(
    *,
    target_epsilon: float,
    dataset_size: int,
    batch_size: int,
    epochs: int,
    delta: float,
    tolerance: float = 0.01,
    accountant: str = "prv",
    max_iterations: int = 60,
) -> NoiseCalibration:
    """Find the least Gaussian noise scale whose epsilon is at most the target."""
    target_epsilon = float(target_epsilon)
    tolerance = float(tolerance)
    if not math.isfinite(target_epsilon) or target_epsilon <= 0.0:
        raise ValueError("target_epsilon must be positive and finite.")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("epsilon tolerance must be positive and finite.")

    kwargs = {
        "dataset_size": int(dataset_size),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "delta": float(delta),
        "accountant": str(accountant),
    }
    high = 1.0
    high_epsilon = epsilon_for_noise_multiplier(high, **kwargs)
    while high_epsilon > target_epsilon:
        high *= 2.0
        if high > 1_000_000.0:
            raise RuntimeError("Could not bracket a DP noise multiplier.")
        high_epsilon = epsilon_for_noise_multiplier(high, **kwargs)

    low = 0.0
    for _ in range(int(max_iterations)):
        middle = (low + high) / 2.0
        middle_epsilon = epsilon_for_noise_multiplier(middle, **kwargs)
        if middle_epsilon <= target_epsilon:
            high = middle
            high_epsilon = middle_epsilon
        else:
            low = middle
        if target_epsilon - high_epsilon <= tolerance:
            break
    if target_epsilon - high_epsilon > tolerance:
        raise RuntimeError(
            "Noise calibration did not reach the configured epsilon tolerance."
        )
    return NoiseCalibration(
        noise_multiplier=float(high), actual_epsilon=float(high_epsilon)
    )


def person_level_group_privacy_bound(
    *, row_epsilon: float, row_delta: float, max_contributions: int
) -> tuple[float, float]:
    """Conservative group-DP bound when one person contributes several rows."""
    max_contributions = int(max_contributions)
    if max_contributions <= 0:
        raise ValueError("max_contributions must be positive.")
    row_epsilon = float(row_epsilon)
    row_delta = float(row_delta)
    person_epsilon = max_contributions * row_epsilon
    try:
        multiplier = sum(
            math.exp(index * row_epsilon)
            for index in range(max_contributions)
        )
    except OverflowError:
        multiplier = math.inf
    return person_epsilon, row_delta * multiplier


def _label_key(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _label_indices(values: Any, classes: Iterable[Any]) -> np.ndarray:
    class_values = list(classes)
    class_to_index = {
        _label_key(value): index for index, value in enumerate(class_values)
    }
    labels = np.asarray(values).reshape(-1)
    missing = sorted(
        {
            _label_key(value)
            for value in labels.tolist()
            if _label_key(value) not in class_to_index
        },
        key=str,
    )
    if missing:
        raise ValueError(f"Labels are absent from the declared classes: {missing}")
    return np.asarray(
        [class_to_index[_label_key(value)] for value in labels.tolist()],
        dtype=np.int64,
    )


def _prepare_tensors(
    dataset: MethodDataset,
    standardizer,
    *,
    classes,
    device,
    dtype,
    allow_cpu_fallback: bool,
):
    import torch

    resolved_device = resolve_device(
        device, allow_cpu_fallback=allow_cpu_fallback
    )
    resolved_dtype = resolve_dtype(dtype)
    train_x = to_device_tensor(
        standardizer.transform(dataset.train_frame),
        device=resolved_device,
        dtype=resolved_dtype,
    )
    evaluation_x = to_device_tensor(
        standardizer.transform(dataset.evaluation_frame),
        device=resolved_device,
        dtype=resolved_dtype,
    )
    train_y = torch.as_tensor(
        _label_indices(dataset.train_y, classes),
        device=resolved_device,
        dtype=torch.long,
    )
    evaluation_y = torch.as_tensor(
        _label_indices(dataset.evaluation_y, classes),
        device=resolved_device,
        dtype=torch.long,
    )
    if train_x.ndim != 2 or evaluation_x.ndim != 2:
        raise ValueError("Model features must be two-dimensional.")
    if train_x.shape[1] != evaluation_x.shape[1]:
        raise ValueError("Training and evaluation feature counts disagree.")
    if not bool(torch.isfinite(train_x).all().detach().cpu()):
        raise ValueError(f"{dataset.method}: training features contain NaN/Inf.")
    if not bool(torch.isfinite(evaluation_x).all().detach().cpu()):
        raise ValueError(f"{dataset.method}: evaluation features contain NaN/Inf.")
    return train_x, train_y, evaluation_x, evaluation_y


def _build_network(
    feature_count: int,
    class_count: int,
    *,
    hidden_sizes,
    device,
    dtype,
    random_state: int,
):
    import torch
    import torch.nn as nn

    hidden_sizes = tuple(int(size) for size in hidden_sizes)
    if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
        raise ValueError("hidden_sizes must contain positive layer widths.")
    torch.manual_seed(int(random_state))
    layers: list[nn.Module] = []
    previous = int(feature_count)
    for size in hidden_sizes:
        layers.append(nn.Linear(previous, size, device=device, dtype=dtype))
        layers.append(nn.ReLU())
        previous = size
    output_count = 1 if int(class_count) == 2 else int(class_count)
    layers.append(nn.Linear(previous, output_count, device=device, dtype=dtype))
    return nn.Sequential(*layers)


def _loss(logits, targets, *, class_count: int, reduction: str):
    import torch.nn.functional as functional

    if int(class_count) == 2:
        return functional.binary_cross_entropy_with_logits(
            logits.squeeze(1), targets.to(dtype=logits.dtype), reduction=reduction
        )
    return functional.cross_entropy(logits, targets, reduction=reduction)


def _test_logloss(network, evaluation_x, evaluation_y, *, class_count: int) -> float:
    import torch

    network.eval()
    with torch.no_grad():
        value = _loss(
            network(evaluation_x),
            evaluation_y,
            class_count=class_count,
            reduction="mean",
        )
    return float(value.detach().cpu())


def train_nonprivate_sgd(
    dataset: MethodDataset,
    standardizer,
    *,
    classes,
    hidden_sizes,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    device,
    dtype,
    allow_cpu_fallback: bool,
    random_state: int,
) -> TrainingOutcome:
    """Run non-private SGD with the same Poisson schedule as DP-SGD."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _PrivacyEngine, _create_accountant, DPDataLoader = _require_opacus()
    train_x, train_y, evaluation_x, evaluation_y = _prepare_tensors(
        dataset,
        standardizer,
        classes=classes,
        device=device,
        dtype=dtype,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    class_count = len(classes)
    network = _build_network(
        train_x.shape[1],
        class_count,
        hidden_sizes=hidden_sizes,
        device=train_x.device,
        dtype=train_x.dtype,
        random_state=random_state,
    )
    base_loader = DataLoader(
        TensorDataset(train_x.detach().cpu(), train_y.detach().cpu()),
        batch_size=effective_batch_size_for_dataset(
            dataset_size=len(train_x), batch_size=batch_size
        ),
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(int(random_state)),
    )
    loader = DPDataLoader.from_data_loader(
        base_loader, batch_first=True, rand_on_empty=False
    )
    sample_rate = 1.0 / float(len(loader))
    expected_batch_size = int(len(train_x) * sample_rate)
    optimizer = torch.optim.SGD(
        network.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    network.train()
    steps = 0
    for _ in range(int(epochs)):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device=train_x.device, dtype=train_x.dtype)
            batch_y = batch_y.to(device=train_x.device)
            optimizer.zero_grad(set_to_none=True)
            loss_sum = _loss(
                network(batch_x),
                batch_y,
                class_count=class_count,
                reduction="sum",
            )
            (loss_sum / float(expected_batch_size)).backward()
            optimizer.step()
            steps += 1
    return TrainingOutcome(
        test_logloss=_test_logloss(
            network, evaluation_x, evaluation_y, class_count=class_count
        ),
        sample_rate=sample_rate,
        expected_batch_size=expected_batch_size,
        training_steps=steps,
        actual_epsilon=None,
    )


def train_private_or_clipped_sgd(
    dataset: MethodDataset,
    standardizer,
    *,
    classes,
    calibration: NoiseCalibration,
    hidden_sizes,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_grad_norm: float,
    delta: float,
    accountant: str,
    device,
    dtype,
    allow_cpu_fallback: bool,
    random_state: int,
) -> TrainingOutcome:
    """Run the shared Opacus clipping path, with optional Gaussian noise."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    PrivacyEngine, _create_accountant, _DPDataLoader = _require_opacus()
    train_x, train_y, evaluation_x, evaluation_y = _prepare_tensors(
        dataset,
        standardizer,
        classes=classes,
        device=device,
        dtype=dtype,
        allow_cpu_fallback=allow_cpu_fallback,
    )
    class_count = len(classes)
    network = _build_network(
        train_x.shape[1],
        class_count,
        hidden_sizes=hidden_sizes,
        device=train_x.device,
        dtype=train_x.dtype,
        random_state=random_state,
    )
    base_loader = DataLoader(
        TensorDataset(train_x.detach().cpu(), train_y.detach().cpu()),
        batch_size=effective_batch_size_for_dataset(
            dataset_size=len(train_x), batch_size=batch_size
        ),
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(int(random_state)),
    )
    optimizer = torch.optim.SGD(
        network.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    privacy_engine = PrivacyEngine(accountant=str(accountant))
    network, optimizer, loader = privacy_engine.make_private(
        module=network,
        optimizer=optimizer,
        data_loader=base_loader,
        noise_multiplier=float(calibration.noise_multiplier),
        max_grad_norm=float(max_grad_norm),
        batch_first=True,
        loss_reduction="mean",
        poisson_sampling=True,
    )
    sample_rate = 1.0 / float(len(loader))
    expected_batch_size = int(len(train_x) * sample_rate)

    network.train()
    steps = 0
    for _ in range(int(epochs)):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device=train_x.device, dtype=train_x.dtype)
            batch_y = batch_y.to(device=train_x.device)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(
                network(batch_x),
                batch_y,
                class_count=class_count,
                reduction="mean",
            )
            loss.backward()
            optimizer.step()
            steps += 1

    actual_epsilon = None
    if float(calibration.noise_multiplier) > 0.0:
        actual_epsilon = float(privacy_engine.get_epsilon(float(delta)))
    return TrainingOutcome(
        test_logloss=_test_logloss(
            network, evaluation_x, evaluation_y, class_count=class_count
        ),
        sample_rate=sample_rate,
        expected_batch_size=expected_batch_size,
        training_steps=steps,
        actual_epsilon=actual_epsilon,
    )
