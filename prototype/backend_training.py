"""Frozen feature standardization for model training and utility estimation.

The estimator, LGA, and Hessian utility estimate share one feature space.
The standardizer therefore computes its statistics once from the generalized
training encoding and reuses them throughout the pipeline.

Standardization is applied when an encoding is consumed by a model, LGA,
Hessian estimation, or loss computation. Privacy measurement, row selection,
and published encodings continue to use the unstandardized encoded values.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch


class FeatureStandardizer:
    """Store frozen statistics and z-score the numeric columns on demand.

    One-hot columns, whose names contain ``=``, pass through unchanged. The
    same statistics are reused for model fitting, prediction, LGA, Hessian
    estimation, and loss computation.
    """

    def __init__(
        self,
        numeric_column_names: Sequence[str],
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.numeric_column_names = tuple(numeric_column_names)
        self.mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        self.std = np.asarray(std, dtype=np.float64).reshape(-1)
        if self.mean.shape != self.std.shape:
            raise ValueError("mean and std must have the same shape.")
        if self.mean.shape[0] != len(self.numeric_column_names):
            raise ValueError(
                "mean/std length must match the numeric column count."
            )
        # Map zero-variance columns to a unit scale.
        self._safe_std = np.where(self.std == 0.0, 1.0, self.std)

    @property
    def numeric_column_count(self) -> int:
        return len(self.numeric_column_names)

    def _numeric_positions(self, columns) -> list[int]:
        """Column indices of the numeric attributes inside an arbitrary column
        indexable by name (DataFrame) or positional header (ndarray/tensor)."""
        column_list = list(columns)
        positions = []
        for name in self.numeric_column_names:
            if name not in column_list:
                raise ValueError(
                    f"Numeric column {name!r} not found in encode columns."
                )
            positions.append(column_list.index(name))
        return positions

    def transform(self, X: Any) -> Any:
        """Return a copy of ``X`` with the numeric columns z-scored.

        Supports pandas.DataFrame, numpy.ndarray and torch.Tensor inputs and
        returns the corresponding input type.
        """
        # --- torch.Tensor path used by run_trim_pipeline. -------------------
        if isinstance(X, torch.Tensor):
            # Numeric columns are positional and match the encode's column order
            # exactly (the encode is built deterministically by dataloader, and
            # this standardizer was fit on a column order identical to the one
            # X carries). We look them up by column-name header when available
            # and otherwise use the stored positional order.
            mean_t = torch.as_tensor(
                self.mean, device=X.device, dtype=X.dtype
            )
            std_t = torch.as_tensor(
                self._safe_std, device=X.device, dtype=X.dtype
            )
            out = X.clone()
            # Numeric attributes always lead the encode column block in the
            # dataloader (numeric_attributes are emitted before categoricals),
            # so they occupy the first len(numeric) columns in order. Verify the
            # count matches to avoid silent mis-indexing.
            if out.shape[1] < self.numeric_column_count:
                raise ValueError(
                    "Tensor has fewer columns than registered numeric attributes."
                )
            positions = slice(0, self.numeric_column_count)
            out[:, positions] = (
                (out[:, positions] - mean_t.unsqueeze(0)) / std_t.unsqueeze(0)
            )
            return out

        # --- numpy.ndarray path ---
        if isinstance(X, np.ndarray):
            out = X.astype(np.float64, copy=True)
            positions = slice(0, self.numeric_column_count)
            if out.shape[1] < self.numeric_column_count:
                raise ValueError(
                    "Array has fewer columns than registered numeric attributes."
                )
            out[:, positions] = (
                out[:, positions] - self.mean[np.newaxis, :]
            ) / self._safe_std[np.newaxis, :]
            return out

        # --- pandas.DataFrame path ---
        columns = list(X.columns)
        positions = self._numeric_positions(columns)
        out = X.copy()
        for pos, name in zip(positions, self.numeric_column_names):
            col_idx = columns.index(name)
            out.iloc[:, col_idx] = (
                (X.iloc[:, col_idx].astype(np.float64) - self.mean[pos])
                / self._safe_std[pos]
            )
        return out


class PassthroughStandardizer:
    """Identity transform for datasets without a standardization policy."""

    numeric_column_count = 0

    def transform(self, X: Any) -> Any:
        return X


def _numeric_mean_std(encode, numeric_attributes) -> tuple[np.ndarray, np.ndarray]:
    """Compute population (mean, std) over the numeric columns of ``encode``.

    Works on a pandas DataFrame (columns selected by name) or a numpy array
    (numeric attributes assumed to lead the column block, matching the
    dataloader's emission order).
    """
    import pandas as pd

    if isinstance(encode, pd.DataFrame):
        cols = [a for a in numeric_attributes if a in encode.columns]
        if not cols:
            return np.empty(0), np.empty(0)
        block = encode.loc[:, cols].astype(np.float64).to_numpy()
        mean = block.mean(axis=0)
        std = block.std(axis=0)
        return mean, std

    arr = np.asarray(encode, dtype=np.float64)
    n_numeric = len(numeric_attributes)
    if arr.ndim != 2 or arr.shape[1] < n_numeric:
        return np.empty(0), np.empty(0)
    block = arr[:, :n_numeric]
    return block.mean(axis=0), block.std(axis=0)


# Datasets whose numeric model features are standardized before model fitting.
# Keys correspond to ``data_loader.dataset_stem``; unlisted datasets use
# ``PassthroughStandardizer``.
_STANDARDIZED_STEMS = {
    "ACSIncome",
    "acs_public_coverage",
    "bank_marketing",
    "bng_credit_g",
    "diabetes_130_us_hospitals",
}


def fit_standardizer(
    full_train_encode,
    data_loader,
) -> FeatureStandardizer | PassthroughStandardizer:
    """Dispatch on ``data_loader.dataset_stem`` and fit a frozen standardizer on
    the numeric columns of ``full_train_encode``.

    Only ``data_loader.numeric_attributes`` are standardized; every one-hot
    column is left untouched. The statistics are computed once and frozen for
    the rest of the pipeline.

    Returns a :class:`PassthroughStandardizer` when the dataset has no matching
    standardization policy.
    """
    stem = getattr(data_loader, "dataset_stem", None)
    numeric_attrs = tuple(getattr(data_loader, "numeric_attributes", ()) or ())
    if stem not in _STANDARDIZED_STEMS or not numeric_attrs:
        return PassthroughStandardizer()

    # Keep only the numeric attributes actually present in the encode columns
    # (a subset of numeric_attributes could be absent if a future loader drops
    # one; we standardize what is there).
    try:
        columns = list(full_train_encode.columns)
    except AttributeError:
        columns = None
    if columns is not None:
        present_numeric = tuple(a for a in numeric_attrs if a in columns)
    else:
        present_numeric = numeric_attrs

    if not present_numeric:
        return PassthroughStandardizer()

    mean, std = _numeric_mean_std(full_train_encode, present_numeric)
    if mean.size == 0:
        return PassthroughStandardizer()

    return FeatureStandardizer(present_numeric, mean, std)
