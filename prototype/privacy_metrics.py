"""Individual disclosure risk on a fixed original population."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def individual_tail_risk_stats(population_ids, released_per_record_k):
    """P99 of log(1/k_i), with absent individuals assigned log risk -inf.

    Use the empirical inverse CDF (rank ceil(.99*N)), avoiding interpolation
    between finite risk and -inf. Population IDs identify D, not only S.
    """
    population = pd.Index(population_ids)
    sizes = pd.Series(released_per_record_k, dtype=np.float64)
    if population.empty or not population.is_unique or not sizes.index.is_unique:
        raise ValueError("Population must be nonempty and row IDs must be unique.")
    if not sizes.index.isin(population).all():
        raise ValueError("Released row IDs must belong to the original population.")
    values = sizes.to_numpy()
    if np.any(~np.isfinite(values)) or np.any(values < 1) or np.any(values != np.floor(values)):
        raise ValueError("Released equivalence-class sizes must be finite positive integers.")
    log_risk = np.full(len(population), -np.inf, dtype=np.float64)
    log_risk[population.get_indexer(sizes.index)] = -np.log(values)
    rank = math.ceil(0.99 * len(population)) - 1
    p99 = float(np.partition(log_risk, rank)[rank])
    return {
        "tail_risk_p99": p99,
        "tail_risk_percentile": 99.0,
        "tail_risk_percentile_method": "inverted_cdf",
        "tail_risk_population": "all_loaded_original_rows",
        "tail_risk_population_size": len(population),
        "tail_risk_released_population_size": len(sizes),
        "tail_risk_semantics": "P99_i_in_D(log(1/k_i) if i in S else -inf)",
    }
