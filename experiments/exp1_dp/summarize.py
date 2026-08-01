"""Paired summaries for the configured Experiment 1 seeds."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

from experiments.common import write_json


REQUIRED_METHODS = ("original", "mondrian", "trim")
REQUIRED_TRAINING_MODES = ("sgd", "clipped_sgd", "dp_sgd")


def _mean_sd(values: list[float]) -> tuple[float, float | None]:
    if not values:
        raise ValueError("Cannot summarize an empty value list.")
    return mean(values), stdev(values) if len(values) > 1 else None


def _one_sided_upper_bound(
    values: list[float], *, confidence_level: float
) -> float:
    if len(values) < 2:
        raise ValueError("A paired Student-t bound requires at least two seeds.")
    try:
        from scipy.stats import t
    except ImportError as exc:
        raise RuntimeError(
            "Experiment 1 paired summaries require the optional `scipy` "
            "package. Install the project environment before summarizing."
        ) from exc
    standard_error = stdev(values) / math.sqrt(len(values))
    return mean(values) + float(
        t.ppf(confidence_level, df=len(values) - 1)
    ) * standard_error


def _as_float(value: Any, *, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}.") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be finite, got {value!r}.")
    return converted


def _row_key(row: dict[str, Any]) -> tuple[int, str, str, float | None]:
    mode = str(row.get("training_mode"))
    x_value = (
        _as_float(row.get("x"), field="x") if mode == "dp_sgd" else None
    )
    return int(row["seed"]), str(row["method"]), mode, x_value


def summarize_rows(
    rows: Iterable[dict[str, Any]],
    *,
    noninferiority_margin: float,
    confidence_level: float,
    expected_seeds: Iterable[int],
    expected_x_values: Iterable[float],
) -> dict[str, Any]:
    """Validate the declared design and compute paired claims."""
    rows = [dict(row) for row in rows]
    expected_seeds = tuple(int(seed) for seed in expected_seeds)
    expected_x_values = tuple(float(value) for value in expected_x_values)
    if len(expected_seeds) < 2 or len(set(expected_seeds)) != len(expected_seeds):
        raise ValueError(
            "Experiment 1 requires at least two distinct seeds for its "
            "paired Student-t summary."
        )
    if not expected_x_values or len(set(expected_x_values)) != len(
        expected_x_values
    ):
        raise ValueError("x values must be distinct and non-empty.")
    margin = float(noninferiority_margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("The noninferiority margin must be finite and nonnegative.")
    confidence_level = float(confidence_level)
    if not math.isfinite(confidence_level) or not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and between 0.5 and 1.")

    observed_seeds = {int(row["seed"]) for row in rows}
    if observed_seeds != set(expected_seeds):
        raise ValueError(
            f"Observed seeds {sorted(observed_seeds)} do not match the "
            f"declared seeds {sorted(expected_seeds)}."
        )
    observed_methods = {str(row["method"]) for row in rows}
    if observed_methods != set(REQUIRED_METHODS):
        raise ValueError(
            f"Experiment 1 methods must be exactly {list(REQUIRED_METHODS)}."
        )
    observed_modes = {str(row["training_mode"]) for row in rows}
    if observed_modes != set(REQUIRED_TRAINING_MODES):
        raise ValueError(
            "Experiment 1 requires SGD, clipped-SGD, and DP-SGD rows."
        )

    rows_by_key: dict[tuple[int, str, str, float | None], dict[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        if key in rows_by_key:
            raise ValueError(f"Duplicate Experiment 1 result row: {key}")
        rows_by_key[key] = row

    for seed in expected_seeds:
        for method in REQUIRED_METHODS:
            for mode in ("sgd", "clipped_sgd"):
                key = (seed, method, mode, None)
                if key not in rows_by_key:
                    raise ValueError(f"Missing Experiment 1 result row: {key}")
            for x_value in expected_x_values:
                key = (seed, method, "dp_sgd", x_value)
                if key not in rows_by_key:
                    raise ValueError(f"Missing Experiment 1 result row: {key}")

    expected_count = len(expected_seeds) * len(REQUIRED_METHODS) * (
        2 + len(expected_x_values)
    )
    if len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} raw result rows, found {len(rows)}."
        )

    grouped_modes: dict[tuple[str, str, float | None], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        key = _row_key(row)
        grouped_modes[(key[2], key[1], key[3])].append(row)
    mode_summaries = []
    for (mode, method, x_value), group in sorted(
        grouped_modes.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            -1.0 if item[0][2] is None else item[0][2],
        ),
    ):
        losses = [_as_float(row["test_logloss"], field="test_logloss") for row in group]
        loss_mean, loss_sd = _mean_sd(losses)
        mode_summaries.append(
            {
                "training_mode": mode,
                "method": method,
                "x": x_value,
                "n_seeds": len(group),
                "seeds": sorted(int(row["seed"]) for row in group),
                "mean_test_logloss": loss_mean,
                "sd_test_logloss": loss_sd,
            }
        )

    paired_runs = []
    paired_by_key = {}
    matching_fields = (
        "optimizer",
        "learning_rate",
        "weight_decay",
        "epochs",
        "requested_batch_size",
        "poisson_sampling",
        "model_hidden_sizes",
        "evaluation_protocol",
    )
    for seed in expected_seeds:
        for method in REQUIRED_METHODS:
            sgd = rows_by_key[(seed, method, "sgd", None)]
            for x_value in expected_x_values:
                private = rows_by_key[(seed, method, "dp_sgd", x_value)]
                for field in matching_fields:
                    if private.get(field) != sgd.get(field):
                        raise ValueError(
                            "DP-SGD and SGD are not matched for "
                            f"seed={seed}, method={method}: {field}."
                        )
                paired = {
                    "seed": seed,
                    "method": method,
                    "x": x_value,
                    "target_epsilon": _as_float(
                        private["target_epsilon"], field="target_epsilon"
                    ),
                    "actual_epsilon": _as_float(
                        private["actual_epsilon"], field="actual_epsilon"
                    ),
                    "nonprivate_logloss": _as_float(
                        sgd["test_logloss"], field="test_logloss"
                    ),
                    "private_logloss": _as_float(
                        private["test_logloss"], field="test_logloss"
                    ),
                    "dp_penalty": _as_float(
                        private["test_logloss"], field="test_logloss"
                    )
                    - _as_float(sgd["test_logloss"], field="test_logloss"),
                    "row_count": int(private["row_count"]),
                    "unique_row_count": int(private["unique_row_count"]),
                    "published_row_count": int(private["published_row_count"]),
                    "final_min_k": int(private["final_min_k"]),
                    "noise_multiplier": _as_float(
                        private["noise_multiplier"], field="noise_multiplier"
                    ),
                    "sample_rate": _as_float(
                        private["sample_rate"], field="sample_rate"
                    ),
                    "training_steps": int(private["training_steps"]),
                }
                paired_runs.append(paired)
                paired_by_key[(x_value, seed, method)] = paired

    dp_summaries = []
    for x_value in expected_x_values:
        for method in REQUIRED_METHODS:
            group = [
                paired_by_key[(x_value, seed, method)]
                for seed in expected_seeds
            ]
            private_losses = [row["private_logloss"] for row in group]
            penalties = [row["dp_penalty"] for row in group]
            private_mean, private_sd = _mean_sd(private_losses)
            penalty_mean, penalty_sd = _mean_sd(penalties)
            summary: dict[str, Any] = {
                "x": x_value,
                "target_epsilon": group[0]["target_epsilon"],
                "method": method,
                "n_seeds": len(group),
                "seeds": list(expected_seeds),
                "row_count": group[0]["row_count"],
                "unique_row_count": group[0]["unique_row_count"],
                "published_row_count": group[0]["published_row_count"],
                "final_min_k": group[0]["final_min_k"],
                "mean_private_logloss": private_mean,
                "sd_private_logloss": private_sd,
                "mean_dp_penalty": penalty_mean,
                "sd_dp_penalty": penalty_sd,
            }
            if method != "mondrian":
                references = [
                    paired_by_key[(x_value, seed, "mondrian")]
                    for seed in expected_seeds
                ]
                private_gaps = [
                    row["private_logloss"] - reference["private_logloss"]
                    for row, reference in zip(group, references)
                ]
                penalty_interactions = [
                    row["dp_penalty"] - reference["dp_penalty"]
                    for row, reference in zip(group, references)
                ]
                private_gap_ucb = _one_sided_upper_bound(
                    private_gaps, confidence_level=confidence_level
                )
                interaction_ucb = _one_sided_upper_bound(
                    penalty_interactions,
                    confidence_level=confidence_level,
                )
                fewer_unique_rows = all(
                    row["unique_row_count"] < reference["unique_row_count"]
                    for row, reference in zip(group, references)
                )
                summary["paired_vs_mondrian"] = {
                    "private_logloss_gaps": private_gaps,
                    "mean_private_logloss_gap": mean(private_gaps),
                    "one_sided_ucb_private_gap": private_gap_ucb,
                    "confidence_level": confidence_level,
                    "noninferiority_margin": margin,
                    "fewer_unique_rows": fewer_unique_rows,
                    "noninferior_private_logloss": private_gap_ucb <= margin,
                    "noninferior_with_fewer_rows": (
                        private_gap_ucb <= margin and fewer_unique_rows
                    ),
                    "dp_penalty_interactions": penalty_interactions,
                    "mean_dp_penalty_interaction": mean(penalty_interactions),
                    "one_sided_ucb_dp_penalty_interaction": interaction_ucb,
                    "smaller_dp_penalty": interaction_ucb < 0.0,
                }
            dp_summaries.append(summary)

    return {
        "protocol": {
            "reference_method": "mondrian",
            "claim_1": (
                "smaller method-relative DP penalty than full-release Mondrian"
            ),
            "claim_2": (
                "noninferior private logloss with fewer unique training rows"
            ),
            "noninferiority_margin": margin,
            "confidence_level": confidence_level,
            "confidence_bound": "paired one-sided Student-t upper bound",
            "seed_count": len(expected_seeds),
        },
        "expected_seeds": list(expected_seeds),
        "expected_x_values": list(expected_x_values),
        "raw_result_count": len(rows),
        "mode_summaries": mode_summaries,
        "paired_runs": paired_runs,
        "dp_summaries": dp_summaries,
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON row list: {path}")
    return [dict(row) for row in payload]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute the declared Experiment 1 paired summary."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--x-values", type=float, nargs="+", required=True)
    parser.add_argument(
        "--noninferiority-margin", type=float, required=True
    )
    parser.add_argument("--confidence-level", type=float, required=True)
    args = parser.parse_args(argv)
    result = summarize_rows(
        _read_rows(args.input),
        noninferiority_margin=args.noninferiority_margin,
        confidence_level=args.confidence_level,
        expected_seeds=args.expected_seeds,
        expected_x_values=args.x_values,
    )
    write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
