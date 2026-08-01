"""Plot Figure 5h from seed-level enumeration run summaries.

Runtime is normalized to TRIM within each seed. Runtime ratios and leak values
are then averaged across seeds, giving every seed equal weight.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

try:
    from .plot_exp3_runtime import _number, _read_rows, _seed_token, _text
except ImportError:  # Direct script execution.
    from plot_exp3_runtime import _number, _read_rows, _seed_token, _text  # type: ignore


METHODS = ("TRIM", "Sample", "KMeans", "IL", "RandomSplit")


def _boolean(value, *, field, location):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Invalid {field} at {location}: {value!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="pubcov")
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--tolerance", required=True, type=float)
    parser.add_argument("--seeds", required=True, nargs="+")
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not math.isfinite(args.tolerance) or args.tolerance < 0.0:
        raise ValueError("--tolerance must be finite and non-negative.")
    seeds = [
        _seed_token(value, field="seed", location="CLI")
        for value in args.seeds
    ]
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds contains duplicates.")
    required = {
        "experiment_kind",
        "method",
        "dataset",
        "model_name",
        "tolerance",
        "random_state",
        "algorithm_time_seconds",
        "original_leak_k",
        "final_leak_k",
        "utility_constraint_met",
    }
    selected = {}
    contributions = {Path(path).expanduser(): 0 for path in args.input}
    for path, row_number, row in _read_rows(args.input):
        location = f"{path}:{row_number}"
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Missing fields at {location}: {missing}.")
        if (
            _text(row["experiment_kind"], field="experiment_kind", location=location)
            != "exp6a1_enumeration"
            or _text(row["dataset"], field="dataset", location=location)
            != args.dataset
            or _text(row["model_name"], field="model_name", location=location)
            != args.model
            or _number(row["tolerance"], field="tolerance", location=location)
            != args.tolerance
        ):
            continue
        row_seed = _seed_token(
            row["random_state"], field="random_state", location=location
        )
        if row_seed not in seeds:
            continue
        method = _text(row["method"], field="method", location=location)
        if method not in METHODS:
            raise ValueError(
                f"Unexpected Figure 5h method {method!r} at {location}."
            )
        key = row_seed, method
        if key in selected:
            raise ValueError(
                f"Duplicate Figure 5h seed/method row {key!r}."
            )
        if not _boolean(
            row["utility_constraint_met"],
            field="utility_constraint_met",
            location=location,
        ):
            raise ValueError(
                f"Method {method!r} does not satisfy the utility constraint."
            )
        elapsed = _number(
            row["algorithm_time_seconds"],
            field="algorithm_time_seconds",
            location=location,
        )
        original_k = _number(
            row["original_leak_k"], field="original_leak_k", location=location
        )
        final_k = _number(
            row["final_leak_k"], field="final_leak_k", location=location
        )
        if elapsed <= 0.0 or original_k <= 0.0 or final_k <= 0.0:
            raise ValueError(
                f"Runtime and K values must be positive at {location}."
            )
        selected[key] = {
            "time": elapsed,
            "leak": math.log(original_k) - math.log(final_k),
        }
        contributions[path] += 1

    empty_inputs = [str(path) for path, count in contributions.items() if count == 0]
    if empty_inputs:
        raise ValueError(
            "Every explicit input must contribute selected rows; none found in "
            f"{empty_inputs}."
        )
    expected = {
        (seed, method) for seed in seeds for method in METHODS
    }
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        extra = sorted(set(selected) - expected)
        raise ValueError(
            "Incomplete Figure 5h seed/method matrix; "
            f"missing={missing}, extra={extra}."
        )

    relative_times_by_method = {method: [] for method in METHODS}
    leaks_by_method = {method: [] for method in METHODS}
    for seed in seeds:
        trim_time = selected[(seed, "TRIM")]["time"]
        for method in METHODS:
            relative_times_by_method[method].append(
                selected[(seed, method)]["time"] / trim_time
            )
            leaks_by_method[method].append(
                selected[(seed, method)]["leak"]
            )
    relative_times = [
        statistics.fmean(relative_times_by_method[method])
        for method in METHODS
    ]
    leaks = [
        statistics.fmean(leaks_by_method[method])
        for method in METHODS
    ]
    positions = list(range(len(METHODS)))

    fig, (time_axis, leak_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(5.5, 5.0),
        gridspec_kw={"hspace": 0.20},
    )
    time_axis.bar(positions, relative_times, color="black", width=0.55)
    leak_axis.bar(positions, leaks, color="black", width=0.55)
    time_axis.set_ylabel("Execution time (x)")
    leak_axis.set_ylabel("leak")
    leak_axis.set_xticks(positions, METHODS)
    time_axis.set_yscale("log")
    time_axis.axhline(1.0, color="0.35", linestyle="--", linewidth=1.0)
    for axis, values in ((time_axis, relative_times), (leak_axis, leaks)):
        axis.grid(True, axis="y", alpha=0.25)
        for position, value in zip(positions, values):
            axis.annotate(
                f"{value:.2f}",
                (position, value),
                xytext=(0, 4 if value >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
            )

    output = Path(args.output).expanduser()
    if not output.suffix:
        raise ValueError("--output must include a Matplotlib-supported suffix.")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {output}")
    if args.dpi is not None and args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    save_options = {} if args.dpi is None else {"dpi": args.dpi}
    fig.savefig(output, **save_options)
    plt.close(fig)


if __name__ == "__main__":
    main()
