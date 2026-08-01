"""Plot Exp-33 execution time, Utility MAE, and rho MAE.

Each input row represents one method-selected action at one seed and iteration.
The producer records its actual Utility Delta and rho gaps to the same
approximate oracle, together with the method's measured iteration runtime.
Each seed is summarized over all of its own iterations before seeds are given
equal weight.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")


REQUIRED_COLUMNS = (
    "seed",
    "iteration",
    "method",
    "actual_utility_gap_to_oracle",
    "ratio_abs_error",
    "time_seconds",
)
METHODS = ("trim", "if", "lga", "retrain")
METHOD_LABELS = ("TRIM", "IF", "LGA", "Retrain")


def _integer_token(value: object, *, field: str, location: str) -> int:
    if value is None or not str(value).strip():
        raise ValueError(f"Missing {field!r} at {location}.")
    try:
        number = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric {field!r} at {location}: {value!r}"
        ) from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(
            f"{field!r} must be a finite integer at {location}: {value!r}"
        )
    return int(number)


def _number(value: object, *, field: str, location: str) -> float:
    if value is None or not str(value).strip():
        raise ValueError(f"Missing {field!r} at {location}.")
    try:
        number = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric {field!r} at {location}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field!r} at {location}: {value!r}")
    return number


def _read_rows(paths: list[str]) -> dict[tuple[int, int, str], dict[str, float]]:
    points: dict[tuple[int, int, str], dict[str, float]] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Exp-33 input must be a CSV file: {path}")
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no header: {path}")
            if any(not str(field).strip() for field in reader.fieldnames):
                raise ValueError(f"CSV file has a blank header: {path}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"CSV file has duplicate headers: {path}")
            missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
            if missing:
                raise ValueError(
                    f"CSV file is missing required columns {missing}: {path}"
                )
            row_count = 0
            for row_number, row in enumerate(reader, start=2):
                row_count += 1
                location = f"{path}:{row_number}"
                if None in row:
                    raise ValueError(
                        f"Row has more values than headers at {location}."
                    )
                seed = _integer_token(row["seed"], field="seed", location=location)
                iteration = _integer_token(
                    row["iteration"], field="iteration", location=location
                )
                if iteration < 0:
                    raise ValueError(
                        f"iteration must be non-negative at {location}."
                    )
                method = "" if row["method"] is None else row["method"].strip()
                if method not in METHODS:
                    raise ValueError(
                        f"Method at {location} must be one of {METHODS}; "
                        f"found {method!r}."
                    )
                utility_error = _number(
                    row["actual_utility_gap_to_oracle"],
                    field="actual_utility_gap_to_oracle",
                    location=location,
                )
                rho_error = _number(
                    row["ratio_abs_error"],
                    field="ratio_abs_error",
                    location=location,
                )
                elapsed = _number(
                    row["time_seconds"], field="time_seconds", location=location
                )
                if utility_error < 0.0 or rho_error < 0.0 or elapsed < 0.0:
                    raise ValueError(
                        f"Errors and runtime must be non-negative at {location}."
                    )
                key = seed, iteration, method
                if key in points:
                    raise ValueError(
                        "Duplicate seed/iteration/method row across inputs: "
                        f"{key}"
                    )
                points[key] = {
                    "utility_error": utility_error,
                    "rho_error": rho_error,
                    "time": elapsed,
                }
            if row_count == 0:
                raise ValueError(f"Input CSV contains no data rows: {path}")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    points = _read_rows(args.input)
    seeds = sorted({seed for seed, _, _ in points})
    iterations_by_seed = {
        seed: {
            iteration
            for point_seed, iteration, _ in points
            if point_seed == seed
        }
        for seed in seeds
    }
    expected = {
        (seed, iteration, method)
        for seed in seeds
        for iteration in iterations_by_seed[seed]
        for method in METHODS
    }
    if set(points) != expected:
        missing = sorted(expected - set(points))
        extra = sorted(set(points) - expected)
        raise ValueError(
            "Incomplete seed/iteration/method matrix; "
            f"missing={missing}, extra={extra}."
        )

    relative_times = {method: [] for method in METHODS}
    utility_mae = {method: [] for method in METHODS}
    rho_mae = {method: [] for method in METHODS}
    for seed in seeds:
        iterations = iterations_by_seed[seed]
        total_times = {
            method: math.fsum(
                points[(seed, iteration, method)]["time"]
                for iteration in iterations
            )
            for method in METHODS
        }
        if total_times["trim"] <= 0.0:
            raise ValueError(
                f"Seed {seed} has non-positive total TRIM runtime."
            )
        for method in METHODS:
            relative_times[method].append(
                total_times[method] / total_times["trim"]
            )
            utility_mae[method].append(statistics.fmean(
                points[(seed, iteration, method)]["utility_error"]
                for iteration in iterations
            ))
            rho_mae[method].append(statistics.fmean(
                points[(seed, iteration, method)]["rho_error"]
                for iteration in iterations
            ))

    plotted_times = [
        statistics.fmean(relative_times[method]) for method in METHODS
    ]
    plotted_utility = [
        statistics.fmean(utility_mae[method]) for method in METHODS
    ]
    plotted_rho = [statistics.fmean(rho_mae[method]) for method in METHODS]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 7.2))
    for axis, values, ylabel in zip(
        axes,
        (plotted_times, plotted_utility, plotted_rho),
        ("Execution time / TRIM", "Utility MAE", r"$\rho$ MAE"),
    ):
        axis.bar(METHOD_LABELS, values)
        axis.set_ylabel(ylabel)
    fig.tight_layout()

    output = Path(args.output).expanduser()
    if not output.suffix:
        raise ValueError("--output must include a Matplotlib-supported suffix.")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
