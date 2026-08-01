"""Plot the three Exp-32 ranking panels from explicit candidate CSV rows.

Each candidate row represents one measured candidate in one iteration and one
seed.  The table is wide: every row contains the scores for all four methods,
so all methods are evaluated on exactly the same candidate pool.  Larger
scores and larger ``exact_utility_delta`` values are ranked first.

For each seed, HitRate@K is the fraction of iterations whose exact best
candidate appears in a method's predicted Top-K.  Recall@K is the mean Top-K
overlap with the exact Top-K.  Runtime is read from a separate table with one
row per seed, iteration, and method.  This records the complete scoring time
once, including candidates outside the measured common pool.  Runtime is
summed within each seed and divided by its corresponding TRIM sum.  The
displayed bars are arithmetic means of these per-seed quantities, giving every
seed equal weight even when seeds terminate after different iterations.

Example::

    python -m plotting.plot_exp32_ranking \
      --input SEED_42.csv SEED_43.csv --input SEED_44.csv \
      --runtime-input SEED_42_RUNTIME.csv SEED_43_RUNTIME.csv \
      --runtime-input SEED_44_RUNTIME.csv \
      --output exp32_ranking.pdf --top-k TOP_K
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")


METHODS = (
    ("trim", "TRIM", "mlp_lga_score"),
    ("if", "IF", "yang_if_distance_score"),
    ("random", "Random", "random_score"),
    ("retrain", "Retrain", "exact_utility_delta"),
)

IDENTITY_COLUMNS = (
    "seed",
    "iteration",
    "candidate_type",
    "candidate_id",
)
CANDIDATE_TYPES = {"retention_class", "vertical_refinement"}

REQUIRED_COLUMNS = IDENTITY_COLUMNS + tuple(
    score_column for _, _, score_column in METHODS
)
RUNTIME_REQUIRED_COLUMNS = (
    "seed",
    "iteration",
    "method",
    "time_seconds",
    "candidate_count",
    "timing_scope",
)
EXPECTED_TIMING_SCOPES = {
    "trim": "all_ranked_candidates_including_setup",
    "if": "all_ranked_candidates_including_setup",
    "random": "all_ranked_candidates_including_setup",
    "retrain": "all_exact_evaluation_attempts_in_common_pool",
}


def _text(value: object, *, field: str, location: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"Missing {field!r} at {location}.")
    return str(value).strip()


def _integer(value: object, *, field: str, location: str) -> int:
    text = _text(value, field=field, location=location)
    try:
        number = int(text)
    except ValueError as exc:
        raise ValueError(
            f"Non-integer {field!r} at {location}: {value!r}"
        ) from exc
    return number


def _finite_number(value: object, *, field: str, location: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Boolean {field!r} at {location} is not numeric data.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Non-numeric {field!r} at {location}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field!r} at {location}: {value!r}")
    return number


def _read_rows(input_groups: list[list[str]]) -> dict[tuple, dict[str, float]]:
    rows = {}

    for raw_path in (value for group in input_groups for value in group):
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Exp-32 input must be a CSV file: {path}")

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            if any(not str(field).strip() for field in reader.fieldnames):
                raise ValueError(f"CSV has a blank header: {path}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"CSV has duplicate headers: {path}")
            missing_columns = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
            if missing_columns:
                raise ValueError(
                    f"CSV is missing required columns {missing_columns}: {path}"
                )

            file_row_count = 0
            for row_number, row in enumerate(reader, start=2):
                file_row_count += 1
                location = f"{path}:{row_number}"
                if None in row:
                    raise ValueError(
                        f"Row at {location} has more values than the CSV header."
                    )

                seed = _integer(row["seed"], field="seed", location=location)
                iteration = _integer(
                    row["iteration"], field="iteration", location=location
                )
                if iteration < 0:
                    raise ValueError(
                        f"Negative iteration at {location}: {iteration!r}"
                    )
                candidate_type = _text(
                    row["candidate_type"],
                    field="candidate_type",
                    location=location,
                )
                if candidate_type not in CANDIDATE_TYPES:
                    raise ValueError(
                        f"candidate_type at {location} must be one of "
                        f"{sorted(CANDIDATE_TYPES)}; found {candidate_type!r}."
                    )
                candidate_id = _text(
                    row["candidate_id"], field="candidate_id", location=location
                )
                key = seed, iteration, candidate_type, candidate_id
                if key in rows:
                    raise ValueError(
                        "Duplicate seed/iteration/candidate key "
                        f"{key!r} at {location}."
                    )

                values = {}
                for _, _, score_column in METHODS:
                    values[score_column] = _finite_number(
                        row[score_column], field=score_column, location=location
                    )
                rows[key] = values

            if file_row_count == 0:
                raise ValueError(f"Input CSV contains no data rows: {path}")

    if not rows:
        raise ValueError("No candidate rows were supplied.")
    return rows


def _read_runtime_rows(input_groups: list[list[str]]) -> dict[tuple, dict]:
    rows = {}
    valid_methods = {method for method, _, _ in METHODS}

    for raw_path in (value for group in input_groups for value in group):
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Runtime CSV does not exist: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Exp-32 runtime input must be a CSV file: {path}")

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Runtime CSV has no header: {path}")
            if any(not str(field).strip() for field in reader.fieldnames):
                raise ValueError(f"Runtime CSV has a blank header: {path}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"Runtime CSV has duplicate headers: {path}")
            missing_columns = sorted(
                set(RUNTIME_REQUIRED_COLUMNS) - set(reader.fieldnames)
            )
            if missing_columns:
                raise ValueError(
                    "Runtime CSV is missing required columns "
                    f"{missing_columns}: {path}"
                )

            file_row_count = 0
            for row_number, row in enumerate(reader, start=2):
                file_row_count += 1
                location = f"{path}:{row_number}"
                if None in row:
                    raise ValueError(
                        f"Row at {location} has more values than the CSV header."
                    )
                seed = _integer(row["seed"], field="seed", location=location)
                iteration = _integer(
                    row["iteration"], field="iteration", location=location
                )
                if iteration < 0:
                    raise ValueError(
                        f"Negative iteration at {location}: {iteration!r}"
                    )
                method = _text(row["method"], field="method", location=location)
                if method not in valid_methods:
                    raise ValueError(
                        f"method at {location} must be one of "
                        f"{sorted(valid_methods)}; found {method!r}."
                    )
                elapsed = _finite_number(
                    row["time_seconds"],
                    field="time_seconds",
                    location=location,
                )
                if elapsed < 0.0:
                    raise ValueError(
                        f"Negative time_seconds at {location}: {elapsed!r}"
                    )
                candidate_count = _integer(
                    row["candidate_count"],
                    field="candidate_count",
                    location=location,
                )
                if candidate_count < 0:
                    raise ValueError(
                        f"Negative candidate_count at {location}: "
                        f"{candidate_count!r}"
                    )
                timing_scope = _text(
                    row["timing_scope"],
                    field="timing_scope",
                    location=location,
                )
                expected_scope = EXPECTED_TIMING_SCOPES[method]
                if timing_scope != expected_scope:
                    raise ValueError(
                        f"timing_scope for {method!r} at {location} must be "
                        f"{expected_scope!r}; found {timing_scope!r}."
                    )
                key = seed, iteration, method
                if key in rows:
                    raise ValueError(
                        f"Duplicate seed/iteration/method key {key!r} "
                        f"at {location}."
                    )
                rows[key] = {
                    "time_seconds": elapsed,
                    "candidate_count": candidate_count,
                    "timing_scope": timing_scope,
                }

            if file_row_count == 0:
                raise ValueError(f"Runtime CSV contains no data rows: {path}")

    if not rows:
        raise ValueError("No runtime rows were supplied.")
    return rows


def _candidate_sort_key(
    item: tuple[tuple, dict[str, float]], score_column: str
) -> tuple:
    key, values = item
    return -values[score_column], str(key[2]), str(key[3])


def _aggregate(
    rows: dict[tuple, dict[str, float]],
    runtime_rows: dict[tuple, dict],
    top_k: int,
) -> dict:
    by_seed_iteration = {}
    for key, values in rows.items():
        seed, iteration, candidate_type, candidate_id = key
        by_seed_iteration.setdefault((seed, iteration), []).append(
            ((seed, iteration, candidate_type, candidate_id), values)
        )

    candidate_iterations = set(by_seed_iteration)
    expected_runtime_keys = {
        (seed, iteration, method)
        for seed, iteration in candidate_iterations
        for method, _, _ in METHODS
    }
    if set(runtime_rows) != expected_runtime_keys:
        missing = sorted(expected_runtime_keys - set(runtime_rows), key=str)
        extra = sorted(set(runtime_rows) - expected_runtime_keys, key=str)
        raise ValueError(
            "Runtime rows must contain exactly one row for every "
            "candidate seed/iteration/method; "
            f"missing={missing}, extra={extra}."
        )

    seeds = sorted({seed for seed, _ in by_seed_iteration})
    per_seed = {}
    for seed in seeds:
        iteration_keys = sorted(
            key for key in by_seed_iteration if key[0] == seed
        )
        if not iteration_keys:
            raise ValueError(f"Seed {seed} has no iterations.")

        hit_values = {method: [] for method, _, _ in METHODS}
        recall_values = {method: [] for method, _, _ in METHODS}
        cumulative_times = {method: 0.0 for method, _, _ in METHODS}

        for seed_iteration in iteration_keys:
            candidates = by_seed_iteration[seed_iteration]
            if len(candidates) < top_k:
                raise ValueError(
                    f"Seed/iteration {seed_iteration!r} has {len(candidates)} "
                    f"candidates, fewer than --top-k={top_k}."
                )

            exact_order = sorted(
                candidates,
                key=lambda item: _candidate_sort_key(
                    item, "exact_utility_delta"
                ),
            )
            exact_best = exact_order[0][0]
            exact_top_k = {key for key, _ in exact_order[:top_k]}

            scoring_counts = [
                runtime_rows[(seed, seed_iteration[1], method)][
                    "candidate_count"
                ]
                for method in ("trim", "if", "random")
            ]
            if len(set(scoring_counts)) != 1:
                raise ValueError(
                    f"Seed/iteration {seed_iteration!r} has inconsistent "
                    f"scored-candidate counts: {scoring_counts!r}."
                )
            if scoring_counts[0] < len(candidates):
                raise ValueError(
                    f"Seed/iteration {seed_iteration!r} reports only "
                    f"{scoring_counts[0]} scored candidates but contains "
                    f"{len(candidates)} measured candidate rows."
                )
            retrain_count = runtime_rows[
                (seed, seed_iteration[1], "retrain")
            ]["candidate_count"]
            if retrain_count < len(candidates):
                raise ValueError(
                    f"Seed/iteration {seed_iteration!r} reports only "
                    f"{retrain_count} exact-evaluation attempts but contains "
                    f"{len(candidates)} successful candidate measurements."
                )

            for method, _, score_column in METHODS:
                predicted_order = sorted(
                    candidates,
                    key=lambda item, column=score_column: _candidate_sort_key(
                        item, column
                    ),
                )
                predicted_top_k = {
                    key for key, _ in predicted_order[:top_k]
                }
                hit_values[method].append(
                    float(exact_best in predicted_top_k)
                )
                recall_values[method].append(
                    len(predicted_top_k & exact_top_k) / float(top_k)
                )
                cumulative_times[method] += runtime_rows[
                    (seed, seed_iteration[1], method)
                ]["time_seconds"]

        trim_time = cumulative_times["trim"]
        if trim_time <= 0.0:
            raise ValueError(
                f"Seed {seed} has non-positive cumulative TRIM time: {trim_time!r}"
            )
        per_seed[seed] = {
            method: {
                "hit_rate": statistics.fmean(hit_values[method]),
                "recall": statistics.fmean(recall_values[method]),
                "relative_time": cumulative_times[method] / trim_time,
            }
            for method, _, _ in METHODS
        }

    return {
        method: {
            metric: statistics.fmean(
                per_seed[seed][method][metric] for seed in seeds
            )
            for metric in ("relative_time", "hit_rate", "recall")
        }
        for method, _, _ in METHODS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        action="append",
        help="One or more candidate-level CSV files; the option may be repeated.",
    )
    parser.add_argument(
        "--runtime-input",
        required=True,
        nargs="+",
        action="append",
        help=(
            "One or more iteration-level runtime CSV files; the option may "
            "be repeated."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be a positive integer.")

    rows = _read_rows(args.input)
    runtime_rows = _read_runtime_rows(args.runtime_input)
    metrics = _aggregate(rows, runtime_rows, args.top_k)

    import matplotlib.pyplot as plt

    method_keys = [method for method, _, _ in METHODS]
    method_labels = [label for _, label, _ in METHODS]
    fig, axes = plt.subplots(3, 1)
    panels = (
        ("relative_time", "Execution time / TRIM"),
        ("hit_rate", f"HitRate@{args.top_k}"),
        ("recall", f"Recall@{args.top_k}"),
    )
    for ax, (metric, y_label) in zip(axes, panels):
        ax.bar(method_labels, [metrics[method][metric] for method in method_keys])
        ax.set_ylabel(y_label)
    axes[-1].set_xlabel("Method")

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
