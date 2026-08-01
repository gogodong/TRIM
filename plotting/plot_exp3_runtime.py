"""Plot one Exp-3 runtime panel from explicit result tables.

The primary CSV/JSON table is the experiment ``run_summaries`` table.  It must
contain the fields named by ``--x-field``, ``--y-field``, and ``--seed-field``.
If it has no method column, ``--primary-method`` supplies that value.  Every
baseline table uses the same long-table schema and must additionally contain
``--method-field``.  JSON input is either a list of row objects or an object
whose ``rows`` member is that list.

Example template (replace every capitalized value)::

    python -m plotting.plot_exp3_runtime \
      --figure real --input RESULTS.csv --baseline-input BASELINES.csv \
      --output OUTPUT.pdf --primary-method PRIMARY \
      --method-field method --x-field dataset \
      --y-field paper_algorithm_time_seconds --seed-field random_state \
      --expected-methods PRIMARY BASELINE \
      --expected-seeds SEED_A SEED_B --x-values DATASET_A DATASET_B \
      --x-type categorical --aggregation mean --error std \
      --x-scale linear --y-scale linear \
      --x-label Dataset --y-label "Runtime (seconds)"

The command aggregates the declared methods, seeds, and axis values.  With
``--normalize-to-method``, each raw runtime is first divided by the reference
method at the same x value and seed; the per-seed ratios are then aggregated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")


def _read_rows(paths):
    rows = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Input table does not exist: {path}")
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV table has no header: {path}")
                if any(not str(field).strip() for field in reader.fieldnames):
                    raise ValueError(f"CSV table has a blank header: {path}")
                if len(reader.fieldnames) != len(set(reader.fieldnames)):
                    raise ValueError(f"CSV table has duplicate headers: {path}")
                loaded = list(reader)
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            loaded = payload.get("rows") if isinstance(payload, dict) else payload
            if not isinstance(loaded, list):
                raise ValueError(
                    f"JSON table must be a row list or contain a row list at 'rows': {path}"
                )
        else:
            raise ValueError(f"Input must be .csv or .json: {path}")
        if not loaded:
            raise ValueError(f"Input table is empty: {path}")
        for row_number, row in enumerate(loaded, start=1):
            if not isinstance(row, dict):
                raise ValueError(f"Row {row_number} in {path} is not an object.")
            if None in row:
                raise ValueError(
                    f"Row {row_number} in {path} has more values than headers."
                )
            rows.append((path, row_number, dict(row)))
    return rows


def _text(value, *, field, location):
    if value is None or not str(value).strip():
        raise ValueError(f"Missing {field!r} at {location}.")
    return str(value).strip()


def _number(value, *, field, location):
    if isinstance(value, bool):
        raise ValueError(f"Boolean {field!r} at {location} is not numeric data.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric {field!r} at {location}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field!r} at {location}: {value!r}")
    return number


def _seed_token(value, *, field, location):
    text = _text(value, field=field, location=location)
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return text


def _unique(values, *, name):
    if not values:
        raise ValueError(f"{name} cannot be empty.")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicates: {values}")
    return values


def _validate_scale(values, scale, *, name):
    if scale == "log" and any(value <= 0.0 for value in values):
        raise ValueError(f"{name} contains a non-positive value for log scale.")
    if scale == "logit" and any(not 0.0 < value < 1.0 for value in values):
        raise ValueError(f"{name} contains a value outside (0, 1) for logit scale.")


def _error(center, values, mode):
    if mode == "none":
        return None
    if mode == "minmax":
        return center - min(values), max(values) - center
    if len(values) < 2:
        raise ValueError(f"{mode} requires at least two seeds per point.")
    spread = statistics.stdev(values)
    if mode == "sem":
        spread /= math.sqrt(len(values))
    return spread, spread


def _add_output_arguments(parser):
    parser.add_argument("--output", required=True)
    parser.add_argument("--title")
    parser.add_argument("--figsize", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--overwrite", action="store_true")


def _save_figure(fig, ax, args):
    if args.title is not None:
        ax.set_title(args.title)
    if args.figsize is not None:
        width, height = args.figsize
        if (
            not math.isfinite(width)
            or not math.isfinite(height)
            or width <= 0.0
            or height <= 0.0
        ):
            raise ValueError("--figsize values must be finite and positive.")
        fig.set_size_inches(width, height)
    if args.dpi is not None and args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    output = Path(args.output).expanduser()
    if not output.suffix:
        raise ValueError("--output must include a Matplotlib-supported suffix.")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_options = {} if args.dpi is None else {"dpi": args.dpi}
    fig.savefig(output, **save_options)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", required=True, choices=("real", "bng"))
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--baseline-input", nargs="*", default=[])
    _add_output_arguments(parser)
    parser.add_argument("--primary-method", required=True)
    parser.add_argument("--method-field", required=True)
    parser.add_argument("--x-field", required=True)
    parser.add_argument("--y-field", required=True)
    parser.add_argument("--seed-field", required=True)
    parser.add_argument("--expected-methods", required=True, nargs="+")
    parser.add_argument("--expected-seeds", required=True, nargs="+")
    parser.add_argument("--x-values", required=True, nargs="+")
    parser.add_argument("--normalize-to-method")
    parser.add_argument("--x-type", required=True, choices=("categorical", "numeric"))
    parser.add_argument("--aggregation", required=True, choices=("mean", "median"))
    parser.add_argument("--error", required=True, choices=("none", "std", "sem", "minmax"))
    parser.add_argument("--x-scale", required=True, choices=("linear", "log", "symlog", "logit"))
    parser.add_argument("--y-scale", required=True, choices=("linear", "log", "symlog", "logit"))
    parser.add_argument("--x-label", required=True)
    parser.add_argument("--y-label", required=True)
    args = parser.parse_args()

    if args.figure == "real" and args.x_type != "categorical":
        raise ValueError("The Exp-3 real-data panel requires categorical x values.")
    if args.figure == "bng" and args.x_type != "numeric":
        raise ValueError("The Exp-3 BNG-size panel requires numeric x values.")
    if args.x_type == "categorical" and args.x_scale != "linear":
        raise ValueError("Categorical x values support only --x-scale linear.")

    methods = _unique(
        [_text(value, field="expected method", location="CLI") for value in args.expected_methods],
        name="--expected-methods",
    )
    primary_method = _text(args.primary_method, field="primary method", location="CLI")
    if primary_method not in methods:
        raise ValueError("--primary-method must appear in --expected-methods.")
    normalize_to_method = None
    if args.normalize_to_method is not None:
        normalize_to_method = _text(
            args.normalize_to_method,
            field="normalization reference method",
            location="CLI",
        )
        if normalize_to_method not in methods:
            raise ValueError(
                "--normalize-to-method must appear in --expected-methods."
            )
    seeds = _unique(
        [_seed_token(value, field="expected seed", location="CLI") for value in args.expected_seeds],
        name="--expected-seeds",
    )
    if args.x_type == "numeric":
        x_values = _unique(
            [_number(value, field="x value", location="CLI") for value in args.x_values],
            name="--x-values",
        )
    else:
        x_values = _unique(
            [_text(value, field="x value", location="CLI") for value in args.x_values],
            name="--x-values",
        )

    normalized = []
    for is_primary, records in (
        (True, _read_rows(args.input)),
        (False, _read_rows(args.baseline_input) if args.baseline_input else []),
    ):
        for path, row_number, row in records:
            location = f"{path}:{row_number}"
            if is_primary and args.method_field not in row:
                method = primary_method
            else:
                method = _text(row.get(args.method_field), field=args.method_field, location=location)
                if is_primary and method != primary_method:
                    raise ValueError(
                        f"Primary input method at {location} is {method!r}, expected {primary_method!r}."
                    )
            if args.x_field not in row or args.y_field not in row or args.seed_field not in row:
                missing = [
                    field
                    for field in (args.x_field, args.y_field, args.seed_field)
                    if field not in row
                ]
                raise ValueError(f"Missing fields at {location}: {missing}")
            x_value = (
                _number(row[args.x_field], field=args.x_field, location=location)
                if args.x_type == "numeric"
                else _text(row[args.x_field], field=args.x_field, location=location)
            )
            normalized.append(
                {
                    "method": method,
                    "x": x_value,
                    "seed": _seed_token(row[args.seed_field], field=args.seed_field, location=location),
                    "y": _number(row[args.y_field], field=args.y_field, location=location),
                }
            )

    if not normalized:
        raise ValueError("No result rows were supplied.")
    found_methods = {row["method"] for row in normalized}
    if found_methods != set(methods):
        raise ValueError(f"Expected methods {methods}, found {sorted(found_methods)}.")
    if {row["x"] for row in normalized} != set(x_values):
        raise ValueError("Input x values do not exactly match --x-values.")
    if args.x_type == "numeric":
        _validate_scale(x_values, args.x_scale, name="x data")
    points = {}
    for row in normalized:
        key = row["method"], row["x"], row["seed"]
        if key in points:
            raise ValueError(f"Duplicate method/x/seed row: {key}")
        points[key] = row["y"]
    expected_keys = {
        (method, x_value, seed)
        for method in methods
        for x_value in x_values
        for seed in seeds
    }
    if set(points) != expected_keys:
        missing = sorted(expected_keys.difference(points), key=str)
        extra = sorted(set(points).difference(expected_keys), key=str)
        raise ValueError(f"Incomplete method/x/seed matrix; missing={missing}, extra={extra}.")

    if normalize_to_method is not None:
        for key, value in points.items():
            if value <= 0.0:
                raise ValueError(
                    "Runtime normalization requires every raw runtime to be "
                    f"finite and positive; found {value!r} at {key}."
                )
        normalized_points = {}
        for method in methods:
            for x_value in x_values:
                for seed in seeds:
                    key = method, x_value, seed
                    normalized_points[key] = _number(
                        points[key]
                        / points[(normalize_to_method, x_value, seed)],
                        field="normalized runtime ratio",
                        location=str(key),
                    )
        points = normalized_points
    _validate_scale(list(points.values()), args.y_scale, name="y data")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_x = list(range(len(x_values))) if args.x_type == "categorical" else x_values
    for method in methods:
        centers = []
        lower = []
        upper = []
        for x_value in x_values:
            values = [points[(method, x_value, seed)] for seed in seeds]
            center = statistics.fmean(values) if args.aggregation == "mean" else statistics.median(values)
            centers.append(center)
            interval = _error(center, values, args.error)
            if interval is not None:
                lower.append(interval[0])
                upper.append(interval[1])
                _validate_scale(
                    [center - interval[0], center + interval[1]],
                    args.y_scale,
                    name=f"{method} error interval",
                )
        if args.error == "none":
            ax.plot(plot_x, centers, label=method)
        else:
            ax.errorbar(plot_x, centers, yerr=[lower, upper], fmt="-", label=method)
    if args.x_type == "categorical":
        ax.set_xticks(plot_x, [str(value) for value in x_values])
    ax.set_xscale(args.x_scale)
    ax.set_yscale(args.y_scale)
    ax.set_xlabel(args.x_label)
    ax.set_ylabel(args.y_label)
    ax.legend()
    _save_figure(fig, ax, args)
    plt.close(fig)


if __name__ == "__main__":
    main()
