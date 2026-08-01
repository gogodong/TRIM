"""Plot the appendix removal view from primary and baseline tables.

Primary ``run_summaries`` rows are converted from wide to long form using one
or more explicit ``--series LABEL=FIELD`` arguments.  Each row must also contain
the configured x and seed fields.  Optional baseline CSV/JSON input is already
long form and uses the separately configured method, x, seed, and value fields.
JSON is a row list or an object with a ``rows`` row list.

Example template::

    python -m plotting.plot_appendix_removal_view \
      --input RESULTS.csv --baseline-input BASELINES.json --output OUTPUT.pdf \
      --x-field dataset --seed-field random_state \
      --series BEFORE=original_leak_k AFTER=final_leak_k \
      --baseline-method-field method --baseline-x-field dataset \
      --baseline-seed-field seed --baseline-value-field value \
      --expected-methods BEFORE AFTER BASELINE \
      --expected-seeds SEED_A SEED_B --x-values DATASET_A DATASET_B \
      --aggregation mean --error std --x-scale linear --y-scale linear \
      --x-label Dataset --y-label "Privacy metric"

With ``--normalize-to-method``, each raw value is first divided by the
reference method at the same dataset and seed, and the resulting per-seed
ratios are then aggregated.
"""

from __future__ import annotations

import argparse
import statistics

try:
    from .plot_exp3_runtime import (
        _add_output_arguments,
        _error,
        _number,
        _read_rows,
        _save_figure,
        _seed_token,
        _text,
        _unique,
        _validate_scale,
    )
except ImportError:  # Direct ``python plotting/plot_appendix_removal_view.py``.
    from plot_exp3_runtime import (  # type: ignore
        _add_output_arguments,
        _error,
        _number,
        _read_rows,
        _save_figure,
        _seed_token,
        _text,
        _unique,
        _validate_scale,
    )


def _parse_series(values):
    series = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --series {value!r}; expected a LABEL=FIELD pair."
            )
        label, field = value.split("=", 1)
        label = _text(label, field="series label", location="CLI")
        field = _text(field, field="series field", location="CLI")
        series.append((label, field))
    labels = [label for label, _field in series]
    fields = [field for _label, field in series]
    _unique(labels, name="--series labels")
    _unique(fields, name="--series fields")
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--baseline-input", nargs="*", default=[])
    _add_output_arguments(parser)
    parser.add_argument("--x-field", required=True)
    parser.add_argument("--seed-field", required=True)
    parser.add_argument("--series", required=True, nargs="+")
    parser.add_argument("--baseline-method-field")
    parser.add_argument("--baseline-x-field")
    parser.add_argument("--baseline-seed-field")
    parser.add_argument("--baseline-value-field")
    parser.add_argument("--expected-methods", required=True, nargs="+")
    parser.add_argument("--expected-seeds", required=True, nargs="+")
    parser.add_argument("--x-values", required=True, nargs="+")
    parser.add_argument("--normalize-to-method")
    parser.add_argument("--aggregation", required=True, choices=("mean", "median"))
    parser.add_argument(
        "--error", required=True, choices=("none", "std", "sem", "minmax")
    )
    parser.add_argument(
        "--x-scale", required=True, choices=("linear", "log", "symlog", "logit")
    )
    parser.add_argument(
        "--y-scale", required=True, choices=("linear", "log", "symlog", "logit")
    )
    parser.add_argument("--x-label", required=True)
    parser.add_argument("--y-label", required=True)
    args = parser.parse_args()

    baseline_fields = (
        args.baseline_method_field,
        args.baseline_x_field,
        args.baseline_seed_field,
        args.baseline_value_field,
    )
    if args.baseline_input and any(field is None for field in baseline_fields):
        raise ValueError(
            "All four --baseline-*-field options are required when "
            "--baseline-input is supplied."
        )
    if not args.baseline_input and any(field is not None for field in baseline_fields):
        raise ValueError(
            "--baseline-*-field options require at least one --baseline-input."
        )

    if args.x_scale != "linear":
        raise ValueError("Categorical x values require --x-scale linear.")
    series = _parse_series(args.series)
    methods = _unique(
        [
            _text(value, field="expected method", location="CLI")
            for value in args.expected_methods
        ],
        name="--expected-methods",
    )
    seeds = _unique(
        [
            _seed_token(value, field="expected seed", location="CLI")
            for value in args.expected_seeds
        ],
        name="--expected-seeds",
    )
    x_values = _unique(
        [_text(value, field="x value", location="CLI") for value in args.x_values],
        name="--x-values",
    )
    series_labels = {label for label, _field in series}
    if not series_labels.issubset(methods):
        raise ValueError("Every --series label must appear in --expected-methods.")
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

    points = {}
    for path, row_number, row in _read_rows(args.input):
        location = f"{path}:{row_number}"
        required = [args.x_field, args.seed_field, *(field for _label, field in series)]
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f"Missing fields at {location}: {missing}")
        x_value = _text(row[args.x_field], field=args.x_field, location=location)
        seed = _seed_token(
            row[args.seed_field], field=args.seed_field, location=location
        )
        for label, field in series:
            key = label, x_value, seed
            if key in points:
                raise ValueError(f"Duplicate method/x/seed row: {key}")
            points[key] = _number(row[field], field=field, location=location)

    if args.baseline_input:
        for path, row_number, row in _read_rows(args.baseline_input):
            location = f"{path}:{row_number}"
            required = (
                args.baseline_method_field,
                args.baseline_x_field,
                args.baseline_seed_field,
                args.baseline_value_field,
            )
            missing = [field for field in required if field not in row]
            if missing:
                raise ValueError(f"Missing baseline fields at {location}: {missing}")
            method = _text(
                row[args.baseline_method_field],
                field=args.baseline_method_field,
                location=location,
            )
            if method in series_labels:
                raise ValueError(
                    f"Baseline method {method!r} at {location} duplicates a primary series."
                )
            x_value = _text(
                row[args.baseline_x_field],
                field=args.baseline_x_field,
                location=location,
            )
            seed = _seed_token(
                row[args.baseline_seed_field],
                field=args.baseline_seed_field,
                location=location,
            )
            key = method, x_value, seed
            if key in points:
                raise ValueError(f"Duplicate method/x/seed row: {key}")
            points[key] = _number(
                row[args.baseline_value_field],
                field=args.baseline_value_field,
                location=location,
            )

    expected_keys = {
        (method, x_value, seed)
        for method in methods
        for x_value in x_values
        for seed in seeds
    }
    if set(points) != expected_keys:
        missing = sorted(expected_keys.difference(points), key=str)
        extra = sorted(set(points).difference(expected_keys), key=str)
        raise ValueError(
            f"Incomplete method/x/seed matrix; missing={missing}, extra={extra}."
        )
    if normalize_to_method is not None:
        for key, value in points.items():
            if value <= 0.0:
                raise ValueError(
                    "Metric normalization requires every raw value to be finite "
                    f"and positive; found {value!r} at {key}."
                )
        normalized_points = {}
        for method in methods:
            for x_value in x_values:
                for seed in seeds:
                    key = method, x_value, seed
                    normalized_points[key] = _number(
                        points[key]
                        / points[(normalize_to_method, x_value, seed)],
                        field="normalized metric ratio",
                        location=str(key),
                    )
        points = normalized_points
    _validate_scale(list(points.values()), args.y_scale, name="metric data")

    centers = {}
    intervals = {}
    for method in methods:
        for x_value in x_values:
            values = [points[(method, x_value, seed)] for seed in seeds]
            center = (
                statistics.fmean(values)
                if args.aggregation == "mean"
                else statistics.median(values)
            )
            centers[method, x_value] = center
            interval = _error(center, values, args.error)
            intervals[method, x_value] = interval
            if interval is not None:
                _validate_scale(
                    [center - interval[0], center + interval[1]],
                    args.y_scale,
                    name=f"{method}/{x_value} error interval",
                )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    group_positions = list(range(len(x_values)))
    width = 0.8 / len(methods)
    offset_origin = (len(methods) - 1) / 2
    for method_index, method in enumerate(methods):
        positions = [
            position + (method_index - offset_origin) * width
            for position in group_positions
        ]
        values = [centers[method, x_value] for x_value in x_values]
        if args.error == "none":
            ax.bar(positions, values, width=width, label=method)
        else:
            lower = [intervals[method, x_value][0] for x_value in x_values]
            upper = [intervals[method, x_value][1] for x_value in x_values]
            ax.bar(
                positions,
                values,
                width=width,
                yerr=[lower, upper],
                label=method,
            )
    ax.set_xticks(group_positions, x_values)
    ax.set_xscale(args.x_scale)
    ax.set_yscale(args.y_scale)
    ax.set_xlabel(args.x_label)
    ax.set_ylabel(args.y_label)
    ax.legend()
    _save_figure(fig, ax, args)
    plt.close(fig)


if __name__ == "__main__":
    main()
