"""Plot one Exp-5 sensitivity panel from long-table results.

Choose ``topk``, ``seed-ratio``, or ``background`` with ``--figure``.  Primary
CSV/JSON rows must contain the configured x and y fields.  Their seed field is
also required unless ``--input-seeds`` assigns one seed to each primary input
file.  ``--primary-method`` supplies the method name when that field is absent.
Baseline CSV/JSON rows use the same schema and must contain the method field.
JSON is a row list or an object with a ``rows`` list.

Example template::

    python -m plotting.plot_exp5_sensitivity \
      --figure topk --input RESULTS.csv --baseline-input BASELINES.json \
      --output OUTPUT.pdf --primary-method PRIMARY --method-field method \
      --x-field rank_top_k --y-field final_leak_k \
      --y-transform negative-log-ratio --y-reference-field original_leak_k \
      --seed-field random_state --expected-methods PRIMARY BASELINE \
      --expected-seeds SEED_A SEED_B --x-values X_A X_B \
      --aggregation mean --error std --x-scale linear --y-scale linear \
      --x-label "Top-k" --y-label "Privacy metric"

``identity`` draws the configured y field directly.  ``negative-log-ratio`` first computes
``-log(y/reference)`` independently for every raw row and only then aggregates
the transformed values across seeds.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
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
except ImportError:  # Direct ``python plotting/plot_exp5_sensitivity.py`` use.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure", required=True, choices=("topk", "seed-ratio", "background")
    )
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--input-seeds", nargs="*")
    parser.add_argument("--baseline-input", nargs="*", default=[])
    _add_output_arguments(parser)
    parser.add_argument("--primary-method", required=True)
    parser.add_argument("--method-field", required=True)
    parser.add_argument("--x-field", required=True)
    parser.add_argument("--y-field", required=True)
    parser.add_argument(
        "--y-transform",
        required=True,
        choices=("identity", "negative-log-ratio"),
    )
    parser.add_argument("--y-reference-field")
    parser.add_argument("--seed-field", required=True)
    parser.add_argument("--expected-methods", required=True, nargs="+")
    parser.add_argument("--expected-seeds", required=True, nargs="+")
    parser.add_argument("--x-values", required=True, nargs="+")
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

    if args.y_transform == "negative-log-ratio":
        if args.y_reference_field is None:
            raise ValueError(
                "--y-reference-field is required with "
                "--y-transform negative-log-ratio."
            )
        y_reference_field = _text(
            args.y_reference_field,
            field="y reference field",
            location="CLI",
        )
    else:
        if args.y_reference_field is not None:
            raise ValueError(
                "--y-reference-field is valid only with "
                "--y-transform negative-log-ratio."
            )
        y_reference_field = None

    methods = _unique(
        [
            _text(value, field="expected method", location="CLI")
            for value in args.expected_methods
        ],
        name="--expected-methods",
    )
    primary_method = _text(
        args.primary_method, field="primary method", location="CLI"
    )
    if primary_method not in methods:
        raise ValueError("--primary-method must appear in --expected-methods.")
    seeds = _unique(
        [
            _seed_token(value, field="expected seed", location="CLI")
            for value in args.expected_seeds
        ],
        name="--expected-seeds",
    )
    x_values = _unique(
        [_number(value, field="x value", location="CLI") for value in args.x_values],
        name="--x-values",
    )
    input_seed_by_path = {}
    if args.input_seeds is not None:
        if len(args.input_seeds) != len(args.input):
            raise ValueError(
                "--input-seeds must contain exactly one seed for every --input file."
            )
        for raw_path, raw_seed in zip(args.input, args.input_seeds):
            path = Path(raw_path).expanduser()
            if path in input_seed_by_path:
                raise ValueError(f"Duplicate primary input path: {path}")
            input_seed_by_path[path] = _seed_token(
                raw_seed, field="input seed", location="CLI"
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
                method = _text(
                    row.get(args.method_field),
                    field=args.method_field,
                    location=location,
                )
                if is_primary and method != primary_method:
                    raise ValueError(
                        f"Primary method at {location} is {method!r}, "
                        f"expected {primary_method!r}."
                    )
            value_fields = [args.x_field, args.y_field]
            if y_reference_field is not None:
                value_fields.append(y_reference_field)
            missing = [field for field in value_fields if field not in row]
            if missing:
                raise ValueError(f"Missing fields at {location}: {missing}")
            if args.seed_field in row and str(row[args.seed_field]).strip():
                seed = _seed_token(
                    row[args.seed_field],
                    field=args.seed_field,
                    location=location,
                )
                assigned_seed = input_seed_by_path.get(path) if is_primary else None
                if assigned_seed is not None and seed != assigned_seed:
                    raise ValueError(
                        f"Seed at {location} is {seed!r}, but --input-seeds "
                        f"assigns {assigned_seed!r}."
                    )
            elif is_primary and path in input_seed_by_path:
                seed = input_seed_by_path[path]
            else:
                raise ValueError(f"Missing {args.seed_field!r} at {location}.")
            y_value = _number(
                row[args.y_field], field=args.y_field, location=location
            )
            reference_value = None
            if y_reference_field is not None:
                reference_value = _number(
                    row[y_reference_field],
                    field=y_reference_field,
                    location=location,
                )
                if y_value <= 0.0 or reference_value <= 0.0:
                    raise ValueError(
                        "negative-log-ratio requires positive y and reference "
                        f"values at {location}; found y={y_value!r}, "
                        f"reference={reference_value!r}."
                    )
            normalized.append(
                {
                    "method": method,
                    "x": _number(
                        row[args.x_field], field=args.x_field, location=location
                    ),
                    "seed": seed,
                    "y": y_value,
                    "y_reference": reference_value,
                }
            )

    if not normalized:
        raise ValueError("No result rows were supplied.")
    found_methods = {row["method"] for row in normalized}
    if found_methods != set(methods):
        raise ValueError(f"Expected methods {methods}, found {sorted(found_methods)}.")
    if {row["x"] for row in normalized} != set(x_values):
        raise ValueError("Input x values do not exactly match --x-values.")
    _validate_scale(x_values, args.x_scale, name="x data")

    points = {}
    references = {}
    for row in normalized:
        key = row["method"], row["x"], row["seed"]
        if key in points:
            raise ValueError(f"Duplicate method/x/seed row: {key}")
        points[key] = row["y"]
        if y_reference_field is not None:
            references[key] = row["y_reference"]
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

    if y_reference_field is not None:
        for seed in seeds:
            reference_by_point = {
                (method, x_value): references[(method, x_value, seed)]
                for method in methods
                for x_value in x_values
            }
            if len(set(reference_by_point.values())) != 1:
                details = ", ".join(
                    f"{method}/{x_value}={value!r}"
                    for (method, x_value), value in reference_by_point.items()
                )
                raise ValueError(
                    f"Inconsistent {y_reference_field!r} for seed {seed!r}; "
                    "negative-log-ratio requires one shared K0 across every "
                    f"method and x value in the panel. Found: {details}."
                )
        points = {
            key: math.log(references[key]) - math.log(value)
            for key, value in points.items()
        }
    _validate_scale(list(points.values()), args.y_scale, name="y data")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for method in methods:
        centers = []
        lower = []
        upper = []
        for x_value in x_values:
            values = [points[(method, x_value, seed)] for seed in seeds]
            center = (
                statistics.fmean(values)
                if args.aggregation == "mean"
                else statistics.median(values)
            )
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
            ax.plot(x_values, centers, label=method)
        else:
            ax.errorbar(
                x_values,
                centers,
                yerr=[lower, upper],
                fmt="-",
                label=method,
            )
    ax.set_xscale(args.x_scale)
    ax.set_yscale(args.y_scale)
    ax.set_xlabel(args.x_label)
    ax.set_ylabel(args.y_label)
    ax.legend()
    _save_figure(fig, ax, args)
    plt.close(fig)


if __name__ == "__main__":
    main()
