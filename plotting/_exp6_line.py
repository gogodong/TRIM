"""Chronological renderer shared by the two Exp-6 line figures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

try:
    from .plot_exp3_runtime import _number, _read_rows, _seed_token, _text, _unique
except ImportError:  # Direct script execution through an Exp-6 wrapper.
    from plot_exp3_runtime import (  # type: ignore
        _number,
        _read_rows,
        _seed_token,
        _text,
        _unique,
    )


@dataclass(frozen=True)
class Exp6LineSpec:
    experiment_kind: str
    description: str


def _negative_log_ratio(value, reference, *, field, location):
    if value <= 0.0 or reference <= 0.0:
        raise ValueError(
            f"{field} requires positive K and K0 at {location}; "
            f"found K={value}, K0={reference}."
        )
    result = math.log(reference) - math.log(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite -log(K/K0) at {location}.")
    return result


def main(spec: Exp6LineSpec) -> None:
    parser = argparse.ArgumentParser(description=spec.description)
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tolerance", required=True, type=float)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--expected-methods", required=True, nargs="+")
    parser.add_argument(
        "--method-label",
        action="append",
        default=[],
        metavar="METHOD=LABEL",
        help="Optional explicit display label; repeat for each relabelled method.",
    )
    parser.add_argument("--experiment-kind-field", default="experiment_kind")
    parser.add_argument("--dataset-field", default="dataset")
    parser.add_argument("--model-field", default="model_name")
    parser.add_argument("--tolerance-field", default="tolerance")
    parser.add_argument("--seed-field", default="random_state")
    parser.add_argument("--method-field", default="method")
    parser.add_argument("--iteration-field", default="iteration")
    parser.add_argument("--x-field", required=True)
    parser.add_argument("--top-k-field", required=True)
    parser.add_argument("--top-reference-field", required=True)
    parser.add_argument("--bottom-field", required=True)
    parser.add_argument("--x-label", required=True)
    parser.add_argument("--top-y-label", required=True)
    parser.add_argument("--bottom-y-label", required=True)
    parser.add_argument(
        "--x-scale", required=True, choices=("linear", "log", "symlog", "logit")
    )
    parser.add_argument(
        "--y-scale", required=True, choices=("linear", "log", "symlog", "logit")
    )
    parser.add_argument("--legend", action="store_true")
    parser.add_argument("--title")
    parser.add_argument("--figsize", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not math.isfinite(args.tolerance) or args.tolerance < 0.0:
        raise ValueError("--tolerance must be finite and non-negative.")
    methods = _unique(
        [
            _text(value, field="expected method", location="CLI")
            for value in args.expected_methods
        ],
        name="--expected-methods",
    )
    method_labels = {}
    for declaration in args.method_label:
        if "=" not in declaration:
            raise ValueError("--method-label must use METHOD=LABEL syntax.")
        method, label = (part.strip() for part in declaration.split("=", 1))
        if method not in methods or not label:
            raise ValueError(
                f"Invalid --method-label {declaration!r}; method must be "
                "declared and label must be non-empty."
            )
        if method in method_labels:
            raise ValueError(f"Duplicate display label for method {method!r}.")
        method_labels[method] = label
    seed = _seed_token(args.seed, field="seed", location="CLI")
    required_fields = {
        args.experiment_kind_field,
        args.dataset_field,
        args.model_field,
        args.tolerance_field,
        args.seed_field,
        args.method_field,
        args.iteration_field,
        args.x_field,
        args.top_k_field,
        args.top_reference_field,
        args.bottom_field,
    }

    selected = []
    contribution = {Path(path).expanduser(): 0 for path in args.input}
    for path, row_number, row in _read_rows(args.input):
        location = f"{path}:{row_number}"
        missing = sorted(required_fields - set(row))
        if missing:
            raise ValueError(f"Missing fields at {location}: {missing}.")
        row_kind = _text(
            row[args.experiment_kind_field],
            field=args.experiment_kind_field,
            location=location,
        )
        row_dataset = _text(
            row[args.dataset_field], field=args.dataset_field, location=location
        )
        row_model = _text(
            row[args.model_field], field=args.model_field, location=location
        )
        row_seed = _seed_token(
            row[args.seed_field], field=args.seed_field, location=location
        )
        row_tolerance = _number(
            row[args.tolerance_field],
            field=args.tolerance_field,
            location=location,
        )
        if (
            row_kind != spec.experiment_kind
            or row_dataset != args.dataset
            or row_model != args.model
            or row_seed != seed
            or row_tolerance != args.tolerance
        ):
            continue
        method = _text(
            row[args.method_field], field=args.method_field, location=location
        )
        if method not in methods:
            raise ValueError(
                f"Unexpected method {method!r} at {location}; expected {methods}."
            )
        iteration_number = _number(
            row[args.iteration_field],
            field=args.iteration_field,
            location=location,
        )
        iteration = int(iteration_number)
        if iteration < 0 or float(iteration) != iteration_number:
            raise ValueError(f"Iteration must be a non-negative integer at {location}.")
        x_value = _number(row[args.x_field], field=args.x_field, location=location)
        top_k = _number(
            row[args.top_k_field], field=args.top_k_field, location=location
        )
        top_reference = _number(
            row[args.top_reference_field],
            field=args.top_reference_field,
            location=location,
        )
        bottom_value = _number(
            row[args.bottom_field], field=args.bottom_field, location=location
        )
        selected.append({
            "method": method,
            "iteration": iteration,
            "x": x_value,
            "top": _negative_log_ratio(
                top_k, top_reference, field=args.top_k_field, location=location
            ),
            "bottom": bottom_value,
            "top_reference": top_reference,
            "location": location,
        })
        contribution[path] += 1
    empty_inputs = [str(path) for path, count in contribution.items() if count == 0]
    if empty_inputs:
        raise ValueError(
            "Every explicit input must contribute selected rows; none found in "
            f"{empty_inputs}."
        )

    rows_by_method = {}
    for method in methods:
        method_rows = sorted(
            (row for row in selected if row["method"] == method),
            key=lambda row: row["iteration"],
        )
        if not method_rows:
            raise ValueError(f"Missing all rows for declared method {method!r}.")
        iterations = [row["iteration"] for row in method_rows]
        if iterations != list(range(len(iterations))):
            raise ValueError(
                f"Method {method!r} must contain chronological iterations "
                f"0..N-1 exactly once; found {iterations}."
            )
        rows_by_method[method] = method_rows
    keys = [(row["method"], row["iteration"]) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate method/iteration rows in selected inputs.")
    top_references = {row["top_reference"] for row in selected}
    if len(top_references) != 1:
        raise ValueError(
            "K0 reference must agree across every method and iteration for "
            "the selected seed."
        )

    all_x = [row["x"] for row in selected]
    all_y = [row["top"] for row in selected] + [row["bottom"] for row in selected]
    if args.x_scale == "log" and any(value <= 0.0 for value in all_x):
        raise ValueError("The selected x values are not valid for log scale.")
    if args.y_scale == "log" and any(value <= 0.0 for value in all_y):
        raise ValueError("The selected privacy values are not valid for log scale.")
    if args.x_scale == "logit" and any(not 0.0 < value < 1.0 for value in all_x):
        raise ValueError("The selected x values are not valid for logit scale.")
    if args.y_scale == "logit" and any(not 0.0 < value < 1.0 for value in all_y):
        raise ValueError("The selected privacy values are not valid for logit scale.")

    fig, (top_axis, bottom_axis) = plt.subplots(2, 1, sharex=True)
    for method in methods:
        method_rows = rows_by_method[method]
        x_values = [row["x"] for row in method_rows]
        top_axis.plot(
            x_values,
            [row["top"] for row in method_rows],
            marker="o",
            label=method_labels.get(method, method),
        )
        bottom_axis.plot(
            x_values,
            [row["bottom"] for row in method_rows],
            marker="o",
            label=method_labels.get(method, method),
        )
    top_axis.set_ylabel(args.top_y_label)
    bottom_axis.set_ylabel(args.bottom_y_label)
    bottom_axis.set_xlabel(args.x_label)
    for axis in (top_axis, bottom_axis):
        axis.set_xscale(args.x_scale)
        axis.set_yscale(args.y_scale)
    if args.legend:
        bottom_axis.legend()
    if args.title is not None:
        fig.suptitle(args.title)
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
    plt.close(fig)


__all__ = ["Exp6LineSpec", "main"]
