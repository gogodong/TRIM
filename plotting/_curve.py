"""Shared curve renderer with optional per-seed ratio transforms."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


BASELINE_FIELDS = {
    "experiment",
    "method",
    "dataset",
    "model",
    "seed",
    "point",
    "x_metric",
    "x_value",
    "y_metric",
    "y_value",
}
TRANSFORMS = ("identity", "negative-log-ratio")


@dataclass(frozen=True)
class CurveSpec:
    """Experiment-specific labels for the shared curve command."""

    experiment: str
    description: str
    trim_input_help: str


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV input has no header: {path}")
            if any(not str(field).strip() for field in reader.fieldnames):
                raise ValueError(f"CSV input has a blank header: {path}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"CSV input has duplicate headers: {path}")
            rows = list(reader)
            for row_number, row in enumerate(rows, start=2):
                if None in row:
                    raise ValueError(
                        f"CSV row has more values than headers at "
                        f"{path} row {row_number}."
                    )
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and set(payload) == {"rows"}:
            payload = payload["rows"]
        if not isinstance(payload, list) or not all(
            isinstance(row, dict) for row in payload
        ):
            raise ValueError(
                f"{path} must contain a JSON row list or exactly "
                "{'rows': [...]}."
            )
        rows = payload
    elif suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"Blank JSONL row at {path}:{line_number}.")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"JSONL row at {path}:{line_number} is not an object."
                    )
                rows.append(row)
    else:
        raise ValueError(f"Unsupported input format for {path}; use CSV/JSON/JSONL.")

    if not rows:
        raise ValueError(f"Input file contains no rows: {path}")
    return rows


def _require_fields(row: dict, fields: set[str], path: Path, row_number: int) -> None:
    missing = sorted(field for field in fields if field not in row)
    if missing:
        raise ValueError(f"Missing fields {missing} at {path} row {row_number}.")


def _text(value, field: str, path: Path, row_number: int) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"Empty {field!r} at {path} row {row_number}.")
    return str(value).strip()


def _seed(value, path: Path, row_number: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid seed at {path} row {row_number}: {value!r}")
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid integer seed at {path} row {row_number}: {value!r}"
        ) from error


def _number(value, field: str, path: Path, row_number: int) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Invalid numeric {field!r} at {path} row {row_number}.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid numeric {field!r} at {path} row {row_number}: {value!r}"
        ) from error
    if not math.isfinite(parsed):
        raise ValueError(
            f"Non-finite numeric {field!r} at {path} row {row_number}: {value!r}"
        )
    return parsed


def _transform_value(
    value: float,
    reference: float | None,
    transform: str,
    *,
    axis: str,
    path: Path,
    row_number: int,
) -> float:
    """Transform one raw seed row before any across-seed aggregation."""
    if transform == "identity":
        return value
    if transform != "negative-log-ratio":
        raise ValueError(f"Unsupported {axis}-axis transform: {transform!r}")
    if reference is None:
        raise ValueError(
            f"{axis}-axis negative-log-ratio needs a reference at "
            f"{path} row {row_number}."
        )
    if value <= 0.0 or reference <= 0.0:
        raise ValueError(
            f"{axis}-axis negative-log-ratio requires finite positive value "
            f"and reference at {path} row {row_number}; got "
            f"value={value!r}, reference={reference!r}."
        )
    # log(reference) - log(value) is numerically safer than forming value/reference.
    transformed = math.log(reference) - math.log(value)
    if not math.isfinite(transformed):
        raise ValueError(
            f"Non-finite {axis}-axis negative-log-ratio at "
            f"{path} row {row_number}."
        )
    return transformed


def _load_trim_rows(args) -> list[dict]:
    required = {
        args.trim_dataset_column,
        args.trim_model_column,
        args.trim_seed_column,
        args.point_column,
        args.x_metric,
        args.y_metric,
    }
    if args.x_transform == "negative-log-ratio":
        required.add(args.trim_x_reference_column)
    if args.y_transform == "negative-log-ratio":
        required.add(args.trim_y_reference_column)
    selected = []
    for value in args.trim_results:
        path = Path(value)
        contributed = 0
        for row_number, row in enumerate(_read_rows(path), start=2):
            _require_fields(row, required, path, row_number)
            dataset = _text(
                row[args.trim_dataset_column],
                args.trim_dataset_column,
                path,
                row_number,
            )
            model = _text(
                row[args.trim_model_column],
                args.trim_model_column,
                path,
                row_number,
            )
            seed = _seed(row[args.trim_seed_column], path, row_number)
            point = _text(row[args.point_column], args.point_column, path, row_number)
            x_value = _number(row[args.x_metric], args.x_metric, path, row_number)
            y_value = _number(row[args.y_metric], args.y_metric, path, row_number)
            x_reference = (
                _number(
                    row[args.trim_x_reference_column],
                    args.trim_x_reference_column,
                    path,
                    row_number,
                )
                if args.x_transform == "negative-log-ratio"
                else None
            )
            y_reference = (
                _number(
                    row[args.trim_y_reference_column],
                    args.trim_y_reference_column,
                    path,
                    row_number,
                )
                if args.y_transform == "negative-log-ratio"
                else None
            )
            if dataset != args.dataset or model != args.model:
                continue
            selected.append(
                {
                    "method": args.trim_method,
                    "seed": seed,
                    "point": point,
                    "x": _transform_value(
                        x_value,
                        x_reference,
                        args.x_transform,
                        axis="x",
                        path=path,
                        row_number=row_number,
                    ),
                    "x_reference": x_reference,
                    "y": _transform_value(
                        y_value,
                        y_reference,
                        args.y_transform,
                        axis="y",
                        path=path,
                        row_number=row_number,
                    ),
                    "y_reference": y_reference,
                    "source": f"{path}:{row_number}",
                }
            )
            contributed += 1
        if contributed == 0:
            raise ValueError(
                f"TRIM input {path} has no rows for dataset={args.dataset!r}, "
                f"model={args.model!r}."
            )
    return selected


def _load_baseline_rows(args, experiment: str) -> list[dict]:
    selected = []
    for value in args.baseline_results:
        path = Path(value)
        contributed = 0
        for row_number, row in enumerate(_read_rows(path), start=2):
            required = set(BASELINE_FIELDS)
            if args.x_transform == "negative-log-ratio":
                required.add("x_reference")
            if args.y_transform == "negative-log-ratio":
                required.add("y_reference")
            _require_fields(row, required, path, row_number)
            row_experiment = _text(row["experiment"], "experiment", path, row_number)
            method = _text(row["method"], "method", path, row_number)
            dataset = _text(row["dataset"], "dataset", path, row_number)
            model = _text(row["model"], "model", path, row_number)
            seed = _seed(row["seed"], path, row_number)
            point = _text(row["point"], "point", path, row_number)
            x_metric = _text(row["x_metric"], "x_metric", path, row_number)
            y_metric = _text(row["y_metric"], "y_metric", path, row_number)
            x_value = _number(row["x_value"], "x_value", path, row_number)
            y_value = _number(row["y_value"], "y_value", path, row_number)
            x_reference = (
                _number(row["x_reference"], "x_reference", path, row_number)
                if args.x_transform == "negative-log-ratio"
                else None
            )
            y_reference = (
                _number(row["y_reference"], "y_reference", path, row_number)
                if args.y_transform == "negative-log-ratio"
                else None
            )
            if (
                row_experiment != experiment
                or dataset != args.dataset
                or model != args.model
                or x_metric != args.x_metric
                or y_metric != args.y_metric
            ):
                continue
            if method == args.trim_method:
                raise ValueError(
                    "Baseline input must not provide the declared TRIM method "
                    f"{method!r}: {path} row {row_number}."
                )
            selected.append(
                {
                    "method": method,
                    "seed": seed,
                    "point": point,
                    "x": _transform_value(
                        x_value,
                        x_reference,
                        args.x_transform,
                        axis="x",
                        path=path,
                        row_number=row_number,
                    ),
                    "x_reference": x_reference,
                    "y": _transform_value(
                        y_value,
                        y_reference,
                        args.y_transform,
                        axis="y",
                        path=path,
                        row_number=row_number,
                    ),
                    "y_reference": y_reference,
                    "source": f"{path}:{row_number}",
                }
            )
            contributed += 1
        if contributed == 0:
            raise ValueError(
                f"Baseline input {path} has no rows matching the explicitly "
                "selected experiment, dataset, model, and metrics."
            )
    return selected


def _parse_expected_points(
    declarations: list[str], methods: list[str]
) -> dict[str, set[str]]:
    expected = {method: set() for method in methods}
    for declaration in declarations:
        if "=" not in declaration:
            raise ValueError(
                f"Invalid --expected-point {declaration!r}; use METHOD=POINT."
            )
        method, point = (part.strip() for part in declaration.split("=", 1))
        if not method or not point:
            raise ValueError(
                f"Invalid --expected-point {declaration!r}; method and point "
                "must both be non-empty."
            )
        if method not in expected:
            raise ValueError(
                f"--expected-point declares method {method!r}, which is absent "
                "from --methods."
            )
        if point in expected[method]:
            raise ValueError(
                f"Duplicate --expected-point declaration: {method}={point}."
            )
        expected[method].add(point)

    missing = [method for method, points in expected.items() if not points]
    if missing:
        raise ValueError(
            "Every declared method needs at least one --expected-point; missing "
            f"methods: {missing}."
        )
    return expected


def _validate_rows(
    rows: list[dict],
    methods: list[str],
    seeds: list[int],
    expected_points: dict[str, set[str]],
    x_transform: str,
    y_transform: str,
) -> None:
    if len(methods) != len(set(methods)):
        raise ValueError("--methods contains duplicate names.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds contains duplicate values.")

    expected_methods = set(methods)
    observed_methods = {row["method"] for row in rows}
    if observed_methods != expected_methods:
        raise ValueError(
            "Declared and observed methods differ: "
            f"missing={sorted(expected_methods - observed_methods)}, "
            f"unexpected={sorted(observed_methods - expected_methods)}."
        )

    observed_points = {method: set() for method in methods}
    expected_seeds = set(seeds)
    by_method_point = defaultdict(list)
    unique_rows = {}
    for row in rows:
        method = row["method"]
        point = row["point"]
        observed_points[method].add(point)
        key = (method, point, row["seed"])
        if key in unique_rows:
            raise ValueError(
                f"Duplicate method/point/seed row {key!r}: "
                f"{unique_rows[key]} and {row['source']}."
            )
        unique_rows[key] = row["source"]
        by_method_point[(method, point)].append(row)

    for method in methods:
        missing = expected_points[method] - observed_points[method]
        unexpected = observed_points[method] - expected_points[method]
        if missing or unexpected:
            raise ValueError(
                f"Declared and observed points differ for method={method!r}: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
            )

    for method in methods:
        for point in expected_points[method]:
            point_rows = by_method_point[(method, point)]
            observed_seeds = {row["seed"] for row in point_rows}
            if observed_seeds != expected_seeds:
                raise ValueError(
                    f"Seed mismatch for method={method!r}, point={point!r}: "
                    f"missing={sorted(expected_seeds - observed_seeds)}, "
                    f"unexpected={sorted(observed_seeds - expected_seeds)}."
                )

    for axis, transform in (("x", x_transform), ("y", y_transform)):
        if transform != "negative-log-ratio":
            continue
        reference_by_seed = {}
        source_by_seed = {}
        for row in rows:
            seed = row["seed"]
            reference = row[f"{axis}_reference"]
            if seed in reference_by_seed and reference != reference_by_seed[seed]:
                raise ValueError(
                    f"Inconsistent raw {axis}-axis reference for seed={seed}: "
                    f"{reference_by_seed[seed]!r} at {source_by_seed[seed]} and "
                    f"{reference!r} at {row['source']}. All methods and points "
                    "for one dataset/model/seed must use the same reference."
                )
            reference_by_seed[seed] = reference
            source_by_seed[seed] = row["source"]


def _center(values: list[float], aggregate: str) -> float:
    if aggregate == "mean":
        return statistics.fmean(values)
    return float(statistics.median(values))


def _error_bounds(
    values: list[float], center: float, error: str
) -> tuple[float, float]:
    if error == "none":
        return 0.0, 0.0
    if error == "minmax":
        return center - min(values), max(values) - center
    if len(values) < 2:
        raise ValueError(f"--error {error} requires at least two declared seeds.")
    spread = statistics.stdev(values)
    if error == "sem":
        spread /= math.sqrt(len(values))
    return spread, spread


def _aggregate_rows(
    rows: list[dict], methods: list[str], aggregate: str, error: str
) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["point"])].append(row)

    output = {}
    for method in methods:
        points = []
        for (group_method, point), point_rows in grouped.items():
            if group_method != method:
                continue
            x_values = [row["x"] for row in point_rows]
            y_values = [row["y"] for row in point_rows]
            x_center = _center(x_values, aggregate)
            y_center = _center(y_values, aggregate)
            x_lower, x_upper = _error_bounds(x_values, x_center, error)
            y_lower, y_upper = _error_bounds(y_values, y_center, error)
            points.append(
                {
                    "point": point,
                    "x": x_center,
                    "y": y_center,
                    "x_lower": x_lower,
                    "x_upper": x_upper,
                    "y_lower": y_lower,
                    "y_upper": y_upper,
                }
            )
        points.sort(key=lambda item: item["x"])
        output[method] = points
    return output


def _validate_scale(points_by_method, axis: str, scale: str) -> None:
    if scale not in {"log", "logit"}:
        return
    for method, points in points_by_method.items():
        for point in points:
            center = point[axis]
            lower = center - point[f"{axis}_lower"]
            upper = center + point[f"{axis}_upper"]
            if scale == "log" and lower <= 0:
                raise ValueError(
                    f"Log {axis}-scale is invalid for method={method!r}, "
                    f"point={point['point']!r}: lower extent is {lower}."
                )
            if scale == "logit" and not (0 < lower <= center <= upper < 1):
                raise ValueError(
                    f"Logit {axis}-scale requires the complete error extent in "
                    f"(0, 1); method={method!r}, point={point['point']!r}."
                )


def _build_parser(spec: CurveSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=spec.description)
    parser.add_argument(
        "--trim-results", action="append", required=True, help=spec.trim_input_help
    )
    parser.add_argument("--baseline-results", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trim-method", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--expected-point",
        action="append",
        required=True,
        metavar="METHOD=POINT",
        help=(
            "Expected raw point for one method; repeat once per method/point pair."
        ),
    )
    parser.add_argument("--point-column", required=True)
    parser.add_argument("--x-metric", required=True)
    parser.add_argument("--y-metric", required=True)
    parser.add_argument(
        "--x-transform",
        choices=TRANSFORMS,
        default="identity",
        help=(
            "Transform each raw seed row before aggregation. "
            "negative-log-ratio computes -log(value/reference)."
        ),
    )
    parser.add_argument(
        "--y-transform",
        choices=TRANSFORMS,
        default="identity",
        help=(
            "Transform each raw seed row before aggregation. "
            "negative-log-ratio computes -log(value/reference)."
        ),
    )
    parser.add_argument("--trim-dataset-column", default="dataset")
    parser.add_argument("--trim-model-column", default="model_name")
    parser.add_argument("--trim-seed-column", default="seed")
    parser.add_argument(
        "--trim-x-reference-column",
        help="TRIM K0/reference column; required only for x negative-log-ratio.",
    )
    parser.add_argument(
        "--trim-y-reference-column",
        help="TRIM K0/reference column; required only for y negative-log-ratio.",
    )
    parser.add_argument("--aggregate", choices=("mean", "median"), default="mean")
    parser.add_argument(
        "--error", choices=("none", "std", "sem", "minmax"), default="std"
    )
    parser.add_argument(
        "--x-scale", choices=("linear", "log", "symlog", "logit"), default="linear"
    )
    parser.add_argument(
        "--y-scale", choices=("linear", "log", "symlog", "logit"), default="linear"
    )
    parser.add_argument("--xlabel")
    parser.add_argument("--ylabel")
    parser.add_argument("--title")
    parser.add_argument("--figsize", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_curve_cli(spec: CurveSpec) -> None:
    """Parse one experiment command, validate all declared data, and render it."""

    args = _build_parser(spec).parse_args()
    if args.figsize is not None and any(
        not math.isfinite(value) or value <= 0.0 for value in args.figsize
    ):
        raise ValueError("--figsize values must be finite and positive.")
    if args.dpi is not None and args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if len(args.methods) != len(set(args.methods)):
        raise ValueError("--methods contains duplicate names.")
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds contains duplicate values.")
    if args.trim_method not in args.methods:
        raise ValueError("--trim-method must also appear in --methods.")
    if (
        args.x_transform == "negative-log-ratio"
        and not args.trim_x_reference_column
    ):
        raise ValueError(
            "--trim-x-reference-column is required for x negative-log-ratio."
        )
    if (
        args.y_transform == "negative-log-ratio"
        and not args.trim_y_reference_column
    ):
        raise ValueError(
            "--trim-y-reference-column is required for y negative-log-ratio."
        )
    expected_points = _parse_expected_points(args.expected_point, args.methods)

    rows = _load_trim_rows(args) + _load_baseline_rows(args, spec.experiment)
    _validate_rows(
        rows,
        args.methods,
        args.seeds,
        expected_points,
        args.x_transform,
        args.y_transform,
    )
    points_by_method = _aggregate_rows(rows, args.methods, args.aggregate, args.error)
    _validate_scale(points_by_method, "x", args.x_scale)
    _validate_scale(points_by_method, "y", args.y_scale)

    output = Path(args.output).expanduser()
    if not output.suffix:
        raise ValueError("--output must include a Matplotlib-supported file suffix.")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    figure_options = {}
    if args.figsize is not None:
        figure_options["figsize"] = tuple(args.figsize)
    figure, axis = plt.subplots(**figure_options)
    for method in args.methods:
        points = points_by_method[method]
        x_values = [point["x"] for point in points]
        y_values = [point["y"] for point in points]
        if args.error == "none":
            axis.plot(x_values, y_values, label=method)
        else:
            axis.errorbar(
                x_values,
                y_values,
                xerr=[
                    [point["x_lower"] for point in points],
                    [point["x_upper"] for point in points],
                ],
                yerr=[
                    [point["y_lower"] for point in points],
                    [point["y_upper"] for point in points],
                ],
                fmt="-",
                label=method,
            )
    axis.set_xscale(args.x_scale)
    axis.set_yscale(args.y_scale)
    default_x_label = (
        f"-log({args.x_metric}/{args.trim_x_reference_column})"
        if args.x_transform == "negative-log-ratio"
        else args.x_metric
    )
    default_y_label = (
        f"-log({args.y_metric}/{args.trim_y_reference_column})"
        if args.y_transform == "negative-log-ratio"
        else args.y_metric
    )
    axis.set_xlabel(args.xlabel or default_x_label)
    axis.set_ylabel(args.ylabel or default_y_label)
    if args.title is not None:
        axis.set_title(args.title)
    axis.legend()

    save_options = {}
    if args.dpi is not None:
        save_options["dpi"] = args.dpi
    figure.savefig(output, **save_options)
    plt.close(figure)
    print(output)
