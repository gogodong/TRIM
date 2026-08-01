"""Plot one Exp-1 DP panel from raw result rows.

The reader accepts only CSV, JSON, or JSONL ``raw_results`` tables.  Every
experiment identity, method, seed, and protocol point is declared on the
command line.  Only ``training_mode=dp_sgd`` rows become plotted points.

``--point-field`` identifies the experimental point used for the exact
method/point/seed matrix.  ``--x-field`` is independent: its numeric values are
aggregated across seeds and drawn on the x axis.  This separation permits, for
example, stable protocol-point identities alongside slightly different
realized privacy-accountant values.

For ``--y-metric dp_penalty``, each DP-SGD row is paired with the unique SGD
row having the same method and seed.  The plotted value is computed as DP-SGD
test logloss minus that matched SGD test logloss.  DP-SGD records provide the
plotted points, SGD records provide the paired reference values, and
clipped-SGD records are outside this figure.
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
from matplotlib import pyplot as plt


BASE_FIELDS = {
    "experiment",
    "dataset",
    "method",
    "training_mode",
    "seed",
    "target_k",
    "test_logloss",
}
PAIR_PROTOCOL_FIELDS = (
    "optimizer",
    "learning_rate",
    "weight_decay",
    "epochs",
    "requested_batch_size",
    "poisson_sampling",
    "model_hidden_sizes",
    "evaluation_protocol",
)
KNOWN_TRAINING_MODES = {"sgd", "clipped_sgd", "dp_sgd"}


def _read_rows(paths):
    records = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Input table does not exist: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV table has no header: {path}")
                if any(not str(field).strip() for field in reader.fieldnames):
                    raise ValueError(f"CSV table has a blank header: {path}")
                if len(reader.fieldnames) != len(set(reader.fieldnames)):
                    raise ValueError(f"CSV table has duplicate headers: {path}")
                loaded = list(reader)
            locations = [f"{path}:{row_number}" for row_number in range(2, len(loaded) + 2)]
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                if set(payload) != {"rows"}:
                    raise ValueError(
                        f"JSON object input must contain only a 'rows' member: {path}"
                    )
                payload = payload["rows"]
            if not isinstance(payload, list):
                raise ValueError(f"JSON input must contain a row list: {path}")
            loaded = payload
            locations = [f"{path}:{row_number}" for row_number in range(1, len(loaded) + 1)]
        elif suffix in {".jsonl", ".ndjson"}:
            loaded = []
            locations = []
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        raise ValueError(f"Blank JSONL row at {path}:{line_number}.")
                    loaded.append(json.loads(line))
                    locations.append(f"{path}:{line_number}")
        else:
            raise ValueError(f"Input must be CSV, JSON, or JSONL: {path}")

        if not loaded:
            raise ValueError(f"Input table is empty: {path}")
        for location, row in zip(locations, loaded):
            if not isinstance(row, dict):
                raise ValueError(f"Row at {location} is not an object.")
            if None in row:
                raise ValueError(f"Row at {location} has more values than headers.")
            records.append((location, dict(row)))
    return records


def _require_fields(row, fields, *, location):
    missing = sorted(field for field in fields if field not in row)
    if missing:
        raise ValueError(f"Missing fields at {location}: {missing}")


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
        raise ValueError(
            f"Non-numeric {field!r} at {location}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field!r} at {location}: {value!r}")
    return number


def _integer(value, *, field, location):
    number = _number(value, field=field, location=location)
    if not number.is_integer():
        raise ValueError(f"Non-integer {field!r} at {location}: {value!r}")
    return int(number)


def _identity_token(value, *, field, location):
    text = _text(value, field=field, location=location)
    try:
        number = float(text)
    except ValueError:
        return text
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field!r} at {location}: {value!r}")
    if number.is_integer():
        return str(int(number))
    return format(number, ".17g")


def _boolean(value, *, field, location):
    if isinstance(value, bool):
        return value
    text = _text(value, field=field, location=location).lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"Invalid Boolean {field!r} at {location}: {value!r}")


def _hidden_sizes(value, *, field, location):
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid {field!r} list at {location}: {value!r}"
            ) from exc
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"{field!r} at {location} must be a list.")
    return tuple(
        _integer(item, field=field, location=location) for item in parsed
    )


def _protocol(row, *, location):
    return {
        "optimizer": _text(
            row["optimizer"], field="optimizer", location=location
        ),
        "learning_rate": _number(
            row["learning_rate"], field="learning_rate", location=location
        ),
        "weight_decay": _number(
            row["weight_decay"], field="weight_decay", location=location
        ),
        "epochs": _integer(row["epochs"], field="epochs", location=location),
        "requested_batch_size": _integer(
            row["requested_batch_size"],
            field="requested_batch_size",
            location=location,
        ),
        "poisson_sampling": _boolean(
            row["poisson_sampling"], field="poisson_sampling", location=location
        ),
        "model_hidden_sizes": _hidden_sizes(
            row["model_hidden_sizes"],
            field="model_hidden_sizes",
            location=location,
        ),
        "evaluation_protocol": _text(
            row["evaluation_protocol"],
            field="evaluation_protocol",
            location=location,
        ),
    }


def _unique(values, *, name):
    if not values:
        raise ValueError(f"{name} cannot be empty.")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicates: {values}")
    return values


def _center(values, aggregation):
    if aggregation == "mean":
        return statistics.fmean(values)
    return float(statistics.median(values))


def _error_bounds(values, center, error):
    if error == "none":
        return 0.0, 0.0
    if error == "minmax":
        return center - min(values), max(values) - center
    if len(values) < 2:
        raise ValueError(f"--error {error} requires at least two seeds per point.")
    spread = statistics.stdev(values)
    if error == "sem":
        spread /= math.sqrt(len(values))
    return spread, spread


def _validate_extent(center, lower, upper, scale, *, name):
    if any(not math.isfinite(value) for value in (center, lower, upper)):
        raise ValueError(f"{name} has a non-finite aggregated value or extent.")
    minimum = center - lower
    maximum = center + upper
    if scale == "log" and minimum <= 0.0:
        raise ValueError(f"{name} has a non-positive extent for log scale.")
    if scale == "logit" and not 0.0 < minimum <= center <= maximum < 1.0:
        raise ValueError(f"{name} has an extent outside (0, 1) for logit scale.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-k", required=True, type=int)
    parser.add_argument("--methods", required=True, nargs="+")
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--point-field", required=True)
    parser.add_argument("--expected-points", required=True, nargs="+")
    parser.add_argument("--x-field", required=True)
    parser.add_argument(
        "--y-metric", required=True, choices=("test_logloss", "dp_penalty")
    )
    parser.add_argument(
        "--aggregation", required=True, choices=("mean", "median")
    )
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
    parser.add_argument("--title")
    parser.add_argument("--figsize", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    experiment = _text(args.experiment, field="experiment", location="CLI")
    dataset = _text(args.dataset, field="dataset", location="CLI")
    if args.target_k < 1:
        raise ValueError("--target-k must be positive.")
    methods = _unique(
        [_text(value, field="method", location="CLI") for value in args.methods],
        name="--methods",
    )
    seeds = _unique(list(args.seeds), name="--seeds")
    expected_points = _unique(
        [
            _identity_token(value, field="expected point", location="CLI")
            for value in args.expected_points
        ],
        name="--expected-points",
    )
    point_field = _text(args.point_field, field="point field", location="CLI")
    x_field = _text(args.x_field, field="x field", location="CLI")
    if args.figsize is not None and any(
        not math.isfinite(value) or value <= 0.0 for value in args.figsize
    ):
        raise ValueError("--figsize values must be finite and positive.")
    if args.dpi is not None and args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    required_fields = BASE_FIELDS.union(PAIR_PROTOCOL_FIELDS)
    dp_rows = {}
    sgd_rows = {}
    clipped_rows = {}
    for location, row in _read_rows(args.input):
        _require_fields(row, required_fields, location=location)
        row_experiment = _text(
            row["experiment"], field="experiment", location=location
        )
        row_dataset = _text(row["dataset"], field="dataset", location=location)
        row_target_k = _integer(
            row["target_k"], field="target_k", location=location
        )
        if (
            row_experiment != experiment
            or row_dataset != dataset
            or row_target_k != args.target_k
        ):
            raise ValueError(
                f"Row at {location} does not match the declared "
                "experiment/dataset/target_k."
            )

        method = _text(row["method"], field="method", location=location)
        seed = _integer(row["seed"], field="seed", location=location)
        if method not in methods:
            raise ValueError(f"Unexpected method {method!r} at {location}.")
        if seed not in seeds:
            raise ValueError(f"Unexpected seed {seed!r} at {location}.")
        training_mode = _text(
            row["training_mode"], field="training_mode", location=location
        )
        if training_mode not in KNOWN_TRAINING_MODES:
            raise ValueError(
                f"Unexpected training_mode {training_mode!r} at {location}."
            )
        record = {
            "method": method,
            "seed": seed,
            "test_logloss": _number(
                row["test_logloss"], field="test_logloss", location=location
            ),
            "protocol": _protocol(row, location=location),
            "source": location,
        }

        if training_mode == "dp_sgd":
            _require_fields(row, {point_field, x_field}, location=location)
            point = _identity_token(
                row[point_field], field=point_field, location=location
            )
            if point not in expected_points:
                raise ValueError(f"Unexpected protocol point {point!r} at {location}.")
            record["point"] = point
            record["x"] = _number(row[x_field], field=x_field, location=location)
            key = method, point, seed
            if key in dp_rows:
                raise ValueError(
                    f"Duplicate DP-SGD method/point/seed row {key}: "
                    f"{dp_rows[key]['source']} and {location}."
                )
            dp_rows[key] = record
        elif training_mode == "sgd":
            key = method, seed
            if key in sgd_rows:
                raise ValueError(
                    f"Duplicate SGD method/seed row {key}: "
                    f"{sgd_rows[key]['source']} and {location}."
                )
            sgd_rows[key] = record
        else:
            key = method, seed
            if key in clipped_rows:
                raise ValueError(
                    f"Duplicate clipped-SGD method/seed row {key}: "
                    f"{clipped_rows[key]['source']} and {location}."
                )
            clipped_rows[key] = record

    expected_dp_keys = {
        (method, point, seed)
        for method in methods
        for point in expected_points
        for seed in seeds
    }
    if set(dp_rows) != expected_dp_keys:
        missing = sorted(expected_dp_keys.difference(dp_rows), key=str)
        extra = sorted(set(dp_rows).difference(expected_dp_keys), key=str)
        raise ValueError(
            "Incomplete DP-SGD method/point/seed matrix; "
            f"missing={missing}, extra={extra}."
        )

    if args.y_metric == "dp_penalty":
        expected_sgd_keys = {
            (method, seed) for method in methods for seed in seeds
        }
        if set(sgd_rows) != expected_sgd_keys:
            missing = sorted(expected_sgd_keys.difference(sgd_rows), key=str)
            extra = sorted(set(sgd_rows).difference(expected_sgd_keys), key=str)
            raise ValueError(
                "Incomplete matched SGD method/seed matrix for DP penalty; "
                f"missing={missing}, extra={extra}."
            )
        for (method, point, seed), private in dp_rows.items():
            control = sgd_rows[method, seed]
            mismatched = [
                field
                for field in PAIR_PROTOCOL_FIELDS
                if private["protocol"][field] != control["protocol"][field]
            ]
            if mismatched:
                raise ValueError(
                    "DP-SGD/SGD protocol mismatch for "
                    f"method={method!r}, point={point!r}, seed={seed}: "
                    f"{mismatched}."
                )

    points_by_method = {}
    for method in methods:
        plotted = []
        for point in expected_points:
            records = [dp_rows[method, point, seed] for seed in seeds]
            x_values = [record["x"] for record in records]
            if args.y_metric == "test_logloss":
                y_values = [record["test_logloss"] for record in records]
            else:
                y_values = [
                    record["test_logloss"]
                    - sgd_rows[method, record["seed"]]["test_logloss"]
                    for record in records
                ]
            x_center = _center(x_values, args.aggregation)
            y_center = _center(y_values, args.aggregation)
            x_lower, x_upper = _error_bounds(
                x_values, x_center, args.error
            )
            y_lower, y_upper = _error_bounds(
                y_values, y_center, args.error
            )
            _validate_extent(
                x_center,
                x_lower,
                x_upper,
                args.x_scale,
                name=f"{method}/{point} x",
            )
            _validate_extent(
                y_center,
                y_lower,
                y_upper,
                args.y_scale,
                name=f"{method}/{point} y",
            )
            plotted.append(
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
        plotted.sort(key=lambda item: item["x"])
        points_by_method[method] = plotted

    output = Path(args.output).expanduser()
    if not output.suffix:
        raise ValueError("--output must include a Matplotlib-supported suffix.")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    figure_options = {}
    if args.figsize is not None:
        figure_options["figsize"] = tuple(args.figsize)
    figure, axis = plt.subplots(**figure_options)
    for method in methods:
        points = points_by_method[method]
        x_values = [point["x"] for point in points]
        y_values = [point["y"] for point in points]
        display_method = "TRIM" if method == "trim" else method
        if args.error == "none":
            axis.plot(x_values, y_values, label=display_method)
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
                label=display_method,
            )
    axis.set_xscale(args.x_scale)
    axis.set_yscale(args.y_scale)
    axis.set_xlabel(args.x_label)
    axis.set_ylabel(args.y_label)
    if args.title is not None:
        axis.set_title(args.title)
    axis.legend()

    save_options = {} if args.dpi is None else {"dpi": args.dpi}
    figure.savefig(output, **save_options)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
