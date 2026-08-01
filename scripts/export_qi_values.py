"""Export observed quasi-identifier domains for TRIM datasets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prototype.dataset_registry import DATASETS, build_data_loader, resolve_dataset


CSV_FIELDS = (
    "QI Attribute Name",
    "Attribute Type",
    "Value Count",
    "Attribute Values",
)
DATASET_KEYS = tuple(DATASETS)


@dataclass(frozen=True)
class ExportResult:
    dataset: str
    csv_path: Path
    markdown_path: Path
    attribute_count: int


def _normalize_value(value: object, *, preserve_text: bool) -> object:
    if preserve_text:
        return str(value).strip()
    if pd.isna(value):
        return "<NA>"
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if math.isfinite(numeric) and numeric.is_integer():
        return int(numeric)
    return numeric


def _sort_values(values: Iterable[object]) -> list[object]:
    materialized = list(values)
    numeric = [
        value
        for value in materialized
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if len(numeric) == len(materialized):
        return sorted(materialized)
    return sorted(materialized, key=str)


def _format_codes(values: Sequence[object]) -> str:
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        pieces: list[str] = []
        start = previous = int(values[0])
        for raw_value in values[1:]:
            value = int(raw_value)
            if value == previous + 1:
                previous = value
                continue
            pieces.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = value
        pieces.append(str(start) if start == previous else f"{start}-{previous}")
        return ", ".join(pieces)
    return ", ".join(str(value) for value in values)


def _format_continuous(values: Sequence[object], *, attribute: str) -> str:
    numbers = [
        value
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if len(numbers) != len(values):
        raise ValueError(
            f"Continuous attribute {attribute!r} contains non-numeric values."
        )
    low, high = min(numbers), max(numbers)
    if isinstance(low, float) and low.is_integer():
        low = int(low)
    if isinstance(high, float) and high.is_integer():
        high = int(high)
    return f"{low} ~ {high}"


def _build_rows(data_loader, features: pd.DataFrame) -> list[dict[str, Any]]:
    quasi_identifiers = tuple(
        getattr(data_loader, "qi_attributes", data_loader.feature_columns)
    )
    continuous = set(getattr(data_loader, "numeric_attributes", ()))
    preserve_text = set(
        getattr(data_loader, "string_categorical_attributes", ())
    )
    rows: list[dict[str, Any]] = []
    for attribute in quasi_identifiers:
        if attribute not in features.columns:
            raise KeyError(f"QI attribute {attribute!r} is missing from the dataset.")
        values = _sort_values({
            _normalize_value(
                value,
                preserve_text=attribute in preserve_text,
            )
            for value in features[attribute].dropna().unique().tolist()
        })
        if not values:
            raise ValueError(f"QI attribute {attribute!r} has no observed values.")
        attribute_type = "continuous" if attribute in continuous else "code"
        rendered = (
            _format_continuous(values, attribute=attribute)
            if attribute_type == "continuous"
            else _format_codes(values)
        )
        rows.append({
            "QI Attribute Name": attribute,
            "Attribute Type": attribute_type,
            "Value Count": len(values),
            "Attribute Values": rendered,
        })
    return rows


def _write_exports(
    rows: Sequence[dict[str, Any]],
    *,
    dataset_name: str,
    output_dir: Path,
) -> ExportResult:
    csv_path = output_dir / f"{dataset_name}_qi_values.csv"
    markdown_path = output_dir / f"{dataset_name}_qi_values.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    header = "| " + " | ".join(CSV_FIELDS) + " |"
    border = "| " + " | ".join("---" for _ in CSV_FIELDS) + " |"
    body = [
        "| "
        + " | ".join(
            str(row[field]).replace("|", "\\|") for field in CSV_FIELDS
        )
        + " |"
        for row in rows
    ]
    markdown_path.write_text(
        "\n".join([f"# QI Values: {dataset_name}", "", header, border, *body, ""]),
        encoding="utf-8",
    )
    return ExportResult(dataset_name, csv_path, markdown_path, len(rows))


def main(argv: Sequence[str] | None = None) -> list[ExportResult]:
    parser = argparse.ArgumentParser(
        description="Export observed QI domains for the five TRIM datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_KEYS,
        default=list(DATASET_KEYS),
        help="Dataset keys to export; defaults to all five datasets.",
    )
    parser.add_argument(
        "--data-path",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Override one dataset path; repeat for multiple datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/qi_values",
        help="Directory for the CSV and Markdown exports.",
    )
    args = parser.parse_args(argv)

    data_paths: dict[str, Path] = {}
    for declaration in args.data_path:
        if "=" not in declaration:
            raise ValueError("--data-path must use DATASET=PATH.")
        dataset_value, path_value = declaration.split("=", 1)
        spec = resolve_dataset(dataset_value)
        if spec.key in data_paths:
            raise ValueError(f"Duplicate --data-path for {spec.key!r}.")
        if not path_value.strip():
            raise ValueError(f"--data-path for {spec.key!r} must not be empty.")
        data_paths[spec.key] = Path(path_value).expanduser()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExportResult] = []
    for dataset in args.datasets:
        spec = resolve_dataset(dataset)
        loader = build_data_loader(dataset, data_path=data_paths.get(spec.key))
        features, _labels = loader.load()
        result = _write_exports(
            _build_rows(loader, features),
            dataset_name=spec.full_name,
            output_dir=output_dir,
        )
        print(
            f"{spec.paper_name}: {result.csv_path} and {result.markdown_path} "
            f"({result.attribute_count} QI attributes)"
        )
        results.append(result)
    return results


if __name__ == "__main__":
    main()
