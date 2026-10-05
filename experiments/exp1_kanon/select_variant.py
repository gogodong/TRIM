"""Record a reviewed, validation-only pilot decision before the full sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from experiments.common import write_json
from .run import implementation_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", required=True)
    parser.add_argument("--variant", required=True, choices=("median", "infogain"))
    parser.add_argument("--rationale", required=True, help="Reason for the single choice, using validation evidence.")
    args = parser.parse_args(argv)
    pilot_dir = Path(args.pilot_dir).expanduser().resolve()
    manifest = json.loads((pilot_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("mode") != "pilot" or manifest.get("status") != "complete":
        raise ValueError("Selection requires a completed validation-only Income pilot.")
    if manifest.get("implementation_sha256") != implementation_sha256():
        raise ValueError("Implementation changed since the pilot; repeat the pilot before selecting.")
    raw_path = pilot_dir / "raw_results.csv"
    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(
        row["dataset"] != "income" or row["evaluation_split"] != "validation"
        or row["test_loss"] or row["test_delta_u"] or not row["validation_loss"]
        for row in rows
    ):
        raise ValueError("Pilot evidence must contain validation metrics only, for Income.")
    points = {}
    for row in rows:
        key = (row["model"], row["seed"], row["point"])
        points.setdefault(key, []).append(row["variant"])
    if any(sorted(variants) != ["infogain", "median"] for variants in points.values()):
        raise ValueError("Every pilot model/seed/K must contain exactly both variants.")
    if len(rows) != manifest.get("raw_point_count") or not args.rationale.strip():
        raise ValueError("Pilot evidence is incomplete or the decision rationale is empty.")
    output = pilot_dir / "variant_selection.json"
    if output.exists():
        raise FileExistsError(f"Selection is already fixed: {output}")
    write_json(output, {
        "schema_version": "kanon_selection.v1", "variant": args.variant,
        "evaluation_split": "validation", "rationale": args.rationale.strip(),
        "pilot_dir": str(pilot_dir), "pilot_results_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "implementation_sha256": manifest["implementation_sha256"],
        "income_protocol_signatures": manifest["income_protocol_signatures"],
        "pilot_seeds": sorted({int(row["seed"]) for row in rows}),
        "selection_scope": "one_variant_for_all_datasets_models_seeds_and_Ks",
    })
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
