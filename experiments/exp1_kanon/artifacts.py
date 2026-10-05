"""Compact, compressed Mondrian releases shared across downstream models."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import time

from .mondrian import FittedMondrian


def load_release_payload(path):
    """Read a plain legacy release, compressed tree, or per-model reference."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == "kanon_release_reference.v1":
        shared_path = Path(payload["shared_release_path"])
        if not shared_path.is_absolute():
            shared_path = path.parent / shared_path
        digest = hashlib.sha256(shared_path.read_bytes()).hexdigest()
        if digest != payload["shared_release_sha256"]:
            raise ValueError("Shared Mondrian release checksum differs from its reference.")
        shared = load_release_payload(shared_path)
        if shared.get("cache_key") != payload["cache_key"]:
            raise ValueError("Shared Mondrian release identity differs from its reference.")
        return shared
    return payload


class MondrianReleaseCache:
    """Fit once per exact training data/tree/code/seed/variant/K in a sweep."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def get_or_fit(self, schema, X, y, *, identity):
        cache_key = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(),
        ).hexdigest()
        path = self.directory / f"{cache_key}.json.gz"
        cache_hit = path.exists()
        started = time.perf_counter()
        if cache_hit:
            payload = load_release_payload(path)
            if payload.get("cache_identity") != identity or payload.get("cache_key") != cache_key:
                raise ValueError("Cached Mondrian tree has a different training protocol.")
            if payload["training_row_ids"] != X.index.tolist():
                raise ValueError("Cached Mondrian training row IDs or order changed.")
            fitted = FittedMondrian.from_dict(schema, payload)
            partition_seconds = 0.0
        else:
            fitted = FittedMondrian(schema, k=identity["k"], variant=identity["variant"]).fit(X, y)
            partition_seconds = time.perf_counter() - started
            payload = {
                **fitted.to_dict(), "cache_identity": identity, "cache_key": cache_key,
                "partition_seconds": partition_seconds,
            }
            temporary_path = path.with_suffix(path.suffix + ".partial")
            with gzip.open(temporary_path, "wt", encoding="utf-8", compresslevel=6) as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            temporary_path.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return fitted, {
            "schema_version": "kanon_release_reference.v1",
            "shared_release_path": str(path.resolve()), "shared_release_sha256": digest,
            "cache_key": cache_key, "cache_hit": cache_hit,
            "partition_seconds": partition_seconds,
            "original_partition_seconds": payload["partition_seconds"],
            "tree_cache_seconds": time.perf_counter() - started,
            "shared_release_bytes": path.stat().st_size,
        }
