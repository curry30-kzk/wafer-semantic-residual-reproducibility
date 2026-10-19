"""Deterministic content-unique manifests for WM-811K deduplication v2."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .images import image_relative_path
from .labels import DEFECT_CLASSES


PROTOCOL_NAME = "wm811k_dedup_8way_v2"


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def class_permutation(
    training: pd.DataFrame, *, seed: int, class_index: int
) -> pd.DataFrame:
    """Return one stable permutation used as every K-shot prefix."""

    ordered = training.sort_values("original_index").reset_index(drop=True)
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(class_index)])
    )
    return ordered.iloc[rng.permutation(len(ordered))].reset_index(drop=True)


def _record(
    row: Mapping[str, Any], *, audit_hash: str, pool_hash: str
) -> dict[str, Any]:
    label = str(row["failure_label"])
    original_index = int(row["original_index"])
    lot_name = _json_scalar(row.get("lotName"))
    if lot_name is not None:
        lot_name = str(lot_name)
    return {
        "original_index": original_index,
        "label": label,
        "source_split": str(row["source_split"]),
        "lot_name": lot_name,
        "wafer_index": _json_scalar(row.get("waferIndex")),
        "image_path": image_relative_path(label, original_index)
        .relative_to("images")
        .as_posix(),
        "content_hash": str(row["content_hash"]),
        "content_group_size": int(row["content_group_size"]),
        "representative_original_index": int(
            row["representative_original_index"]
        ),
        "dedup_audit_hash": str(audit_hash),
        "candidate_pool_hash": str(pool_hash),
        "protocol_version": PROTOCOL_NAME,
    }


def _require_hash(value: Any, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex string")
    return text


def build_v2_manifest_grid(
    records: pd.DataFrame,
    *,
    shots: Sequence[int],
    seeds: Sequence[int],
    image_size: int,
    audit_hash: str,
    pool_hash: str,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Build nested manifests from one permutation per seed and class."""

    required = {
        "original_index",
        "failure_label",
        "source_split",
        "content_hash",
        "content_group_size",
        "representative_original_index",
    }
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"v2 records are missing columns: {missing}")
    shot_values = tuple(sorted({int(value) for value in shots}))
    seed_values = tuple(int(value) for value in seeds)
    if not shot_values or shot_values[0] <= 0:
        raise ValueError("shots must contain positive integers")
    if not seed_values or len(seed_values) != len(set(seed_values)) or min(seed_values) < 0:
        raise ValueError("seeds must contain unique non-negative integers")
    if int(image_size) <= 0:
        raise ValueError("image_size must be positive")
    audit_hash = _require_hash(audit_hash, name="audit_hash")
    pool_hash = _require_hash(pool_hash, name="pool_hash")

    training = records[records["source_split"] == "Training"]
    query_frame = records[records["source_split"] == "Test"]
    query: list[dict[str, Any]] = []
    for label in DEFECT_CLASSES:
        class_query = query_frame[query_frame["failure_label"] == label].sort_values(
            "original_index"
        )
        if class_query.empty:
            raise ValueError(f"class {label} has no fixed Test records")
        query.extend(
            _record(row, audit_hash=audit_hash, pool_hash=pool_hash)
            for row in class_query.to_dict(orient="records")
        )

    permutations: dict[tuple[int, str], pd.DataFrame] = {}
    max_shot = max(shot_values)
    for seed in seed_values:
        for class_index, label in enumerate(DEFECT_CLASSES):
            class_training = training[
                training["failure_label"] == label
            ].sort_values("original_index")
            if len(class_training) < max_shot:
                raise ValueError(
                    f"class {label} has {len(class_training)} unique Training contents; "
                    f"{max_shot} required"
                )
            permutations[(seed, label)] = class_permutation(
                class_training, seed=seed, class_index=class_index
            )

    manifests: dict[tuple[int, int], dict[str, Any]] = {}
    for seed in seed_values:
        for shot in shot_values:
            support: list[dict[str, Any]] = []
            for label in DEFECT_CLASSES:
                selected = permutations[(seed, label)].iloc[:shot].sort_values(
                    "original_index"
                )
                support.extend(
                    _record(row, audit_hash=audit_hash, pool_hash=pool_hash)
                    for row in selected.to_dict(orient="records")
                )
            manifest = {
                "protocol": {
                    "name": PROTOCOL_NAME,
                    "n_way": len(DEFECT_CLASSES),
                    "classes": list(DEFECT_CLASSES),
                    "shot": int(shot),
                    "seed": int(seed),
                    "image_size": int(image_size),
                    "support_source": "Training",
                    "query_source": "Test",
                    "nested_support_across_shots": True,
                    "dedup_audit_hash": audit_hash,
                    "candidate_pool_hash": pool_hash,
                },
                "source": dict(source_metadata or {}),
                "support": support,
                "query": list(query),
            }
            validate_v2_manifest(manifest)
            manifests[(shot, seed)] = manifest
    validate_manifest_grid(manifests, shots=shot_values, seeds=seed_values)
    return manifests


def validate_v2_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject index-, path-, content-, label-, or provenance-level leakage."""

    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("name") != PROTOCOL_NAME:
        raise ValueError("manifest protocol is not WM-811K dedup v2")
    classes = tuple(str(value) for value in protocol.get("classes", ()))
    if classes != DEFECT_CLASSES:
        raise ValueError("manifest classes or class order do not match WM-811K")
    shot = int(protocol.get("shot", 0))
    if shot <= 0:
        raise ValueError("manifest shot must be positive")
    audit_hash = _require_hash(protocol.get("dedup_audit_hash"), name="audit hash")
    pool_hash = _require_hash(protocol.get("candidate_pool_hash"), name="pool hash")
    support = list(manifest.get("support", ()))
    query = list(manifest.get("query", ()))
    if not query:
        raise ValueError("manifest query must not be empty")

    required = {
        "original_index",
        "label",
        "source_split",
        "lot_name",
        "wafer_index",
        "image_path",
        "content_hash",
        "content_group_size",
        "representative_original_index",
        "dedup_audit_hash",
        "candidate_pool_hash",
        "protocol_version",
    }
    labels_by_hash: dict[str, set[str]] = {}
    duplicate_content_sections: set[str] = set()
    for section, records, expected_source in (
        ("support", support, "Training"),
        ("query", query, "Test"),
    ):
        paths: set[str] = set()
        hashes: set[str] = set()
        indices: set[int] = set()
        for position, record in enumerate(records):
            if not isinstance(record, Mapping) or set(record) != required:
                raise ValueError(f"{section}[{position}] has invalid fields")
            if record["source_split"] != expected_source:
                raise ValueError(f"{section} contains the wrong source split")
            if record["protocol_version"] != PROTOCOL_NAME:
                raise ValueError(f"{section} has the wrong protocol version")
            if record["dedup_audit_hash"] != audit_hash or record["candidate_pool_hash"] != pool_hash:
                raise ValueError(f"{section} provenance hashes do not match protocol")
            content_hash = str(record["content_hash"])
            if not content_hash:
                raise ValueError(f"{section} has a blank content hash")
            image_path = str(record["image_path"])
            original_index = int(record["original_index"])
            if image_path in paths or original_index in indices:
                raise ValueError(f"{section} contains a duplicate path or index")
            if content_hash in hashes:
                duplicate_content_sections.add(section)
            hashes.add(content_hash)
            paths.add(image_path)
            indices.add(original_index)
            labels_by_hash.setdefault(content_hash, set()).add(str(record["label"]))

    support_hashes = {str(record["content_hash"]) for record in support}
    query_hashes = {str(record["content_hash"]) for record in query}
    if support_hashes & query_hashes:
        raise ValueError("support/query content overlap detected")
    if any(len(labels) > 1 for labels in labels_by_hash.values()):
        raise ValueError("one content hash has multiple labels")
    if duplicate_content_sections:
        raise ValueError(
            f"{sorted(duplicate_content_sections)[0]} contains duplicate content"
        )
    for label in DEFECT_CLASSES:
        class_support = [record for record in support if record["label"] == label]
        class_query = [record for record in query if record["label"] == label]
        if len(class_support) != shot:
            raise ValueError(f"class {label} support count does not equal shot")
        if not class_query:
            raise ValueError(f"class {label} query is empty")


def validate_manifest_grid(
    manifests: Mapping[tuple[int, int], Mapping[str, Any]],
    *,
    shots: Sequence[int],
    seeds: Sequence[int],
) -> None:
    """Validate completeness, fixed Test identity, and shot-prefix nesting."""

    shot_values = tuple(sorted(int(value) for value in shots))
    seed_values = tuple(int(value) for value in seeds)
    expected = {(shot, seed) for shot in shot_values for seed in seed_values}
    if set(manifests) != expected:
        raise ValueError("manifest grid is incomplete or contains unexpected tasks")
    reference_query: list[dict[str, Any]] | None = None
    for key in sorted(expected):
        manifest = manifests[key]
        validate_v2_manifest(manifest)
        query = list(manifest["query"])
        if reference_query is None:
            reference_query = query
        elif query != reference_query:
            raise ValueError("manifest grid does not preserve fixed Test identity")
    for seed in seed_values:
        previous: set[str] = set()
        for shot in shot_values:
            current = {
                str(record["content_hash"])
                for record in manifests[(shot, seed)]["support"]
            }
            if previous and not previous < current:
                raise ValueError("manifest grid support is not strictly nested")
            previous = current


def write_v2_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> Path:
    """Atomically write one stable UTF-8 manifest."""

    validate_v2_manifest(manifest)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    return output


def manifest_set_sha256(manifests: Mapping[tuple[int, int], Mapping[str, Any]]) -> str:
    """Hash the ordered mapping of task names to canonical manifest hashes."""

    entries = [
        {
            "name": f"8way_{shot}shot_seed{seed}.json",
            "sha256": canonical_json_sha256(manifests[(shot, seed)]),
        }
        for shot, seed in sorted(manifests)
    ]
    return canonical_json_sha256(entries)
