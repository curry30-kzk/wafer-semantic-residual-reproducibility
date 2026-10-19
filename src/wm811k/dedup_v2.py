"""Content-level audit primitives for the WM-811K deduplication v2 protocol."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .labels import DEFECT_CLASSES


CONTENT_HASH_VERSION = "wm811k_matrix_sha256_le_i8_u8_v1"


def canonical_wafer_bytes(values: Any) -> bytes:
    """Return canonical bytes for one valid WM-811K wafer matrix.

    Validation happens before the unsigned-integer conversion so fractional,
    out-of-range, or non-finite values cannot be silently truncated.
    """

    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("waferMap must be a non-empty 2D matrix")
    try:
        numeric = matrix.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("waferMap must be numeric") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("waferMap contains non-finite values")
    if not np.isin(numeric, (0.0, 1.0, 2.0)).all():
        raise ValueError("waferMap values must belong exactly to {0,1,2}")

    shape_bytes = np.asarray(matrix.shape, dtype="<i8").tobytes(order="C")
    matrix_bytes = np.ascontiguousarray(numeric, dtype=np.uint8).tobytes(order="C")
    return shape_bytes + matrix_bytes


def wafer_content_hash(values: Any) -> str:
    """Hash a wafer matrix independently of its original NumPy dtype."""

    return hashlib.sha256(canonical_wafer_bytes(values)).hexdigest()


AUDIT_FILENAMES = (
    "all_content_groups.csv",
    "cross_split_duplicates.csv",
    "label_conflicts.csv",
    "training_duplicate_groups.csv",
    "clean_training_pool.csv",
    "class_counts_before_after.csv",
)


@dataclass(frozen=True)
class DedupResult:
    """Clean active records plus immutable evidence for excluded contents."""

    training: pd.DataFrame
    query: pd.DataFrame
    conflicts: pd.DataFrame
    groups: pd.DataFrame


def sha256_file(path: str | Path) -> str:
    """Hash one file without loading it fully into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_audit_columns(records: pd.DataFrame, *, hashed: bool) -> None:
    required = {"original_index", "failure_label", "source_split", "waferMap"}
    if hashed:
        required.add("content_hash")
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"missing audit columns: {missing}")
    invalid_splits = sorted(set(records["source_split"]) - {"Training", "Test"})
    if invalid_splits:
        raise ValueError(f"invalid source splits: {invalid_splits}")


def attach_content_hashes(records: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical content identity and group audit columns."""

    _require_audit_columns(records, hashed=False)
    result = records.copy()
    result["original_index"] = result["original_index"].astype(np.int64)
    result["content_hash"] = [wafer_content_hash(value) for value in result["waferMap"]]
    sizes = result.groupby("content_hash", sort=True)["content_hash"].transform("size")
    representatives = result.groupby("content_hash", sort=True)[
        "original_index"
    ].transform("min")
    result["content_group_size"] = sizes.astype(np.int64)
    result["representative_original_index"] = representatives.astype(np.int64)
    return result


def _group_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for content_hash, group in records.groupby("content_hash", sort=True):
        ordered = group.sort_values("original_index")
        training_count = int((ordered["source_split"] == "Training").sum())
        test_count = int((ordered["source_split"] == "Test").sum())
        labels = sorted(str(value) for value in ordered["failure_label"].unique())
        rows.append(
            {
                "content_hash": str(content_hash),
                "group_size": int(len(ordered)),
                "training_count": training_count,
                "test_count": test_count,
                "cross_split": bool(training_count and test_count),
                "label_conflict": len(labels) > 1,
                "labels": "|".join(labels),
                "original_indices": "|".join(
                    str(int(value)) for value in ordered["original_index"]
                ),
                "representative_original_index": int(
                    ordered["original_index"].min()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_content_groups(records: pd.DataFrame) -> dict[str, int]:
    """Compute the full-dataset statistics cited by the v2 protocol."""

    _require_audit_columns(records, hashed=True)
    groups = _group_summary(records)
    cross_hashes = set(groups.loc[groups["cross_split"], "content_hash"])
    cross_records = records[records["content_hash"].isin(cross_hashes)]
    return {
        "valid_records": int(len(records)),
        "unique_contents": int(len(groups)),
        "duplicate_content_groups": int((groups["group_size"] > 1).sum()),
        "cross_split_duplicate_groups": int(groups["cross_split"].sum()),
        "cross_split_training_records": int(
            (cross_records["source_split"] == "Training").sum()
        ),
        "cross_split_test_records": int(
            (cross_records["source_split"] == "Test").sum()
        ),
        "label_conflict_groups": int(groups["label_conflict"].sum()),
        "training_internal_duplicate_groups": int(
            (groups["training_count"] > 1).sum()
        ),
        "test_internal_duplicate_groups": int((groups["test_count"] > 1).sum()),
        "fixed_test_records": int((records["source_split"] == "Test").sum()),
    }


def clean_training_candidates(records: pd.DataFrame) -> DedupResult:
    """Apply the fixed-Test-first rule using content identity only."""

    _require_audit_columns(records, hashed=True)
    ordered = records.sort_values(["content_hash", "original_index"]).copy()
    query = ordered[ordered["source_split"] == "Test"].sort_values(
        "original_index"
    )
    training = ordered[ordered["source_split"] == "Training"]
    test_hashes = set(query["content_hash"])
    cross_records = ordered[ordered["content_hash"].isin(test_hashes)]
    conflicts = cross_records.groupby("content_hash", sort=True, group_keys=False).filter(
        lambda group: group["failure_label"].nunique() > 1
    )

    training = training[~training["content_hash"].isin(test_hashes)]
    internal_conflicts = training.groupby(
        "content_hash", sort=True, group_keys=False
    ).filter(lambda group: group["failure_label"].nunique() > 1)
    if not internal_conflicts.empty:
        raise ValueError("remaining Training contains conflicting labels")
    training = training.drop_duplicates("content_hash", keep="first")
    training = training.sort_values(["failure_label", "original_index"]).reset_index(
        drop=True
    )
    return DedupResult(
        training=training,
        query=query.reset_index(drop=True),
        conflicts=conflicts.sort_values(
            ["content_hash", "original_index"]
        ).reset_index(drop=True),
        groups=_group_summary(ordered),
    )


def _evidence_columns(records: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "content_hash",
        "original_index",
        "failure_label",
        "source_split",
        "content_group_size",
        "representative_original_index",
    ]
    return records.loc[:, columns].sort_values(
        ["content_hash", "original_index"]
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))


def write_audit_bundle(
    records: pd.DataFrame,
    output_root: str | Path,
    *,
    source_path: str | Path,
    script_path: str | Path,
    git_commit: str,
) -> dict[str, Any]:
    """Write deterministic full-audit evidence and a hash-indexed report."""

    _require_audit_columns(records, hashed=True)
    source = Path(source_path)
    script = Path(script_path)
    if not source.is_file() or not script.is_file():
        raise FileNotFoundError("source_path and script_path must exist")
    if not str(git_commit).strip():
        raise ValueError("git_commit must not be blank")

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    result = clean_training_candidates(records)
    groups = result.groups
    cross_hashes = set(groups.loc[groups["cross_split"], "content_hash"])
    training_duplicate_hashes = set(
        groups.loc[groups["training_count"] > 1, "content_hash"]
    )
    cross_records = records[records["content_hash"].isin(cross_hashes)]
    training_duplicates = records[
        records["content_hash"].isin(training_duplicate_hashes)
        & (records["source_split"] == "Training")
    ]

    before = (
        records[records["source_split"] == "Training"]["failure_label"]
        .value_counts()
        .reindex(DEFECT_CLASSES, fill_value=0)
    )
    after = (
        result.training["failure_label"]
        .value_counts()
        .reindex(DEFECT_CLASSES, fill_value=0)
    )
    class_counts = pd.DataFrame(
        {
            "failure_label": list(DEFECT_CLASSES),
            "training_before": [int(before[label]) for label in DEFECT_CLASSES],
            "training_after": [int(after[label]) for label in DEFECT_CLASSES],
        }
    )
    class_counts["removed"] = (
        class_counts["training_before"] - class_counts["training_after"]
    )

    frames = {
        "all_content_groups.csv": groups,
        "cross_split_duplicates.csv": _evidence_columns(cross_records),
        "label_conflicts.csv": _evidence_columns(result.conflicts),
        "training_duplicate_groups.csv": _evidence_columns(training_duplicates),
        "clean_training_pool.csv": _evidence_columns(result.training),
        "class_counts_before_after.csv": class_counts,
    }
    for name in AUDIT_FILENAMES:
        _write_csv(root / name, frames[name])
    csv_hashes = {name: sha256_file(root / name) for name in AUDIT_FILENAMES}

    report: dict[str, Any] = {
        "schema_version": "wm811k_dedup_audit_v2",
        "protocol": "wm811k_dedup_8way_v2",
        "content_hash_version": CONTENT_HASH_VERSION,
        "source": {
            "path": str(source.resolve()),
            "size_bytes": int(source.stat().st_size),
            "sha256": sha256_file(source),
        },
        "script": {
            "path": str(script.resolve()),
            "sha256": sha256_file(script),
            "git_commit": str(git_commit),
        },
        "counts": summarize_content_groups(records),
        "class_counts_after": {
            label: int(after[label]) for label in DEFECT_CLASSES
        },
        "csv_sha256": csv_hashes,
    }
    _atomic_write_text(
        root / "audit_report.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return report
