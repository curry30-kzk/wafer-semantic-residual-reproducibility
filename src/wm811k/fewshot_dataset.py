"""Validated loading of frozen few-shot task manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


@dataclass(frozen=True)
class ImageRecord:
    """One immutable support or query record resolved from a manifest."""

    absolute_path: Path
    relative_path: str
    label: str
    label_index: int
    original_index: int
    source_split: str
    lot_name: str | None = None
    wafer_index: float | None = None
    content_hash: str | None = None
    content_group_size: int | None = None
    representative_original_index: int | None = None


@dataclass(frozen=True)
class FewShotTask:
    """A complete, validated few-shot classification task."""

    manifest_path: Path
    protocol_name: str
    class_names: tuple[str, ...]
    image_size: int
    shot: int
    seed: int
    support: tuple[ImageRecord, ...]
    query: tuple[ImageRecord, ...]
    support_source: str = "Training"
    query_source: str = "Test"
    dedup_audit_hash: str | None = None
    candidate_pool_hash: str | None = None

    @property
    def n_way(self) -> int:
        return len(self.class_names)


class WaferImageDataset:
    """Minimal lazy Pillow dataset that does not require PyTorch."""

    def __init__(self, records: Sequence[ImageRecord]) -> None:
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Image.Image, int, str]:
        record = self.records[index]
        with Image.open(record.absolute_path) as image:
            rgb_image = image.convert("RGB")
        return rgb_image, record.label_index, record.relative_path


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _parse_records(
    raw_records: Any,
    *,
    data_root: Path,
    class_to_index: dict[str, int],
    expected_source: str,
    section: str,
    require_content_identity: bool = False,
    expected_audit_hash: str | None = None,
    expected_pool_hash: str | None = None,
) -> tuple[ImageRecord, ...]:
    if not isinstance(raw_records, list):
        raise ValueError(f"manifest {section} must be a list")

    records: list[ImageRecord] = []
    seen_paths: set[str] = set()
    for position, raw_record in enumerate(raw_records):
        record = _require_mapping(raw_record, f"{section}[{position}]")
        label = str(record.get("label", ""))
        if label not in class_to_index:
            raise ValueError(f"unknown label in {section}[{position}]: {label!r}")

        source_split = str(record.get("source_split", ""))
        if source_split != expected_source:
            raise ValueError(
                f"{section}[{position}] must come from {expected_source}, "
                f"got {source_split!r}"
            )

        relative_path = str(record.get("image_path", ""))
        if not relative_path:
            raise ValueError(f"{section}[{position}] has no image_path")
        normalized_path = Path(relative_path).as_posix()
        if normalized_path in seen_paths:
            raise ValueError(f"duplicate image path in {section}: {normalized_path}")
        seen_paths.add(normalized_path)

        absolute_path = data_root / Path(relative_path)
        if not absolute_path.is_file():
            raise FileNotFoundError(
                f"manifest image does not exist: {normalized_path} "
                f"(resolved to {absolute_path})"
            )

        try:
            original_index = int(record["original_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{section}[{position}] has invalid original_index"
            ) from exc

        wafer_value = record.get("wafer_index")
        wafer_index = None if wafer_value is None else float(wafer_value)
        lot_value = record.get("lot_name")
        lot_name = None if lot_value is None else str(lot_value)
        content_hash = None
        content_group_size = None
        representative_original_index = None
        if require_content_identity:
            content_hash = str(record.get("content_hash", ""))
            if not content_hash:
                raise ValueError(f"{section}[{position}] has no content_hash")
            try:
                content_group_size = int(record["content_group_size"])
                representative_original_index = int(
                    record["representative_original_index"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{section}[{position}] has invalid content identity"
                ) from exc
            if content_group_size <= 0:
                raise ValueError(
                    f"{section}[{position}] content_group_size must be positive"
                )
            if (
                record.get("dedup_audit_hash") != expected_audit_hash
                or record.get("candidate_pool_hash") != expected_pool_hash
                or record.get("protocol_version") != "wm811k_dedup_8way_v2"
            ):
                raise ValueError(
                    f"{section}[{position}] v2 provenance does not match protocol"
                )
        records.append(
            ImageRecord(
                absolute_path=absolute_path,
                relative_path=normalized_path,
                label=label,
                label_index=class_to_index[label],
                original_index=original_index,
                source_split=source_split,
                lot_name=lot_name,
                wafer_index=wafer_index,
                content_hash=content_hash,
                content_group_size=content_group_size,
                representative_original_index=representative_original_index,
            )
        )
    return tuple(records)


def validate_task(task: FewShotTask) -> None:
    """Enforce the frozen K-shot protocol after parsing."""

    expected_support = task.n_way * task.shot
    if len(task.support) != expected_support:
        raise ValueError(
            f"support must contain exactly {expected_support} records "
            f"({task.n_way}-way {task.shot}-shot), got {len(task.support)}"
        )
    if not task.query:
        raise ValueError("query must not be empty")

    class_counts = [0] * task.n_way
    for record in task.support:
        class_counts[record.label_index] += 1
    for class_index, count in enumerate(class_counts):
        if count != task.shot:
            class_name = task.class_names[class_index]
            raise ValueError(
                f"support class {class_name!r} must contain {task.shot} records, "
                f"got {count}"
            )

    support_paths = {record.relative_path for record in task.support}
    query_paths = {record.relative_path for record in task.query}
    overlap = sorted(support_paths & query_paths)
    if overlap:
        raise ValueError(f"support and query paths overlap: {overlap[0]}")

    if task.protocol_name == "wm811k_dedup_8way_v2":
        if not task.dedup_audit_hash or not task.candidate_pool_hash:
            raise ValueError("v2 task provenance hashes must not be blank")
        support_hashes = [record.content_hash for record in task.support]
        query_hashes = [record.content_hash for record in task.query]
        if any(not value for value in (*support_hashes, *query_hashes)):
            raise ValueError("v2 task records require content hashes")
        if len(support_hashes) != len(set(support_hashes)):
            raise ValueError("v2 support contains duplicate content")
        if len(query_hashes) != len(set(query_hashes)):
            raise ValueError("v2 query contains duplicate content")
        content_overlap = set(support_hashes).intersection(query_hashes)
        if content_overlap:
            raise ValueError(
                f"v2 support/query content overlap: {sorted(content_overlap)[0]}"
            )
        labels_by_hash: dict[str, set[str]] = {}
        for record in (*task.support, *task.query):
            labels_by_hash.setdefault(str(record.content_hash), set()).add(record.label)
        if any(len(labels) > 1 for labels in labels_by_hash.values()):
            raise ValueError("v2 content identity maps to multiple labels")


def load_fewshot_task(
    manifest_path: str | Path,
    data_root: str | Path,
    *,
    image_root: str | Path | None = None,
) -> FewShotTask:
    """Read and validate one frozen JSON manifest without resampling."""

    manifest_path = Path(manifest_path)
    data_root = Path(data_root)
    resolved_image_root = data_root if image_root is None else Path(image_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = _require_mapping(json.load(handle), "manifest")
    protocol = _require_mapping(manifest.get("protocol"), "manifest protocol")

    raw_classes = protocol.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError("protocol classes must be a non-empty list")
    class_names = tuple(str(name) for name in raw_classes)
    if len(set(class_names)) != len(class_names):
        raise ValueError("protocol classes must be unique")
    class_to_index = {name: index for index, name in enumerate(class_names)}

    try:
        shot = int(protocol["shot"])
        seed = int(protocol["seed"])
        image_size = int(protocol["image_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("protocol shot, seed and image_size must be integers") from exc
    if shot <= 0 or image_size <= 0:
        raise ValueError("protocol shot and image_size must be positive")

    n_way = int(protocol.get("n_way", len(class_names)))
    if n_way != len(class_names):
        raise ValueError(
            f"protocol n_way={n_way} does not match {len(class_names)} classes"
        )

    protocol_name = str(protocol.get("name", ""))
    support_source = str(protocol.get("support_source", "Training"))
    query_source = str(protocol.get("query_source", "Test"))
    if not support_source:
        raise ValueError("protocol support_source must not be empty")
    if not query_source:
        raise ValueError("protocol query_source must not be empty")
    require_content_identity = protocol_name == "wm811k_dedup_8way_v2"
    dedup_audit_hash = (
        str(protocol.get("dedup_audit_hash", ""))
        if require_content_identity
        else None
    )
    candidate_pool_hash = (
        str(protocol.get("candidate_pool_hash", ""))
        if require_content_identity
        else None
    )
    if require_content_identity and (
        len(dedup_audit_hash or "") != 64
        or len(candidate_pool_hash or "") != 64
    ):
        raise ValueError("v2 protocol provenance hashes must be SHA-256 strings")

    support = _parse_records(
        manifest.get("support"),
        data_root=resolved_image_root,
        class_to_index=class_to_index,
        expected_source=support_source,
        section="support",
        require_content_identity=require_content_identity,
        expected_audit_hash=dedup_audit_hash,
        expected_pool_hash=candidate_pool_hash,
    )
    query = _parse_records(
        manifest.get("query"),
        data_root=resolved_image_root,
        class_to_index=class_to_index,
        expected_source=query_source,
        section="query",
        require_content_identity=require_content_identity,
        expected_audit_hash=dedup_audit_hash,
        expected_pool_hash=candidate_pool_hash,
    )

    task = FewShotTask(
        manifest_path=manifest_path,
        protocol_name=protocol_name,
        class_names=class_names,
        image_size=image_size,
        shot=shot,
        seed=seed,
        support=support,
        query=query,
        support_source=support_source,
        query_source=query_source,
        dedup_audit_hash=dedup_audit_hash,
        candidate_pool_hash=candidate_pool_hash,
    )
    validate_task(task)
    return task
