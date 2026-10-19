"""Validated preparation of official FGVC-Aircraft few-shot manifests."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tarfile
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from wm811k.protocols import FGVC_AIRCRAFT_PROTOCOL_NAME


OFFICIAL_ARCHIVE_URL = (
    "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/"
    "fgvc-aircraft-2013b.tar.gz"
)
OFFICIAL_ARCHIVE_NAME = "fgvc-aircraft-2013b.tar.gz"
OFFICIAL_ARCHIVE_MD5 = "d4acdd33327262359767eeaa97a4f732"


@dataclass(frozen=True)
class DatasetExpectations:
    image_count: int
    variant_count: int
    family_count: int
    manufacturer_count: int
    images_per_variant: int


OFFICIAL_EXPECTATIONS = DatasetExpectations(
    image_count=10_000,
    variant_count=100,
    family_count=70,
    manufacturer_count=30,
    images_per_variant=100,
)


def compute_md5(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming MD5 digest for an archive integrity check."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(
    url: str,
    destination: str | Path,
    *,
    expected_md5: str,
    chunk_size: int = 8 * 1024 * 1024,
    show_progress: bool = True,
) -> Path:
    """Download an archive with resume support and atomic checksum validation."""

    destination = Path(destination)
    if not url:
        raise ValueError("download URL must not be empty")
    if len(expected_md5) != 32:
        raise ValueError("expected MD5 must contain 32 hexadecimal characters")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual_md5 = compute_md5(destination, chunk_size=chunk_size)
        if actual_md5.lower() == expected_md5.lower():
            return destination
        raise ValueError(
            f"existing archive MD5 mismatch: expected {expected_md5}, got {actual_md5}; "
            "the existing file was not modified"
        )

    partial = destination.with_name(destination.name + ".part")
    resume_offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if resume_offset:
        request.add_header("Range", f"bytes={resume_offset}-")
    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", response.getcode())
        append = resume_offset > 0 and status == 206
        if resume_offset > 0 and not append:
            resume_offset = 0
        mode = "ab" if append else "wb"
        length_header = response.headers.get("Content-Length")
        response_length = int(length_header) if length_header else None
        total = (
            resume_offset + response_length
            if response_length is not None
            else None
        )
        downloaded = resume_offset
        with partial.open(mode) as handle:
            while chunk := response.read(chunk_size):
                handle.write(chunk)
                downloaded += len(chunk)
                if show_progress:
                    if total:
                        percent = downloaded * 100.0 / total
                        print(
                            f"\r下载航空器数据：{downloaded}/{total} 字节 "
                            f"({percent:.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\r下载航空器数据：{downloaded} 字节",
                            end="",
                            flush=True,
                        )
    if show_progress:
        print()
    actual_md5 = compute_md5(partial, chunk_size=chunk_size)
    if actual_md5.lower() != expected_md5.lower():
        raise ValueError(
            f"downloaded archive MD5 mismatch: expected {expected_md5}, got {actual_md5}; "
            f"partial file retained at {partial}"
        )
    os.replace(partial, destination)
    return destination


def safe_extract_archive(
    archive_path: str | Path, destination: str | Path
) -> Path:
    """Extract the official tar archive after rejecting unsafe members."""

    archive_path = Path(archive_path)
    destination = Path(destination)
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive does not exist: {archive_path}")
    destination.mkdir(parents=True, exist_ok=True)
    resolved_root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            member_path = Path(member.name)
            target = (destination / member_path).resolve()
            try:
                target.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(
                    f"unsafe archive member outside extraction root: {member.name}"
                ) from exc
            if member_path.is_absolute() or member.isdev() or member.issym() or member.islnk():
                raise ValueError(f"unsafe archive member type: {member.name}")
        handle.extractall(destination, members=members)
    return destination


@dataclass(frozen=True)
class AircraftRecord:
    image_id: str
    variant: str
    family: str
    manufacturer: str
    split: str
    bounding_box: tuple[int, int, int, int]

    @property
    def relative_path(self) -> str:
        return f"images/{self.image_id}.jpg"


def _read_names(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"annotation file does not exist: {path}")
    names = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not names:
        raise ValueError(f"annotation list must not be empty: {path.name}")
    if len(set(names)) != len(names):
        raise ValueError(f"annotation list contains duplicate names: {path.name}")
    return names


def _read_labeled_images(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"annotation file does not exist: {path}")
    labels: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError(f"invalid annotation at {path.name}:{line_number}")
        image_id, label = parts[0], parts[1].strip()
        if image_id in labels:
            raise ValueError(f"duplicate image id in {path.name}: {image_id}")
        labels[image_id] = label
    if not labels:
        raise ValueError(f"annotation file must not be empty: {path.name}")
    return labels


def _read_boxes(path: Path) -> dict[str, tuple[int, int, int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"annotation file does not exist: {path}")
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw_line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"invalid bounding box at {path.name}:{line_number}")
        image_id = parts[0]
        if image_id in boxes:
            raise ValueError(f"duplicate image id in {path.name}: {image_id}")
        try:
            box = tuple(int(value) for value in parts[1:])
        except ValueError as exc:
            raise ValueError(
                f"invalid bounding box at {path.name}:{line_number}"
            ) from exc
        if len(box) != 4 or box[0] > box[2] or box[1] > box[3]:
            raise ValueError(f"invalid bounding box at {path.name}:{line_number}")
        boxes[image_id] = box
    return boxes


def _load_records(data_root: Path) -> tuple[
    tuple[AircraftRecord, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    variants = _read_names(data_root / "variants.txt")
    families = _read_names(data_root / "families.txt")
    manufacturers = _read_names(data_root / "manufacturers.txt")
    boxes = _read_boxes(data_root / "images_box.txt")
    records: list[AircraftRecord] = []
    seen_ids: set[str] = set()
    for split, suffix in (
        ("Training", "train"),
        ("Validation", "val"),
        ("Test", "test"),
    ):
        variant_by_id = _read_labeled_images(
            data_root / f"images_variant_{suffix}.txt"
        )
        family_by_id = _read_labeled_images(
            data_root / f"images_family_{suffix}.txt"
        )
        manufacturer_by_id = _read_labeled_images(
            data_root / f"images_manufacturer_{suffix}.txt"
        )
        image_ids = set(variant_by_id)
        if set(family_by_id) != image_ids or set(manufacturer_by_id) != image_ids:
            raise ValueError(f"{split} hierarchy annotation image ids do not match")
        for image_id in sorted(image_ids):
            if image_id in seen_ids:
                raise ValueError(f"official splits overlap at image id {image_id}")
            seen_ids.add(image_id)
            if image_id not in boxes:
                raise ValueError(f"missing bounding box for image id {image_id}")
            image_path = data_root / "images" / f"{image_id}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(f"annotated image does not exist: {image_path}")
            records.append(
                AircraftRecord(
                    image_id=image_id,
                    variant=variant_by_id[image_id],
                    family=family_by_id[image_id],
                    manufacturer=manufacturer_by_id[image_id],
                    split=split,
                    bounding_box=boxes[image_id],
                )
            )
    if set(boxes) != seen_ids:
        raise ValueError("bounding-box image ids do not match labeled image ids")
    return tuple(records), variants, families, manufacturers


def _validate_dataset(
    records: Sequence[AircraftRecord],
    variants: Sequence[str],
    families: Sequence[str],
    manufacturers: Sequence[str],
    expectations: DatasetExpectations,
) -> dict[str, dict[str, str]]:
    if len(records) != expectations.image_count:
        raise ValueError(
            f"expected {expectations.image_count} images, got {len(records)}"
        )
    if len(variants) != expectations.variant_count:
        raise ValueError(
            f"expected {expectations.variant_count} variants, got {len(variants)}"
        )
    if len(families) != expectations.family_count:
        raise ValueError(
            f"expected {expectations.family_count} families, got {len(families)}"
        )
    if len(manufacturers) != expectations.manufacturer_count:
        raise ValueError(
            "expected "
            f"{expectations.manufacturer_count} manufacturers, got {len(manufacturers)}"
        )
    if {record.variant for record in records} != set(variants):
        raise ValueError("observed variants do not match variants.txt")
    if {record.family for record in records} != set(families):
        raise ValueError("observed families do not match families.txt")
    if {record.manufacturer for record in records} != set(manufacturers):
        raise ValueError("observed manufacturers do not match manufacturers.txt")

    counts = Counter(record.variant for record in records)
    wrong_counts = {
        variant: counts.get(variant, 0)
        for variant in variants
        if counts.get(variant, 0) != expectations.images_per_variant
    }
    if wrong_counts:
        first_variant = next(iter(wrong_counts))
        raise ValueError(
            f"variant {first_variant!r} must contain exactly "
            f"{expectations.images_per_variant} images, got {wrong_counts[first_variant]}"
        )

    hierarchy: dict[str, dict[str, str]] = {}
    for variant in variants:
        pairs = {
            (record.family, record.manufacturer)
            for record in records
            if record.variant == variant
        }
        if len(pairs) != 1:
            raise ValueError(f"variant hierarchy is inconsistent: {variant}")
        family, manufacturer = next(iter(pairs))
        hierarchy[variant] = {
            "family": family,
            "manufacturer": manufacturer,
        }
    return hierarchy


def _manifest_record(record: AircraftRecord) -> dict[str, object]:
    return {
        "bounding_box": list(record.bounding_box),
        "family": record.family,
        "image_path": record.relative_path,
        "label": record.variant,
        "manufacturer": record.manufacturer,
        "original_index": int(record.image_id),
        "source_split": record.split,
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def inspect_aircraft_dataset(
    raw_data_root: str | Path,
    *,
    expectations: DatasetExpectations = OFFICIAL_EXPECTATIONS,
) -> dict[str, object]:
    """Validate an extracted dataset without creating processed artifacts."""

    records, variants, families, manufacturers = _load_records(
        Path(raw_data_root)
    )
    _validate_dataset(
        records, variants, families, manufacturers, expectations
    )
    split_counts = Counter(record.split for record in records)
    return {
        "dataset": "FGVC-Aircraft-2013b",
        "image_count": len(records),
        "variant_count": len(variants),
        "family_count": len(families),
        "manufacturer_count": len(manufacturers),
        "split_counts": {
            split: split_counts[split]
            for split in ("Training", "Validation", "Test")
        },
    }


def prepare_aircraft_dataset(
    raw_data_root: str | Path,
    output_root: str | Path,
    *,
    shots: Sequence[int] = (1, 5, 10),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    expectations: DatasetExpectations = OFFICIAL_EXPECTATIONS,
    protocol_name: str = FGVC_AIRCRAFT_PROTOCOL_NAME,
) -> dict[str, object]:
    """Validate official annotations and write paired few-shot manifests."""

    raw_data_root = Path(raw_data_root)
    output_root = Path(output_root)
    shot_tuple = tuple(int(shot) for shot in shots)
    seed_tuple = tuple(int(seed) for seed in seeds)
    if not shot_tuple or any(shot <= 0 for shot in shot_tuple):
        raise ValueError("shots must contain positive integers")
    if not seed_tuple or any(seed < 0 for seed in seed_tuple):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(shot_tuple)) != len(shot_tuple):
        raise ValueError("shots must be unique")
    if len(set(seed_tuple)) != len(seed_tuple):
        raise ValueError("seeds must be unique")

    records, variants, families, manufacturers = _load_records(raw_data_root)
    hierarchy = _validate_dataset(
        records, variants, families, manufacturers, expectations
    )
    by_split_variant: dict[str, dict[str, list[AircraftRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_split_variant[record.split][record.variant].append(record)
    max_shot = max(shot_tuple)
    for variant in variants:
        training_count = len(by_split_variant["Training"][variant])
        if training_count < max_shot:
            raise ValueError(
                f"variant {variant!r} has only {training_count} Training images; "
                f"at least {max_shot} are required"
            )

    class_hierarchy = [
        {
            "class_index": index,
            "variant": variant,
            "family": hierarchy[variant]["family"],
            "manufacturer": hierarchy[variant]["manufacturer"],
        }
        for index, variant in enumerate(variants)
    ]
    validation_query = [
        _manifest_record(record)
        for variant in variants
        for record in sorted(
            by_split_variant["Validation"][variant],
            key=lambda item: item.image_id,
        )
    ]
    test_query = [
        _manifest_record(record)
        for variant in variants
        for record in sorted(
            by_split_variant["Test"][variant],
            key=lambda item: item.image_id,
        )
    ]

    manifest_count = 0
    for seed in seed_tuple:
        rng = np.random.default_rng(seed)
        selected_by_variant: dict[str, list[AircraftRecord]] = {}
        for variant in variants:
            candidates = sorted(
                by_split_variant["Training"][variant],
                key=lambda item: item.image_id,
            )
            order = rng.permutation(len(candidates))
            selected_by_variant[variant] = [
                candidates[int(index)] for index in order[:max_shot]
            ]
        for shot in sorted(shot_tuple):
            support = [
                _manifest_record(record)
                for variant in variants
                for record in selected_by_variant[variant][:shot]
            ]
            support_sha256 = _canonical_sha256(support)
            common_protocol = {
                "classes": list(variants),
                "class_hierarchy": class_hierarchy,
                "dataset": "FGVC-Aircraft-2013b",
                "image_size": 224,
                "n_way": len(variants),
                "name": protocol_name,
                "seed": seed,
                "shot": shot,
                "support_sha256": support_sha256,
                "support_source": "Training",
            }
            filename = f"{len(variants)}way_{shot}shot_seed{seed}.json"
            for query_source, query, directory in (
                ("Validation", validation_query, "validation"),
                ("Test", test_query, "test"),
            ):
                manifest = {
                    "protocol": {
                        **common_protocol,
                        "query_source": query_source,
                    },
                    "query": query,
                    "support": support,
                }
                _write_json(
                    output_root / "splits" / directory / filename,
                    manifest,
                )
                manifest_count += 1

    split_counts = Counter(record.split for record in records)
    summary: dict[str, object] = {
        "dataset": "FGVC-Aircraft-2013b",
        "protocol": protocol_name,
        "image_count": len(records),
        "variant_count": len(variants),
        "family_count": len(families),
        "manufacturer_count": len(manufacturers),
        "split_counts": {
            split: split_counts[split]
            for split in ("Training", "Validation", "Test")
        },
        "shots": list(shot_tuple),
        "seeds": list(seed_tuple),
        "validation_manifest_count": manifest_count // 2,
        "test_manifest_count": manifest_count // 2,
    }
    _write_json(output_root / "summaries" / "dataset_summary.json", summary)
    _write_csv(
        output_root / "summaries" / "split_counts.csv",
        ("split", "count"),
        (
            {"split": split, "count": split_counts[split]}
            for split in ("Training", "Validation", "Test")
        ),
    )
    _write_csv(
        output_root / "summaries" / "class_hierarchy.csv",
        ("class_index", "variant", "family", "manufacturer"),
        class_hierarchy,
    )
    class_split_counts = Counter((record.variant, record.split) for record in records)
    _write_csv(
        output_root / "summaries" / "class_counts.csv",
        ("class_index", "variant", "Training", "Validation", "Test", "total"),
        (
            {
                "class_index": index,
                "variant": variant,
                "Training": class_split_counts[(variant, "Training")],
                "Validation": class_split_counts[(variant, "Validation")],
                "Test": class_split_counts[(variant, "Test")],
                "total": sum(
                    class_split_counts[(variant, split)]
                    for split in ("Training", "Validation", "Test")
                ),
            }
            for index, variant in enumerate(variants)
        ),
    )
    return summary
