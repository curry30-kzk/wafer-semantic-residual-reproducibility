"""High-level inspect and prepare workflows for WM-811K."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
from tqdm import tqdm

from .images import save_wafer_png
from .labels import (
    DEFECT_CLASSES,
    canonicalize_failure_label,
    prepare_labeled_records,
)
from .pickle_compat import load_wm811k_pickle
from .splits import PROTOCOL_NAME, build_few_shot_split, write_split_manifest


@dataclass(frozen=True)
class PreparationConfig:
    mode: str
    input_path: Path
    output_root: Path
    image_size: int = 224
    shots: tuple[int, ...] = (1, 5, 10)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    samples_per_class: int = 5
    limit_per_class: int | None = None
    overwrite_images: bool = False
    show_progress: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "shots", tuple(self.shots))
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if self.mode not in {"inspect", "prepare"}:
            raise ValueError("mode 只允许 inspect 或 prepare")
        if self.image_size <= 0:
            raise ValueError("image_size 必须是正整数")
        if not self.shots or any(shot <= 0 for shot in self.shots):
            raise ValueError("shots 必须包含正整数")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds 必须包含非负整数")
        if self.samples_per_class <= 0:
            raise ValueError("samples_per_class 必须是正整数")
        if self.limit_per_class is not None and self.limit_per_class <= 0:
            raise ValueError("limit_per_class 必须是正整数")


def create_output_layout(output_root: str | Path) -> dict[str, Path]:
    """Create the stable directory layout consumed by later experiments."""

    root = Path(output_root)
    layout = {
        "root": root,
        "summaries": root / "summaries",
        "samples": root / "samples",
        "images": root / "images",
        "splits": root / "splits",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _write_json(data: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_summaries(
    raw_frame: pd.DataFrame,
    records: pd.DataFrame,
    summaries_dir: Path,
) -> None:
    """Write audit-friendly counts in stable class order."""

    all_labels = raw_frame["failureType"].map(canonicalize_failure_label)
    all_counts = (
        all_labels.value_counts(dropna=False)
        .rename_axis("label")
        .reset_index(name="count")
    )
    all_counts.to_csv(
        summaries_dir / "all_label_counts.csv", index=False, encoding="utf-8-sig"
    )

    class_counts = (
        records["failure_label"]
        .value_counts()
        .reindex(DEFECT_CLASSES, fill_value=0)
        .rename_axis("label")
        .reset_index(name="count")
    )
    class_counts.to_csv(
        summaries_dir / "class_counts.csv", index=False, encoding="utf-8-sig"
    )

    split_counts = pd.crosstab(
        records["failure_label"], records["source_split"]
    ).reindex(index=DEFECT_CLASSES, columns=["Training", "Test"], fill_value=0)
    split_counts = split_counts.rename_axis("label").reset_index()
    split_counts.to_csv(
        summaries_dir / "split_counts.csv", index=False, encoding="utf-8-sig"
    )


def _select_per_class(records: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    ordered = records.sort_values(["failure_label", "original_index"])
    if limit is None:
        return ordered
    return ordered.groupby("failure_label", sort=False, group_keys=False).head(limit)


def _export_records(
    records: pd.DataFrame,
    output_root: Path,
    image_size: int,
    *,
    collection: str,
    overwrite: bool,
    show_progress: bool,
) -> int:
    iterable: Iterable[tuple[int, pd.Series]] = records.iterrows()
    if show_progress:
        iterable = tqdm(
            iterable,
            total=len(records),
            desc=f"导出 {collection}",
            unit="张",
        )
    exported = 0
    for _, row in iterable:
        save_wafer_png(
            row["waferMap"],
            output_root=output_root,
            label=row["failure_label"],
            original_index=int(row["original_index"]),
            image_size=image_size,
            overwrite=overwrite,
            collection=collection,
        )
        exported += 1
    return exported


def export_sample_images(
    records: pd.DataFrame,
    output_root: Path,
    image_size: int,
    samples_per_class: int,
    *,
    overwrite: bool = False,
) -> int:
    selected = _select_per_class(records, samples_per_class)
    return _export_records(
        selected,
        output_root,
        image_size,
        collection="samples",
        overwrite=overwrite,
        show_progress=False,
    )


def export_all_images(
    records: pd.DataFrame,
    output_root: Path,
    image_size: int,
    *,
    limit_per_class: int | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> int:
    selected = _select_per_class(records, limit_per_class)
    return _export_records(
        selected,
        output_root,
        image_size,
        collection="images",
        overwrite=overwrite,
        show_progress=show_progress,
    )


def _source_metadata(
    input_path: Path, raw_frame: pd.DataFrame, records: pd.DataFrame
) -> dict[str, object]:
    return {
        "file_name": input_path.name,
        "input_path": str(input_path.resolve()),
        "file_size_bytes": input_path.stat().st_size if input_path.is_file() else None,
        "dataframe_rows": int(len(raw_frame)),
        "labeled_defect_records": int(len(records)),
    }


def create_all_splits(
    records: pd.DataFrame,
    splits_dir: Path,
    shots: tuple[int, ...],
    seeds: tuple[int, ...],
    image_size: int,
    source_metadata: Mapping[str, object],
) -> list[str]:
    """Create all paper manifests and return paths relative to output root."""

    relative_paths: list[str] = []
    for shot in shots:
        for seed in seeds:
            manifest = build_few_shot_split(
                records,
                shot=shot,
                seed=seed,
                image_size=image_size,
                source_metadata=source_metadata,
            )
            name = f"8way_{shot}shot_seed{seed}.json"
            output = write_split_manifest(manifest, splits_dir / name)
            relative_paths.append((Path("splits") / output.name).as_posix())
    return relative_paths


def run_preparation(
    config: PreparationConfig,
    *,
    frame: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Run inspect or prepare and persist a machine-readable run report."""

    layout = create_output_layout(config.output_root)
    raw_frame = frame if frame is not None else load_wm811k_pickle(config.input_path)
    records = prepare_labeled_records(raw_frame)
    write_summaries(raw_frame, records, layout["summaries"])
    source = _source_metadata(config.input_path, raw_frame, records)

    report: dict[str, object] = {
        "mode": config.mode,
        "protocol": PROTOCOL_NAME,
        "input_path": source["input_path"],
        "dataframe_rows": int(len(raw_frame)),
        "labeled_defect_records": int(len(records)),
        "image_size": config.image_size,
        "classes": list(DEFECT_CLASSES),
        "shots": list(config.shots),
        "seeds": list(config.seeds),
        "debug_limit_per_class": config.limit_per_class,
        "image_export_complete": False,
        "exported_images": 0,
        "manifests": [],
    }

    if config.mode == "inspect":
        report["exported_samples"] = export_sample_images(
            records,
            config.output_root,
            config.image_size,
            config.samples_per_class,
            overwrite=config.overwrite_images,
        )
    else:
        report["exported_images"] = export_all_images(
            records,
            config.output_root,
            config.image_size,
            limit_per_class=config.limit_per_class,
            overwrite=config.overwrite_images,
            show_progress=config.show_progress,
        )
        report["image_export_complete"] = config.limit_per_class is None
        report["manifests"] = create_all_splits(
            records,
            layout["splits"],
            config.shots,
            config.seeds,
            config.image_size,
            source,
        )

    _write_json(report, layout["summaries"] / "preparation_report.json")
    return report
