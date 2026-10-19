"""Create deterministic and auditable WM-811K few-shot manifests."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .images import image_relative_path
from .labels import DEFECT_CLASSES


PROTOCOL_NAME = "wm811k_official_8way_v1"


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def record_to_manifest(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one prepared DataFrame row to a JSON-safe audit record."""

    original_index = int(record["original_index"])
    label = str(record["failure_label"])
    lot_name = _json_scalar(record.get("lotName"))
    if lot_name is not None:
        lot_name = str(lot_name)
    return {
        "original_index": original_index,
        "label": label,
        "source_split": str(record["source_split"]),
        "lot_name": lot_name,
        "wafer_index": _json_scalar(record.get("waferIndex")),
        "image_path": image_relative_path(label, original_index).as_posix(),
    }


def build_few_shot_split(
    records: pd.DataFrame,
    shot: int,
    seed: int,
    image_size: int = 224,
    *,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one strict K-shot support set and fixed official Test query set."""

    if shot <= 0:
        raise ValueError("shot 必须是正整数")
    if seed < 0:
        raise ValueError("seed 必须是非负整数")
    if image_size <= 0:
        raise ValueError("image_size 必须是正整数")

    required = {"original_index", "failure_label", "source_split"}
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"用于划分的数据缺少列: {', '.join(missing)}")

    support: list[dict[str, Any]] = []
    query: list[dict[str, Any]] = []

    for class_index, label in enumerate(DEFECT_CLASSES):
        class_rows = records[records["failure_label"] == label]
        training = class_rows[class_rows["source_split"] == "Training"].sort_values(
            "original_index"
        )
        testing = class_rows[class_rows["source_split"] == "Test"].sort_values(
            "original_index"
        )
        if len(training) < shot:
            raise ValueError(
                f"类别 {label} 的 Training 样本不足：需要 {shot}，实际 {len(training)}"
            )
        if testing.empty:
            raise ValueError(f"类别 {label} 没有 Test/query 样本")

        class_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), class_index])
        )
        selected_positions = class_rng.permutation(len(training))[:shot]
        selected = training.iloc[np.sort(selected_positions)].sort_values(
            "original_index"
        )
        support.extend(
            record_to_manifest(row) for row in selected.to_dict(orient="records")
        )
        query.extend(
            record_to_manifest(row) for row in testing.to_dict(orient="records")
        )

    manifest: dict[str, Any] = {
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
        },
        "source": dict(source_metadata or {}),
        "support": support,
        "query": query,
    }
    validate_split_manifest(manifest)
    return manifest


def validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail loudly when a manifest violates the paper protocol."""

    protocol = manifest.get("protocol", {})
    shot = int(protocol.get("shot", 0))
    support = list(manifest.get("support", []))
    query = list(manifest.get("query", []))

    support_indices = [int(item["original_index"]) for item in support]
    query_indices = [int(item["original_index"]) for item in query]
    if len(support_indices) != len(set(support_indices)):
        raise ValueError("support 内部存在重复样本")
    if len(query_indices) != len(set(query_indices)):
        raise ValueError("query 内部存在重复样本")
    if set(support_indices).intersection(query_indices):
        raise ValueError("support 与 query 存在重叠，检测到数据泄漏")

    for label in DEFECT_CLASSES:
        class_support = [item for item in support if item["label"] == label]
        class_query = [item for item in query if item["label"] == label]
        if len(class_support) != shot:
            raise ValueError(
                f"类别 {label} 的 support 数量应为 {shot}，实际 {len(class_support)}"
            )
        if not class_query:
            raise ValueError(f"类别 {label} 的 query 为空")
        if any(item["source_split"] != "Training" for item in class_support):
            raise ValueError(f"类别 {label} 的 support 含非 Training 样本")
        if any(item["source_split"] != "Test" for item in class_query):
            raise ValueError(f"类别 {label} 的 query 含非 Test 样本")


def write_split_manifest(manifest: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write deterministic UTF-8 JSON using an atomic replace."""

    validate_split_manifest(manifest)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            manifest,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    os.replace(temporary, output)
    return output
