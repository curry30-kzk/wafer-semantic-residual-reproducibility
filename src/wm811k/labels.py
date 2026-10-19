"""Normalize the unusual nested labels stored in WM-811K."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFECT_CLASSES: tuple[str, ...] = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
)

_FAILURE_LOOKUP = {label.casefold(): label for label in DEFECT_CLASSES}
_FAILURE_LOOKUP.update({"none": "none", "unlabeled": "unlabeled"})
_SPLIT_LOOKUP = {
    "training": "Training",
    "train": "Training",
    "test": "Test",
    "testing": "Test",
    "unlabeled": "unlabeled",
}


def clean_nested_label(value: Any) -> str:
    """Flatten a historical WM-811K label cell into one stripped string."""

    if value is None:
        return "unlabeled"

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "unlabeled"
        value = value.tolist()

    while isinstance(value, (list, tuple)):
        if not value:
            return "unlabeled"
        value = value[0]
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return "unlabeled"
            value = value.tolist()

    if value is None:
        return "unlabeled"
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "unlabeled"

    text = str(value).strip()
    return text if text else "unlabeled"


def canonicalize_failure_label(value: Any) -> str:
    """Return the public spelling for a failure label when it is known."""

    cleaned = clean_nested_label(value)
    return _FAILURE_LOOKUP.get(cleaned.casefold(), cleaned)


def canonicalize_split_label(value: Any) -> str:
    """Normalize the original Training/Test cell to a stable spelling."""

    cleaned = clean_nested_label(value)
    return _SPLIT_LOOKUP.get(cleaned.casefold(), cleaned)


def prepare_labeled_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only the eight labeled defect classes with audit columns."""

    required = {"waferMap", "failureType", "trianTestLabel"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"数据缺少必需列: {', '.join(missing)}")

    records = frame.copy()
    records.insert(0, "original_index", frame.index)
    records["failure_label"] = records["failureType"].map(
        canonicalize_failure_label
    )
    records["source_split"] = records["trianTestLabel"].map(
        canonicalize_split_label
    )
    records = records[records["failure_label"].isin(DEFECT_CLASSES)].copy()

    invalid_splits = sorted(
        set(records["source_split"]).difference({"Training", "Test"})
    )
    if invalid_splits:
        raise ValueError(
            "发现未知的数据划分标签: " + ", ".join(map(str, invalid_splits))
        )

    return records.reset_index(drop=True)
