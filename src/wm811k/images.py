"""Convert WM-811K integer wafer maps into consistent RGB PNG files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image


WAFER_PALETTE: Mapping[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (210, 218, 226),
    2: (230, 57, 70),
}

_PALETTE_ARRAY = np.array(
    [WAFER_PALETTE[index] for index in range(3)], dtype=np.uint8
)


def wafer_map_to_image(wafer_map: np.ndarray, image_size: int = 224) -> Image.Image:
    """Map values 0/1/2 to RGB and resize without inventing new classes."""

    array = np.asarray(wafer_map)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("waferMap 必须是非空二维数组")
    if image_size <= 0:
        raise ValueError("image_size 必须是正整数")

    unique_values = np.unique(array)
    if not np.all(np.isin(unique_values, (0, 1, 2))):
        raise ValueError(
            "waferMap 只允许包含 0、1、2，实际发现: "
            + ", ".join(map(str, unique_values.tolist()))
        )

    rgb = _PALETTE_ARRAY[array.astype(np.uint8, copy=False)]
    image = Image.fromarray(rgb, mode="RGB")
    if image.size != (image_size, image_size):
        image = image.resize(
            (image_size, image_size), resample=Image.Resampling.NEAREST
        )
    return image


def image_relative_path(
    label: str, original_index: int, collection: str = "images"
) -> Path:
    """Return the stable paper-dataset path for one exported image."""

    if collection not in {"images", "samples"}:
        raise ValueError("collection 只允许 images 或 samples")
    return Path(collection) / label / f"{int(original_index)}.png"


def _is_valid_png(path: Path, expected_size: int) -> bool:
    try:
        with Image.open(path) as image:
            valid = (
                image.format == "PNG"
                and image.mode == "RGB"
                and image.size == (expected_size, expected_size)
            )
            image.verify()
        return valid
    except (OSError, ValueError):
        return False


def save_wafer_png(
    wafer_map: np.ndarray,
    output_root: str | Path,
    label: str,
    original_index: int,
    image_size: int = 224,
    *,
    overwrite: bool = False,
    collection: str = "images",
) -> str:
    """Save one PNG atomically and return its POSIX-style relative path."""

    relative = image_relative_path(label, original_index, collection=collection)
    target = Path(output_root) / relative
    if not overwrite and target.exists() and _is_valid_png(target, image_size):
        return relative.as_posix()

    target.parent.mkdir(parents=True, exist_ok=True)
    image = wafer_map_to_image(wafer_map, image_size=image_size)
    temporary = target.with_name(target.name + ".tmp")
    image.save(temporary, format="PNG", optimize=True)
    os.replace(temporary, target)
    return relative.as_posix()
