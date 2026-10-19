"""Frozen human-written WM-811K geometry prompts for controlled ablations."""

from __future__ import annotations

from collections.abc import Sequence

from .labels import DEFECT_CLASSES


HUMAN_GEOMETRY_PROMPT_VERSION = "wm811k_human_geometry_prompts_v1"

HUMAN_GEOMETRY_PROMPTS = {
    "Center": (
        "a semiconductor wafer map showing defects concentrated in a compact "
        "cluster at the center, while the surrounding wafer is mostly clear"
    ),
    "Donut": (
        "a semiconductor wafer map showing defects forming an annular "
        "donut-shaped band around the center, with a clearer center and outer region"
    ),
    "Edge-Loc": (
        "a semiconductor wafer map showing a compact localized defect cluster "
        "touching one limited section of the outer wafer edge, rather than "
        "encircling the wafer"
    ),
    "Edge-Ring": (
        "a semiconductor wafer map showing defects forming a continuous or nearly "
        "continuous ring along the outer circumference of the wafer"
    ),
    "Loc": (
        "a semiconductor wafer map showing a compact localized defect cluster "
        "inside the wafer, away from both the center and the outer edge"
    ),
    "Random": (
        "a semiconductor wafer map showing isolated defects scattered irregularly "
        "across many positions without a coherent geometric structure"
    ),
    "Scratch": (
        "a semiconductor wafer map showing a narrow elongated linear streak of "
        "defects crossing part of the wafer like a scratch"
    ),
    "Near-full": (
        "a semiconductor wafer map showing defects covering nearly the entire "
        "wafer area, with only sparse clear sites remaining"
    ),
}


def build_human_geometry_prompts(
    class_names: Sequence[str],
) -> tuple[str, ...]:
    """Return the frozen prompts only for the exact official class grid."""

    names = tuple(str(name) for name in class_names)
    if names != DEFECT_CLASSES:
        raise ValueError(
            "human geometry prompts require the official 8-class WM-811K order"
        )
    return tuple(HUMAN_GEOMETRY_PROMPTS[name] for name in names)
