"""Frozen label-only prompt templates for FGVC-Aircraft baselines."""

from __future__ import annotations

from collections.abc import Sequence


AIRCRAFT_PROMPT_VERSION = "fgvc_aircraft_variant_name_prompt_v1"


def build_aircraft_class_prompts(
    class_names: Sequence[str],
) -> tuple[str, ...]:
    """Build one deterministic natural-image prompt per aircraft variant."""

    names = tuple(str(name).strip() for name in class_names)
    if not names or any(not name for name in names):
        raise ValueError("aircraft class_names must not be empty or blank")
    if len(set(names)) != len(names):
        raise ValueError("aircraft class_names must be unique")
    prompts = []
    for name in names:
        article = "an" if name[0].lower() in "aeiou" else "a"
        prompts.append(f"a photo of {article} {name} aircraft")
    return tuple(prompts)
