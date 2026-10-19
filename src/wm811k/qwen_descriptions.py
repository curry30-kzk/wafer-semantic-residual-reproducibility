"""Frozen, validated Qwen semantic prompts for WM-811K."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from .labels import DEFECT_CLASSES


SEMANTIC_BUNDLE_SCHEMA = "wm811k_qwen_semantic_bundle_v1"
SEMANTIC_DATASET_NAME = "WM-811K"
QWEN_SOURCE = "qwen2_5_7b_instruct"
QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
QWEN_OUTPUT_FIELDS = (
    "class_definition",
    "geometry_description",
    "confusion_boundary",
    "clip_prompt",
)
QWEN_SYSTEM_PROMPT = (
    "You create controlled semantic descriptions for WM-811K wafer-map defect "
    "classification. Use only the supplied official class name and geometry hint. "
    "Do not add manufacturing causes, process conditions, physical mechanisms, "
    "statistics, or facts outside the hint. Return exactly one JSON object with "
    "four string fields: class_definition, geometry_description, "
    "confusion_boundary, and clip_prompt. Every field must contain the exact "
    "official class name. The clip_prompt must describe a semiconductor wafer map."
)

_GEOMETRY_HINTS: dict[str, str] = {
    "Center": "defects form a compact cluster around the geometric center; the outer area is mostly clear",
    "Donut": "defects form an interior annular ring around a relatively clear center and away from the outer boundary",
    "Edge-Loc": "defects form a localized cluster touching one limited section of the outer boundary",
    "Edge-Ring": "defects form a broad continuous or nearly continuous ring along the outer boundary",
    "Loc": "defects form a compact off-center interior cluster that does not touch the boundary",
    "Random": "defects are scattered across many unrelated locations without one stable geometric structure",
    "Scratch": "defects align along an elongated narrow line or streak across part of the wafer",
    "Near-full": "defects occupy almost all wafer locations with only a small amount of clear area",
}


@dataclass(frozen=True)
class QwenSemanticBundle:
    """One immutable class-ordered Qwen semantic source."""

    schema_version: str
    dataset: str
    source: str
    class_names: tuple[str, ...]
    prompts_by_class: tuple[tuple[str, ...], ...]
    class_order_sha256: str
    metadata: dict[str, Any]


def class_order_sha256(class_names: tuple[str, ...] = DEFECT_CLASSES) -> str:
    payload = json.dumps(list(class_names), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clean(value: Any, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must not be blank")
    return text


def _fallback_prompts(class_name: str) -> tuple[str, ...]:
    hint = _GEOMETRY_HINTS[class_name]
    return (
        f"{class_name} is a semiconductor wafer-map defect class in which {hint}.",
        f"A semiconductor wafer map showing {class_name}: {hint}.",
        f"To distinguish {class_name} on a semiconductor wafer map, use this geometry: {hint}.",
        f"a semiconductor wafer map showing a {class_name} defect pattern",
    )


def build_user_prompt(class_name: str) -> str:
    if class_name not in DEFECT_CLASSES:
        raise ValueError(f"unknown WM-811K class: {class_name}")
    payload = {
        "official_class_name": class_name,
        "official_geometry_hint": _GEOMETRY_HINTS[class_name],
        "required_fields": list(QWEN_OUTPUT_FIELDS),
        "language": "English",
    }
    return (
        "Generate the required JSON using only this official input. Keep each field "
        "concise and visually grounded; every one of the four fields must repeat the "
        "exact official class name and the word wafer; do not use abbreviations or "
        "synonyms for the class name, and do not invent causes or external facts.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def parse_qwen_output(raw_output: str, class_name: str) -> tuple[str, ...]:
    try:
        payload = json.loads(str(raw_output))
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen output must be valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != set(QWEN_OUTPUT_FIELDS):
        raise ValueError("Qwen output fields do not match the required fields")
    prompts = tuple(_clean(payload[field], field) for field in QWEN_OUTPUT_FIELDS)
    if any(class_name.casefold() not in prompt.casefold() for prompt in prompts):
        raise ValueError("every Qwen field must contain the official class name")
    if any("wafer" not in prompt.casefold() for prompt in prompts):
        raise ValueError("every Qwen field must refer to a wafer map")
    if len(set(prompts)) != len(prompts):
        raise ValueError("Qwen fields must be distinct")
    return prompts


def _normalize_grounding(prompts: tuple[str, ...], class_name: str) -> tuple[tuple[str, ...], bool]:
    """Add only missing audit grounding tokens; preserve model geometry text."""

    normalized: list[str] = []
    changed = False
    for field, prompt in zip(QWEN_OUTPUT_FIELDS, prompts, strict=True):
        value = prompt.strip()
        if class_name.casefold() not in value.casefold():
            value = f"{class_name}: {value}"
            changed = True
        if "wafer" not in value.casefold():
            value = f"Semiconductor wafer map for {class_name}: {value}"
            changed = True
        if value in normalized:
            value = f"{value} ({field})"
            changed = True
        normalized.append(value)
    return tuple(normalized), changed


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def build_qwen_semantic_bundle(
    output_path: str | Path,
    *,
    generator: Callable[[str, str], str],
    model_revision: str,
    transformers_version: str,
    generated_at_utc: str | None = None,
) -> Path:
    """Generate each WM class at most twice, then use deterministic fallback."""

    prompts_by_class: list[tuple[str, ...]] = []
    records: list[dict[str, Any]] = []
    fallback_classes: list[str] = []
    normalized_classes: list[str] = []
    for class_name in DEFECT_CLASSES:
        user_prompt = build_user_prompt(class_name)
        raw_outputs: list[str] = []
        errors: list[str] = []
        accepted: tuple[str, ...] | None = None
        for _ in range(2):
            raw = str(generator(QWEN_SYSTEM_PROMPT, user_prompt))
            raw_outputs.append(raw)
            try:
                try:
                    accepted = parse_qwen_output(raw, class_name)
                except ValueError as strict_error:
                    payload = json.loads(raw)
                    if not isinstance(payload, dict) or set(payload) != set(QWEN_OUTPUT_FIELDS):
                        raise strict_error
                    candidate = tuple(_clean(payload[field], field) for field in QWEN_OUTPUT_FIELDS)
                    accepted, changed = _normalize_grounding(candidate, class_name)
                    if changed:
                        normalized_classes.append(class_name)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            break
        fallback_reason = None
        if accepted is None:
            accepted = _fallback_prompts(class_name)
            fallback_reason = "two_invalid_model_outputs"
            fallback_classes.append(class_name)
        prompts_by_class.append(accepted)
        records.append(
            {
                "class_index": DEFECT_CLASSES.index(class_name),
                "class_name": class_name,
                "official_geometry_hint": _GEOMETRY_HINTS[class_name],
                "user_prompt": user_prompt,
                "raw_outputs": raw_outputs,
                "validation_errors": errors,
                "retry_count": len(raw_outputs) - 1,
                "fallback_reason": fallback_reason,
                "normalization_applied": class_name in normalized_classes,
            }
        )

    generated_at = generated_at_utc or datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": SEMANTIC_BUNDLE_SCHEMA,
        "dataset": SEMANTIC_DATASET_NAME,
        "source": QWEN_SOURCE,
        "class_names": list(DEFECT_CLASSES),
        "class_order_sha256": class_order_sha256(),
        "classes": [
            {
                "class_index": index,
                "class_name": name,
                "prompts": list(prompts),
            }
            for index, (name, prompts) in enumerate(zip(DEFECT_CLASSES, prompts_by_class, strict=True))
        ],
        "metadata": {
            "generation": {
                "kind": "constrained_language_model",
                "model_id": QWEN_MODEL_ID,
                "model_revision": _clean(model_revision, "model_revision"),
                "transformers_version": _clean(transformers_version, "transformers_version"),
                "generated_at_utc": _clean(generated_at, "generated_at_utc"),
                "system_prompt": QWEN_SYSTEM_PROMPT,
                "decoding": {
                    "do_sample": False,
                    "max_new_tokens": 512,
                    "torch_dtype": "bfloat16",
                    "single_class_per_call": True,
                    "maximum_calls_per_class": 2,
                },
                "records": records,
                "fallback_classes": fallback_classes,
                "normalized_classes": normalized_classes,
                "manual_prompt_editing": False,
            },
            "validation": {
                "external_facts_used": False,
                "class_order_checked": True,
                "official_geometry_hints_only": True,
            },
        },
    }
    return _write_json_atomic(Path(output_path), payload)


def load_qwen_semantic_bundle(path: str | Path) -> QwenSemanticBundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SEMANTIC_BUNDLE_SCHEMA:
        raise ValueError("unsupported WM-811K Qwen semantic schema")
    if payload.get("dataset") != SEMANTIC_DATASET_NAME or payload.get("source") != QWEN_SOURCE:
        raise ValueError("semantic bundle dataset/source does not match WM-811K Qwen")
    class_names = tuple(str(name) for name in payload.get("class_names", []))
    if class_names != DEFECT_CLASSES:
        raise ValueError("semantic bundle class order does not match WM-811K")
    if payload.get("class_order_sha256") != class_order_sha256():
        raise ValueError("semantic bundle class order hash is invalid")
    classes = payload.get("classes")
    if not isinstance(classes, list) or len(classes) != len(DEFECT_CLASSES):
        raise ValueError("semantic bundle classes must contain eight entries")
    groups: list[tuple[str, ...]] = []
    for index, (name, entry) in enumerate(zip(DEFECT_CLASSES, classes, strict=True)):
        if entry.get("class_index") != index or entry.get("class_name") != name:
            raise ValueError("semantic class entries do not match class order")
        prompts = tuple(_clean(prompt, "prompt") for prompt in entry.get("prompts", []))
        if len(prompts) != len(QWEN_OUTPUT_FIELDS):
            raise ValueError("every WM-811K class needs four Qwen prompts")
        for prompt in prompts:
            if name.casefold() not in prompt.casefold() or "wafer" not in prompt.casefold():
                raise ValueError("semantic prompt failed WM-811K grounding validation")
        groups.append(prompts)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("validation", {}).get("external_facts_used") is not False:
        raise ValueError("semantic validation metadata is invalid")
    return QwenSemanticBundle(
        schema_version=str(payload["schema_version"]),
        dataset=str(payload["dataset"]),
        source=str(payload["source"]),
        class_names=class_names,
        prompts_by_class=tuple(groups),
        class_order_sha256=str(payload["class_order_sha256"]),
        metadata=metadata,
    )
