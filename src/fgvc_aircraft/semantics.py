"""Frozen and validated semantic prompt bundles for FGVC-Aircraft."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable


SEMANTIC_BUNDLE_SCHEMA = "fgvc_aircraft_semantic_bundle_v1"
SEMANTIC_DATASET_NAME = "FGVC-Aircraft-2013b"
VARIANT_NAME_SOURCE = "variant_name"
HIERARCHY_DETERMINISTIC_SOURCE = "hierarchy_deterministic"
QWEN_SOURCE = "qwen2_5_7b_instruct"
QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
QWEN_OUTPUT_FIELDS = (
    "name_description",
    "hierarchy_description",
    "sibling_distinction",
    "photo_description",
)
QWEN_SYSTEM_PROMPT = (
    "You create controlled FGVC-Aircraft class descriptions. Use only the official "
    "variant, family, manufacturer, and sibling-variant strings supplied by the user. "
    "Do not add appearance, performance, history, geometry, engine, wing, size, or any "
    "other outside fact. Return exactly one JSON object with four string fields: "
    "name_description, hierarchy_description, sibling_distinction, and "
    "photo_description. Every field must explicitly contain the target variant name."
)


@dataclass(frozen=True)
class SemanticBundle:
    """One immutable, class-ordered semantic source."""

    schema_version: str
    dataset: str
    source: str
    class_names: tuple[str, ...]
    prompts_by_class: tuple[tuple[str, ...], ...]
    class_order_sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _HierarchyRow:
    class_index: int
    variant: str
    family: str
    manufacturer: str


def class_order_sha256(class_names: Sequence[str]) -> str:
    """Hash the exact UTF-8 class order used by manifests and prompts."""

    names = tuple(str(name) for name in class_names)
    if not names or any(not name.strip() for name in names):
        raise ValueError("class_names must be non-empty and non-blank")
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")
    payload = json.dumps(
        list(names),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _clean_text(value: Any, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must not be blank")
    return text


def _parse_hierarchy_rows(
    hierarchy: Sequence[Mapping[str, Any]],
) -> tuple[_HierarchyRow, ...]:
    rows = []
    for expected_index, raw in enumerate(hierarchy):
        try:
            class_index = int(raw.get("class_index", expected_index))
        except (TypeError, ValueError) as exc:
            raise ValueError("class_index must be an integer") from exc
        if class_index != expected_index:
            raise ValueError(
                "class_index values must be sequential and match row order"
            )
        rows.append(
            _HierarchyRow(
                class_index=class_index,
                variant=_clean_text(raw.get("variant"), field="variant"),
                family=_clean_text(raw.get("family"), field="family"),
                manufacturer=_clean_text(
                    raw.get("manufacturer"), field="manufacturer"
                ),
            )
        )
    if not rows:
        raise ValueError("hierarchy must contain at least one class")
    variants = tuple(row.variant for row in rows)
    if len(set(variants)) != len(variants):
        raise ValueError("hierarchy variants must be unique")
    return tuple(rows)


def _photo_article(variant: str) -> str:
    return "an" if variant[0].lower() in "aeiou" else "a"


def _variant_name_prompts(row: _HierarchyRow) -> tuple[str, ...]:
    return (f"a photo of {_photo_article(row.variant)} {row.variant} aircraft",)


def _hierarchy_prompts(row: _HierarchyRow) -> tuple[str, ...]:
    return (
        f"a photo of {_photo_article(row.variant)} {row.variant} aircraft",
        f"{row.variant} is an aircraft variant in the {row.family} family",
        f"{row.variant} is a {row.manufacturer} aircraft in the {row.family} family",
        (
            f"a photo of a {row.manufacturer} {row.variant} aircraft "
            f"from the {row.family} family"
        ),
    )


def _bundle_payload(
    rows: Sequence[_HierarchyRow],
    *,
    source: str,
    prompts_by_class: Sequence[Sequence[str]],
) -> dict[str, Any]:
    class_names = tuple(row.variant for row in rows)
    return {
        "schema_version": SEMANTIC_BUNDLE_SCHEMA,
        "dataset": SEMANTIC_DATASET_NAME,
        "source": source,
        "class_names": list(class_names),
        "class_order_sha256": class_order_sha256(class_names),
        "classes": [
            {
                "class_index": row.class_index,
                "variant": row.variant,
                "family": row.family,
                "manufacturer": row.manufacturer,
                "prompts": list(prompts),
            }
            for row, prompts in zip(rows, prompts_by_class, strict=True)
        ],
        "metadata": {
            "generation": {
                "kind": "deterministic_template",
                "model": None,
                "manual_prompt_editing": False,
            },
            "validation": {
                "external_facts_used": False,
                "class_order_checked": True,
                "official_hierarchy_only": True,
            },
        },
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(serialized)
        handle.write("\n")
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def build_control_semantic_bundles(
    hierarchy: Sequence[Mapping[str, Any]], output_root: str | Path
) -> dict[str, Path]:
    """Write name-only and official-hierarchy semantic controls."""

    rows = _parse_hierarchy_rows(hierarchy)
    output_root = Path(output_root)
    paths = {
        VARIANT_NAME_SOURCE: output_root / "variant_name_v1.json",
        HIERARCHY_DETERMINISTIC_SOURCE: (
            output_root / "hierarchy_deterministic_v1.json"
        ),
    }
    payloads = {
        VARIANT_NAME_SOURCE: _bundle_payload(
            rows,
            source=VARIANT_NAME_SOURCE,
            prompts_by_class=tuple(_variant_name_prompts(row) for row in rows),
        ),
        HIERARCHY_DETERMINISTIC_SOURCE: _bundle_payload(
            rows,
            source=HIERARCHY_DETERMINISTIC_SOURCE,
            prompts_by_class=tuple(_hierarchy_prompts(row) for row in rows),
        ),
    }
    for source, path in paths.items():
        _write_json_atomic(path, payloads[source])
    return paths


def _qwen_user_prompt(
    row: _HierarchyRow, siblings: Sequence[str]
) -> str:
    payload = {
        "target_variant": row.variant,
        "official_family": row.family,
        "official_manufacturer": row.manufacturer,
        "official_same_family_variants": list(siblings),
        "output_schema": {field: "string" for field in QWEN_OUTPUT_FIELDS},
    }
    return (
        "Using only this official JSON input, write the required output JSON. "
        "Every output field must explicitly contain the exact target_variant "
        "string; do not omit or abbreviate it in any field. "
        "If no sibling variants are supplied, state only that the target must be "
        "distinguished within its supplied official family; invent nothing.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )


def _parse_qwen_output(raw_output: str, row: _HierarchyRow) -> tuple[str, ...]:
    try:
        payload = json.loads(str(raw_output))
    except json.JSONDecodeError as exc:
        raise ValueError("Qwen output must be one valid JSON object") from exc
    if not isinstance(payload, dict) or set(payload) != set(QWEN_OUTPUT_FIELDS):
        raise ValueError("Qwen output fields do not match the frozen schema")
    prompts = tuple(
        _clean_text(payload[field], field=field) for field in QWEN_OUTPUT_FIELDS
    )
    if any(row.variant not in prompt for prompt in prompts):
        raise ValueError("every Qwen prompt must contain the target variant")
    hierarchy_prompt = prompts[1]
    if row.family not in hierarchy_prompt or row.manufacturer not in hierarchy_prompt:
        raise ValueError(
            "Qwen hierarchy description must contain official family and manufacturer"
        )
    if len(set(prompts)) != len(prompts):
        raise ValueError("Qwen prompt fields must be distinct")
    return prompts


def build_qwen_semantic_bundle(
    hierarchy: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    generator: Callable[[str, str], str],
    model_revision: str,
    transformers_version: str,
    generated_at_utc: str | None = None,
) -> Path:
    """Generate each class at most twice, then use the official hierarchy fallback."""

    rows = _parse_hierarchy_rows(hierarchy)
    revision = _clean_text(model_revision, field="model_revision")
    library_version = _clean_text(
        transformers_version, field="transformers_version"
    )
    generated_at = (
        datetime.now(timezone.utc).isoformat()
        if generated_at_utc is None
        else _clean_text(generated_at_utc, field="generated_at_utc")
    )
    family_members: dict[str, tuple[str, ...]] = {}
    for row in rows:
        family_members[row.family] = tuple(
            candidate.variant
            for candidate in rows
            if candidate.family == row.family
        )

    prompts_by_class: list[tuple[str, ...]] = []
    records: list[dict[str, Any]] = []
    fallback_classes: list[str] = []
    for row in rows:
        siblings = tuple(
            name for name in family_members[row.family] if name != row.variant
        )
        user_prompt = _qwen_user_prompt(row, siblings)
        raw_outputs: list[str] = []
        accepted: tuple[str, ...] | None = None
        errors: list[str] = []
        for _ in range(2):
            raw = str(generator(QWEN_SYSTEM_PROMPT, user_prompt))
            raw_outputs.append(raw)
            try:
                accepted = _parse_qwen_output(raw, row)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            break
        fallback_reason = None
        if accepted is None:
            accepted = _hierarchy_prompts(row)
            fallback_reason = "two_invalid_model_outputs"
            fallback_classes.append(row.variant)
        prompts_by_class.append(accepted)
        records.append(
            {
                "class_index": row.class_index,
                "variant": row.variant,
                "official_family": row.family,
                "official_manufacturer": row.manufacturer,
                "official_same_family_variants": list(siblings),
                "user_prompt": user_prompt,
                "raw_outputs": raw_outputs,
                "validation_errors": errors,
                "retry_count": len(raw_outputs) - 1,
                "fallback_reason": fallback_reason,
            }
        )

    class_names = tuple(row.variant for row in rows)
    payload = {
        "schema_version": SEMANTIC_BUNDLE_SCHEMA,
        "dataset": SEMANTIC_DATASET_NAME,
        "source": QWEN_SOURCE,
        "class_names": list(class_names),
        "class_order_sha256": class_order_sha256(class_names),
        "classes": [
            {
                "class_index": row.class_index,
                "variant": row.variant,
                "family": row.family,
                "manufacturer": row.manufacturer,
                "prompts": list(prompts),
            }
            for row, prompts in zip(rows, prompts_by_class, strict=True)
        ],
        "metadata": {
            "generation": {
                "kind": "constrained_language_model",
                "model_id": QWEN_MODEL_ID,
                "model_revision": revision,
                "transformers_version": library_version,
                "generated_at_utc": generated_at,
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
                "manual_prompt_editing": False,
            },
            "validation": {
                "external_facts_used": False,
                "class_order_checked": True,
                "official_hierarchy_only": True,
                "annotation_consistency_checked": False,
            },
        },
    }
    output = Path(output_path)
    _write_json_atomic(output, payload)
    return output


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid semantic bundle JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("semantic bundle root must be an object")
    return payload


def load_semantic_bundle(
    path: str | Path, expected_class_names: Sequence[str]
) -> SemanticBundle:
    """Load a frozen semantic bundle without reordering or silent repair."""

    path = Path(path)
    payload = _load_payload(path)
    if payload.get("schema_version") != SEMANTIC_BUNDLE_SCHEMA:
        raise ValueError("unsupported semantic bundle schema")
    if payload.get("dataset") != SEMANTIC_DATASET_NAME:
        raise ValueError("semantic bundle dataset does not match FGVC-Aircraft")
    source = _clean_text(payload.get("source"), field="source")
    expected = tuple(str(name) for name in expected_class_names)
    actual_names_raw = payload.get("class_names")
    if not isinstance(actual_names_raw, list):
        raise ValueError("semantic bundle class_names must be a list")
    actual = tuple(str(name) for name in actual_names_raw)
    if actual != expected:
        raise ValueError("semantic bundle class order does not match protocol")
    expected_hash = class_order_sha256(expected)
    if payload.get("class_order_sha256") != expected_hash:
        raise ValueError("semantic bundle class order hash is invalid")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("semantic bundle metadata must be an object")
    validation = metadata.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("semantic bundle validation metadata is required")
    if validation.get("external_facts_used") is not False:
        raise ValueError("semantic bundle external facts are not allowed")

    classes = payload.get("classes")
    if not isinstance(classes, list) or len(classes) != len(expected):
        raise ValueError("semantic bundle classes must match protocol length")
    prompt_groups = []
    for index, (name, entry) in enumerate(zip(expected, classes, strict=True)):
        if not isinstance(entry, dict):
            raise ValueError("semantic class entries must be objects")
        if entry.get("class_index") != index or entry.get("variant") != name:
            raise ValueError("semantic class entries do not match class order")
        raw_prompts = entry.get("prompts")
        if not isinstance(raw_prompts, list) or not raw_prompts:
            raise ValueError("every semantic class needs at least one prompt")
        prompts = tuple(_clean_text(prompt, field="prompt") for prompt in raw_prompts)
        if len(set(prompts)) != len(prompts):
            raise ValueError(f"duplicate prompts for variant {name!r}")
        if any(name not in prompt for prompt in prompts):
            raise ValueError(f"every prompt must contain variant {name!r}")
        prompt_groups.append(prompts)

    if source == QWEN_SOURCE:
        generation = metadata.get("generation")
        if not isinstance(generation, dict):
            raise ValueError("Qwen semantic generation metadata is required")
        if generation.get("model_id") != QWEN_MODEL_ID:
            raise ValueError("Qwen semantic model identifier is invalid")
        _clean_text(generation.get("model_revision"), field="model_revision")
        _clean_text(
            generation.get("transformers_version"),
            field="transformers_version",
        )
        if generation.get("system_prompt") != QWEN_SYSTEM_PROMPT:
            raise ValueError("Qwen system prompt does not match the frozen protocol")
        decoding = generation.get("decoding")
        if not isinstance(decoding, dict) or decoding.get("do_sample") is not False:
            raise ValueError("Qwen decoding metadata is invalid")
        if decoding.get("max_new_tokens") != 512:
            raise ValueError("Qwen max_new_tokens must remain frozen at 512")
        records = generation.get("records")
        if not isinstance(records, list) or len(records) != len(expected):
            raise ValueError("Qwen generation records must match every class")
        declared_fallbacks = generation.get("fallback_classes")
        if not isinstance(declared_fallbacks, list):
            raise ValueError("Qwen fallback_classes must be a list")
        actual_fallbacks = []
        for index, (entry, prompts, record) in enumerate(
            zip(classes, prompt_groups, records, strict=True)
        ):
            if not isinstance(record, dict):
                raise ValueError("Qwen generation records must be objects")
            variant = expected[index]
            if record.get("class_index") != index or record.get("variant") != variant:
                raise ValueError("Qwen generation records do not match class order")
            row = _HierarchyRow(
                class_index=index,
                variant=variant,
                family=_clean_text(entry.get("family"), field="family"),
                manufacturer=_clean_text(
                    entry.get("manufacturer"), field="manufacturer"
                ),
            )
            expected_user_prompt = _qwen_user_prompt(
                row,
                tuple(
                    str(name)
                    for name in record.get("official_same_family_variants", [])
                ),
            )
            if record.get("user_prompt") != expected_user_prompt:
                raise ValueError("Qwen user prompt provenance is invalid")
            raw_outputs = record.get("raw_outputs")
            if not isinstance(raw_outputs, list) or len(raw_outputs) not in (1, 2):
                raise ValueError("Qwen must record one or two raw outputs per class")
            if record.get("retry_count") != len(raw_outputs) - 1:
                raise ValueError("Qwen retry count does not match raw outputs")
            fallback_reason = record.get("fallback_reason")
            if fallback_reason is None:
                accepted = _parse_qwen_output(str(raw_outputs[-1]), row)
                if prompts != accepted:
                    raise ValueError("Qwen accepted prompts do not match raw output")
            elif fallback_reason == "two_invalid_model_outputs":
                if len(raw_outputs) != 2 or prompts != _hierarchy_prompts(row):
                    raise ValueError("Qwen deterministic fallback record is invalid")
                actual_fallbacks.append(variant)
            else:
                raise ValueError("Qwen fallback reason is unsupported")
        if declared_fallbacks != actual_fallbacks:
            raise ValueError("Qwen fallback class list does not match records")

    return SemanticBundle(
        schema_version=SEMANTIC_BUNDLE_SCHEMA,
        dataset=SEMANTIC_DATASET_NAME,
        source=source,
        class_names=actual,
        prompts_by_class=tuple(prompt_groups),
        class_order_sha256=expected_hash,
        metadata=metadata,
    )
