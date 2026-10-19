"""Frozen provenance contract for WM-811K deduplication protocol v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_NAME = "wm811k_dedup_8way_v2"
SCHEMA_VERSION = "wm811k_dedup_v2_frozen_config_v1"
MODEL_IDENTIFIER = "open_clip:ViT-B-32-quickgelu:openai"
PREPROCESS_IDENTIFIER = "wm811k_rgb224_fullframe_openai_clip_normalize_v1"
AUGMENTATION_IDENTIFIER = "tip_adapter_random_resized_crop_hflip_clip_normalize_v1"


def _identity_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def clean_feature_identity(
    *,
    content_hash: str,
    model: str,
    weights: str,
    preprocess: str,
    dtype: str,
) -> str:
    """Hash every field needed to prove one clean embedding is reusable."""

    payload = {
        "schema_version": "wm811k_clean_feature_identity_v1",
        "content_hash": str(content_hash).strip(),
        "model": str(model).strip(),
        "weights": str(weights).strip(),
        "preprocess": str(preprocess).strip(),
        "dtype": str(dtype).strip(),
    }
    if any(not value for key, value in payload.items() if key != "schema_version"):
        raise ValueError("clean feature identity fields must not be blank")
    return _identity_sha256(payload)


def augmented_feature_identity(
    *,
    base: str,
    implementation: str,
    seed: int,
    view: int,
    record_index: int,
    order: int,
) -> str:
    """Hash exact stochastic augmentation provenance for one feature row."""

    base_value = str(base).strip()
    implementation_value = str(implementation).strip()
    integer_fields = {
        "seed": int(seed),
        "view": int(view),
        "record_index": int(record_index),
        "order": int(order),
    }
    if not base_value or not implementation_value:
        raise ValueError("augmented feature identity fields must not be blank")
    if any(value < 0 for value in integer_fields.values()):
        raise ValueError("augmented feature identity integer fields must be non-negative")
    return _identity_sha256(
        {
            "schema_version": "wm811k_augmented_feature_identity_v1",
            "base": base_value,
            "implementation": implementation_value,
            **integer_fields,
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return text


@dataclass(frozen=True)
class WM811KDedupV2Contract:
    path: Path
    file_sha256: str
    payload: dict[str, Any]
    protocol: str
    source_pickle_sha256: str
    dedup_audit_sha256: str
    candidate_pool_sha256: str
    manifest_set_sha256: str
    visual_encoder: str
    preprocess: str
    augmentation: str
    linear_c_grid: tuple[float, ...]
    lambda_value: float
    rho: float
    structured_semantic_sha256: str
    qwen_semantic_sha256: str

    def validate_cache_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Reject caches outside the frozen v2 data/model/OOF identity."""

        if metadata.get("protocol") != self.protocol:
            raise ValueError("cache protocol does not match the frozen v2 contract")
        expected = {
            "dedup_audit_sha256": self.dedup_audit_sha256,
            "candidate_pool_sha256": self.candidate_pool_sha256,
            "frozen_config_sha256": self.file_sha256,
            "visual_encoder": self.visual_encoder,
            "preprocess": self.preprocess,
            "augmentation": self.augmentation,
        }
        for name, value in expected.items():
            if metadata.get(name) != value:
                raise ValueError(f"cache {name} does not match the frozen v2 contract")
        for name in (
            "clean_feature_set_sha256",
            "augmented_feature_set_sha256",
        ):
            _require_sha256(metadata.get(name), name=name)
        try:
            shot = int(metadata["shot"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("cache shot is invalid") from exc
        if shot not in (1, 5, 10):
            raise ValueError("cache shot is outside the frozen v2 grid")
        fold_hash = metadata.get("fold_assignment_sha256")
        if shot == 1:
            if fold_hash is not None:
                raise ValueError("one-shot v2 cache must not declare an OOF fold hash")
        else:
            _require_sha256(fold_hash, name="fold_assignment_sha256")


def load_v2_contract(path: str | Path) -> WM811KDedupV2Contract:
    """Load and strictly validate the pre-Test frozen configuration."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"v2 frozen configuration does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("v2 frozen configuration must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported v2 frozen configuration schema")
    if payload.get("protocol") != PROTOCOL_NAME:
        raise ValueError("v2 frozen configuration has the wrong protocol")
    if payload.get("visual_encoder") != MODEL_IDENTIFIER:
        raise ValueError("v2 visual encoder is not the frozen OpenAI CLIP model")
    if payload.get("preprocess") != PREPROCESS_IDENTIFIER:
        raise ValueError("v2 preprocess identifier changed")
    if payload.get("augmentation") != AUGMENTATION_IDENTIFIER:
        raise ValueError("v2 augmentation identifier changed")
    if payload.get("one_shot_anchor") != "visual_prototype":
        raise ValueError("v2 one-shot anchor must be visual_prototype")
    if payload.get("five_ten_shot_anchor") != "linear_probe":
        raise ValueError("v2 five/ten-shot anchor must be linear_probe")
    c_grid = tuple(float(value) for value in payload.get("linear_c_grid", ()))
    if c_grid != (0.01, 0.1, 1.0, 10.0, 100.0):
        raise ValueError("v2 linear C grid changed")
    if int(payload.get("support_augment_views", 0)) != 10:
        raise ValueError("v2 support augmentation budget changed")
    lambda_value = float(payload.get("lambda", float("nan")))
    rho = float(payload.get("rho", float("nan")))
    if lambda_value != 1.0 or rho != 0.01:
        raise ValueError("v2 reliability hyperparameters changed")
    if tuple(payload.get("formal_shots", ())) != (1, 5, 10):
        raise ValueError("v2 formal shots changed")
    if tuple(payload.get("formal_seeds", ())) != tuple(range(10)):
        raise ValueError("v2 formal seeds changed")
    if payload.get("query_labels_used_for_selection") is not False:
        raise ValueError("v2 must freeze query_labels_used_for_selection=false")

    structured = payload.get("structured_semantic")
    qwen = payload.get("qwen_semantic")
    if not isinstance(structured, Mapping) or not isinstance(qwen, Mapping):
        raise ValueError("v2 semantic bundle identities are missing")
    if structured.get("version") != "wm811k_codex_gpt5_structured_v1":
        raise ValueError("v2 structured semantic version changed")
    if qwen.get("version") != "wm811k_qwen_semantic_bundle_v1":
        raise ValueError("v2 Qwen semantic version changed")

    return WM811KDedupV2Contract(
        path=config_path.resolve(),
        file_sha256=_sha256_file(config_path),
        payload=payload,
        protocol=PROTOCOL_NAME,
        source_pickle_sha256=_require_sha256(
            payload.get("source_pickle_sha256"), name="source_pickle_sha256"
        ),
        dedup_audit_sha256=_require_sha256(
            payload.get("dedup_audit_sha256"), name="dedup_audit_sha256"
        ),
        candidate_pool_sha256=_require_sha256(
            payload.get("candidate_pool_sha256"), name="candidate_pool_sha256"
        ),
        manifest_set_sha256=_require_sha256(
            payload.get("manifest_set_sha256"), name="manifest_set_sha256"
        ),
        visual_encoder=MODEL_IDENTIFIER,
        preprocess=PREPROCESS_IDENTIFIER,
        augmentation=AUGMENTATION_IDENTIFIER,
        linear_c_grid=c_grid,
        lambda_value=lambda_value,
        rho=rho,
        structured_semantic_sha256=_require_sha256(
            structured.get("sha256"), name="structured semantic sha256"
        ),
        qwen_semantic_sha256=_require_sha256(
            qwen.get("sha256"), name="Qwen semantic sha256"
        ),
    )
