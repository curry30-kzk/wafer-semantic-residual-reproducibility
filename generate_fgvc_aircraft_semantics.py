"""Create and inspect frozen FGVC-Aircraft semantic prompt bundles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_ROOT))

from fgvc_aircraft.semantics import (  # noqa: E402
    QWEN_MODEL_ID,
    build_control_semantic_bundles,
    build_qwen_semantic_bundle,
    load_semantic_bundle,
)


DEFAULT_HIERARCHY_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "FGVC-Aircraft"
    / "summaries"
    / "class_hierarchy.csv"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "FGVC-Aircraft"
    / "splits"
    / "validation"
    / "100way_1shot_seed0.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "FGVC-Aircraft"
    / "semantics"
)
DEFAULT_QWEN_OUTPUT = DEFAULT_OUTPUT_ROOT / "qwen2_5_7b_instruct_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成并冻结 FGVC-Aircraft 类别语义"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    controls = subparsers.add_parser(
        "controls", help="生成类别名称和官方层级确定性语义"
    )
    controls.add_argument(
        "--hierarchy-csv",
        type=Path,
        default=DEFAULT_HIERARCHY_CSV,
    )
    controls.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    controls.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    qwen = subparsers.add_parser(
        "qwen", help="使用 Qwen2.5-7B 一次性生成受约束类别语义"
    )
    qwen.add_argument(
        "--hierarchy-csv", type=Path, default=DEFAULT_HIERARCHY_CSV
    )
    qwen.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    qwen.add_argument("--device", default="cuda:0")
    qwen.add_argument("--output", type=Path, default=DEFAULT_QWEN_OUTPUT)
    inspect = subparsers.add_parser("inspect", help="严格校验一个冻结语义包")
    inspect.add_argument(
        "--source",
        choices=(
            "variant_name_v1",
            "hierarchy_deterministic_v1",
            "qwen2_5_7b_instruct_v1",
        ),
        default="qwen2_5_7b_instruct_v1",
    )
    inspect.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    inspect.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def _read_hierarchy(path: Path) -> tuple[dict[str, Any], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {
            "class_index",
            "variant",
            "family",
            "manufacturer",
        }
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(
                "class hierarchy CSV needs class_index, variant, family, "
                "and manufacturer columns"
            )
        return tuple(dict(row) for row in reader)


def _read_manifest_classes(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be an object")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("manifest protocol is required")
    classes = protocol.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("manifest protocol classes must be a non-empty list")
    names = tuple(str(name) for name in classes)
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("manifest protocol classes must be unique and non-blank")
    return names


def _run_controls(args: argparse.Namespace) -> dict[str, Any]:
    hierarchy = _read_hierarchy(args.hierarchy_csv)
    class_names = _read_manifest_classes(args.manifest)
    hierarchy_names = tuple(str(row["variant"]) for row in hierarchy)
    if hierarchy_names != class_names:
        raise ValueError(
            "class hierarchy order does not match the frozen protocol manifest"
        )
    paths = build_control_semantic_bundles(hierarchy, args.output_root)
    bundles = {
        source: load_semantic_bundle(path, class_names)
        for source, path in paths.items()
    }
    return {
        "mode": "controls",
        "class_count": len(class_names),
        "class_order_sha256": next(
            iter(bundles.values())
        ).class_order_sha256,
        "bundles": {
            source: path.resolve().as_posix()
            for source, path in paths.items()
        },
    }


def _validate_hierarchy_manifest_order(
    hierarchy: Sequence[dict[str, Any]], class_names: Sequence[str]
) -> None:
    hierarchy_names = tuple(str(row["variant"]) for row in hierarchy)
    if hierarchy_names != tuple(class_names):
        raise ValueError(
            "class hierarchy order does not match the frozen protocol manifest"
        )


def _load_qwen_generator(device: str):
    """Lazily load Qwen so control generation and tests need no Transformers."""

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Qwen generation requires transformers and accelerate; install "
            "the project requirements in the 4090 environment"
        ) from exc
    resolved_device = str(device).strip()
    if not resolved_device:
        raise ValueError("device must not be blank")
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Qwen generation requested CUDA but CUDA is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=resolved_device,
    )
    model.eval()
    revision = str(getattr(model.config, "_commit_hash", None) or "unknown")

    def generate(system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([rendered], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_ids[:, inputs.input_ids.shape[1] :]
        return tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()

    return generate, {
        "model_revision": revision,
        "transformers_version": str(transformers.__version__),
    }


def _run_qwen(args: argparse.Namespace) -> dict[str, Any]:
    hierarchy = _read_hierarchy(args.hierarchy_csv)
    class_names = _read_manifest_classes(args.manifest)
    _validate_hierarchy_manifest_order(hierarchy, class_names)
    generator, provenance = _load_qwen_generator(args.device)
    path = build_qwen_semantic_bundle(
        hierarchy,
        args.output,
        generator=generator,
        model_revision=provenance["model_revision"],
        transformers_version=provenance["transformers_version"],
    )
    bundle = load_semantic_bundle(path, class_names)
    generation = bundle.metadata["generation"]
    return {
        "mode": "qwen",
        "model_id": QWEN_MODEL_ID,
        "model_revision": generation["model_revision"],
        "class_count": len(bundle.class_names),
        "class_order_sha256": bundle.class_order_sha256,
        "fallback_count": len(generation["fallback_classes"]),
        "fallback_classes": generation["fallback_classes"],
        "output": path.resolve().as_posix(),
    }


def _run_inspect(args: argparse.Namespace) -> dict[str, Any]:
    class_names = _read_manifest_classes(args.manifest)
    path = args.output_root / f"{args.source}.json"
    bundle = load_semantic_bundle(path, class_names)
    generation = bundle.metadata.get("generation", {})
    return {
        "mode": "inspect",
        "source": bundle.source,
        "class_count": len(bundle.class_names),
        "prompts_per_class": [
            len(prompts) for prompts in bundle.prompts_by_class
        ],
        "class_order_sha256": bundle.class_order_sha256,
        "fallback_classes": generation.get("fallback_classes", []),
        "external_facts_used": bundle.metadata["validation"][
            "external_facts_used"
        ],
        "path": path.resolve().as_posix(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "controls":
        summary = _run_controls(args)
    elif args.mode == "qwen":
        summary = _run_qwen(args)
    elif args.mode == "inspect":
        summary = _run_inspect(args)
    else:
        raise ValueError(f"unsupported semantics mode: {args.mode}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
