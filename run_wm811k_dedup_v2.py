"""Run the isolated, frozen WM-811K deduplication protocol v2 experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wm811k.v2_contract import load_v2_contract  # noqa: E402
from wm811k.class_reliability_experiment import load_task_cache  # noqa: E402
from wm811k.wm_class_reliability import (  # noqa: E402
    DEFAULT_QWEN_SEMANTICS_PATH,
    QWEN_SEMANTIC_SOURCE,
    SEMANTIC_SOURCE,
    SUPPORTED_SEMANTIC_SOURCES,
    VARIANTS,
    WMRunContext,
    build_wm_caches,
    evaluate_wm_cache,
    run_wm_formal_evaluations,
    summarize_wm_results,
    write_wm_report,
)


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "WM811K_dedup_v2"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "processed" / "WM811K" / "images"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "wm811k_dedup_v2"
DEFAULT_LINEAR_ROOT = DEFAULT_OUTPUT_ROOT / "baselines" / "linear_probe" / "raw"
DEFAULT_FROZEN_CONFIG = PROJECT_ROOT / "configs" / "wm811k_dedup_v2_frozen.json"
FORMAL_SEEDS = tuple(range(10))


def build_v2_context(
    *,
    data_root: Path,
    image_root: Path,
    output_root: Path,
    linear_root: Path,
    frozen_config: Path,
) -> WMRunContext:
    """Create the explicit v2 namespace used by every orchestration call."""

    return WMRunContext(
        protocol="wm811k_dedup_8way_v2",
        data_root=Path(data_root),
        image_root=Path(image_root),
        output_root=Path(output_root),
        linear_root=Path(linear_root),
        frozen_config_path=Path(frozen_config),
    )


def _add_shared_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frozen-config", type=Path, default=DEFAULT_FROZEN_CONFIG)


class V2ArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args=args, namespace=namespace)
        if parsed.command == "cache":
            if parsed.run_all and (parsed.shot is not None or parsed.seed is not None):
                self.error("--all cannot be combined with --shot or --seed")
            if parsed.run_all and parsed.limit_query is not None:
                self.error("--limit-query is debug-only and cannot be combined with --all")
            if parsed.shot is None:
                parsed.shot = 5
            if parsed.seed is None:
                parsed.seed = 0
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = V2ArgumentParser(
        description="WM-811K 去重协议 v2：冻结缓存、主方法、消融和审计"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    cache = commands.add_parser("cache", help="构建带完整 v2 溯源的视觉/语义缓存")
    _add_shared_paths(cache)
    cache.add_argument("--linear-baseline-root", type=Path, default=DEFAULT_LINEAR_ROOT)
    cache.add_argument(
        "--semantic-source", choices=SUPPORTED_SEMANTIC_SOURCES, default=SEMANTIC_SOURCE
    )
    cache.add_argument("--qwen-bundle", type=Path, default=DEFAULT_QWEN_SEMANTICS_PATH)
    cache.add_argument("--reference-visual-cache-root", type=Path, default=None)
    cache.add_argument("--shot", type=int, choices=(1, 5, 10), default=None)
    cache.add_argument("--seed", type=int, choices=range(10), default=None)
    cache.add_argument("--seeds", type=int, nargs="+", choices=range(10), default=FORMAL_SEEDS)
    cache.add_argument("--all", dest="run_all", action="store_true")
    cache.add_argument("--batch-size", type=int, default=32)
    cache.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    cache.add_argument("--limit-query", type=int, default=None)
    cache.add_argument("--no-progress", action="store_true")

    for name in ("run", "ablate"):
        command = commands.add_parser(name)
        _add_shared_paths(command)
        command.add_argument(
            "--semantic-source",
            choices=SUPPORTED_SEMANTIC_SOURCES,
            default=SEMANTIC_SOURCE,
        )
        command.add_argument("--seeds", type=int, nargs="+", choices=range(10), default=FORMAL_SEEDS)
        command.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS))

    verify = commands.add_parser("verify-config", help="只读核对冻结配置")
    _add_shared_paths(verify)
    smoke = commands.add_parser("smoke-run", help="只评价一个明确标记的调试缓存")
    _add_shared_paths(smoke)
    smoke.add_argument("--cache", type=Path, required=True)
    return parser


def _context(args: argparse.Namespace) -> WMRunContext:
    linear_root = getattr(args, "linear_baseline_root", DEFAULT_LINEAR_ROOT)
    return build_v2_context(
        data_root=args.data_root,
        image_root=args.image_root,
        output_root=args.output_root,
        linear_root=linear_root,
        frozen_config=args.frozen_config,
    )


def _verified_contract(args: argparse.Namespace):
    context = _context(args)
    contract = load_v2_contract(args.frozen_config)
    if contract.protocol != context.protocol:
        raise ValueError("frozen config protocol does not match the v2 run context")
    report_path = context.data_root / "preparation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"v2 preparation report does not exist: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = {
        "dedup_audit_sha256": contract.dedup_audit_sha256,
        "candidate_pool_sha256": contract.candidate_pool_sha256,
        "manifest_set_sha256": contract.manifest_set_sha256,
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise ValueError(f"v2 preparation {name} does not match frozen config")
    return context, contract


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context, contract = _verified_contract(args)
    if args.command == "verify-config":
        print(
            json.dumps(
                {
                    "protocol": contract.protocol,
                    "frozen_config_sha256": contract.file_sha256,
                    "manifest_set_sha256": contract.manifest_set_sha256,
                    "verified": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "smoke-run":
        cache = load_task_cache(args.cache)
        if not bool((cache.runtime_metadata or {}).get("debug_run")):
            raise ValueError("smoke-run only accepts a cache with debug_run=true")
        result = evaluate_wm_cache(
            cache,
            formal_seeds=FORMAL_SEEDS,
            context=context,
            v2_contract=contract,
        )
        result["debug_run"] = True
        result["paper_eligible"] = False
        semantic_source = str(
            (cache.runtime_metadata or {}).get("semantic_source", SEMANTIC_SOURCE)
        )
        output_path = (
            context.output_root
            / semantic_source
            / "debug"
            / f"8way_{cache.shot}shot_seed{cache.seed}_smoke.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "command": "smoke-run",
                    "output": output_path.as_posix(),
                    "macro_f1": result["metrics"]["macro_f1"],
                    "anchor_macro_f1": result["anchor_metrics"]["macro_f1"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    seeds = tuple(args.seeds)
    if args.command == "cache":
        paths = build_wm_caches(
            context=context,
            v2_contract=contract,
            shot=args.shot,
            seed=args.seed,
            run_all=args.run_all,
            formal_seeds=seeds if args.run_all else (args.seed,),
            batch_size=args.batch_size,
            device=args.device,
            limit_query=args.limit_query,
            no_progress=args.no_progress,
            semantic_source=args.semantic_source,
            qwen_bundle_path=args.qwen_bundle,
            reference_visual_cache_root=args.reference_visual_cache_root,
        )
        print(json.dumps({"command": "cache", "cache_count": len(paths)}, ensure_ascii=False, indent=2))
        return 0
    results = run_wm_formal_evaluations(
        context=context,
        v2_contract=contract,
        variants=args.variants,
        semantic_source=args.semantic_source,
        formal_seeds=seeds,
    )
    if args.command == "ablate":
        rows = summarize_wm_results(
            results,
            semantic_source=args.semantic_source,
            formal_seeds=seeds,
            context=context,
        )
        print(json.dumps({"command": "ablate", "summary_rows": len(rows)}, ensure_ascii=False, indent=2))
        return 0
    report = write_wm_report(
        results,
        semantic_source=args.semantic_source,
        formal_seeds=seeds,
        context=context,
    )
    print(json.dumps({"command": "run", "summary_rows": len(report["summary_rows"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
