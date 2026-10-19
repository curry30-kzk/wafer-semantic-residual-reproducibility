"""Run the paper-facing WM-811K classwise reliability experiment."""

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

from wm811k.wm_class_reliability import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_LINEAR_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_QWEN_SEMANTICS_PATH,
    QWEN_SEMANTIC_SOURCE,
    SEMANTIC_SOURCE,
    SUPPORTED_SEMANTIC_SOURCES,
    VARIANTS,
    build_wm_caches,
    load_wm_formal_caches,
    run_wm_formal_evaluations,
    summarize_wm_results,
    write_wm_report,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


class WMArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args=args, namespace=namespace)
        if parsed.command == "cache":
            if parsed.run_all and parsed.limit_query is not None:
                self.error("--limit-query is debug-only and cannot be combined with --all")
            if parsed.run_all and (parsed.shot is not None or parsed.seed is not None):
                self.error("--all cannot be combined with --shot or --seed")
            if parsed.run_all and parsed.seeds is not None and len(set(parsed.seeds)) != len(parsed.seeds):
                self.error("--seeds must contain unique values")
            if parsed.shot is None:
                parsed.shot = 5
            if parsed.seed is None:
                parsed.seed = 0
            if parsed.seeds is None:
                parsed.seeds = (0, 1, 2, 3, 4)
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = WMArgumentParser(description="WM-811K 类别级语义可靠性残差主实验")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache", help="构建冻结视觉/语义缓存")
    cache.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    cache.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    cache.add_argument("--linear-baseline-root", type=Path, default=DEFAULT_LINEAR_ROOT)
    cache.add_argument("--semantic-source", choices=SUPPORTED_SEMANTIC_SOURCES, default=SEMANTIC_SOURCE)
    cache.add_argument("--qwen-bundle", type=Path, default=DEFAULT_QWEN_SEMANTICS_PATH)
    cache.add_argument("--reference-visual-cache-root", type=Path, default=None)
    cache.add_argument("--image-root", type=Path, default=None)
    cache.add_argument("--shot", type=int, choices=(1, 5, 10), default=None)
    cache.add_argument("--seed", type=int, choices=range(10), default=None)
    cache.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        choices=range(10),
        default=None,
        help="正式运行的随机种子列表；默认 0 1 2 3 4",
    )
    cache.add_argument("--all", dest="run_all", action="store_true")
    cache.add_argument("--batch-size", type=_positive_int, default=32)
    cache.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    cache.add_argument("--limit-query", type=_positive_int, default=None)
    cache.add_argument("--no-progress", action="store_true")

    run = subparsers.add_parser("run", help="评价全部冻结方法并写出主表")
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--semantic-source", choices=SUPPORTED_SEMANTIC_SOURCES, default=SEMANTIC_SOURCE)
    run.add_argument("--seeds", type=int, nargs="+", choices=range(10), default=None)
    run.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=tuple(VARIANTS),
    )

    report = subparsers.add_parser("report", help="从已写出的结果重建审计报告")
    report.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    report.add_argument("--semantic-source", choices=SUPPORTED_SEMANTIC_SOURCES, default=SEMANTIC_SOURCE)
    report.add_argument("--seeds", type=int, nargs="+", choices=range(10), default=None)

    ablate = subparsers.add_parser("ablate", help="运行并汇总预声明消融")
    ablate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    ablate.add_argument("--semantic-source", choices=SUPPORTED_SEMANTIC_SOURCES, default=SEMANTIC_SOURCE)
    ablate.add_argument("--seeds", type=int, nargs="+", choices=range(10), default=None)
    ablate.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=tuple(VARIANTS),
    )
    return parser


def _written_result_paths(
    output_root: Path,
    *,
    semantic_source: str = SEMANTIC_SOURCE,
    variants: Sequence[str],
) -> list[Path]:
    return [
        output_root / semantic_source / "raw" / variant / f"8way_{shot}shot_seed{seed}.json"
        for variant in variants
        for shot in (1, 5, 10)
        for seed in (0, 1, 2, 3, 4)
    ]


def _load_written_results(
    output_root: Path,
    variants: Sequence[str],
    *,
    semantic_source: str = SEMANTIC_SOURCE,
    formal_seeds: Sequence[int] | None = None,
):
    seeds = tuple((0, 1, 2, 3, 4) if formal_seeds is None else formal_seeds)
    results = {}
    for variant in variants:
        variant_root = output_root / semantic_source / "raw" / variant
        current = []
        for shot in (1, 5, 10):
            for seed in seeds:
                path = variant_root / f"8way_{shot}shot_seed{seed}.json"
                if not path.is_file():
                    raise FileNotFoundError(f"formal result does not exist: {path}")
                current.append(json.loads(path.read_text(encoding="utf-8")))
        results[variant] = current
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "cache":
        paths = build_wm_caches(
            data_root=args.data_root,
            output_root=args.output_root,
            shot=args.shot,
            seed=args.seed,
            formal_seeds=(
                tuple(args.seeds)
                if args.seeds is not None
                else ((0, 1, 2, 3, 4) if args.run_all else (args.seed,))
            ),
            run_all=args.run_all,
            image_root=args.image_root,
            batch_size=args.batch_size,
            device=args.device,
            limit_query=args.limit_query,
            no_progress=args.no_progress,
            linear_baseline_root=args.linear_baseline_root,
            semantic_source=args.semantic_source,
            qwen_bundle_path=args.qwen_bundle,
            reference_visual_cache_root=args.reference_visual_cache_root,
        )
        print(json.dumps({"command": "cache", "cache_count": len(paths), "cache_files": [p.as_posix() for p in paths]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        seeds = tuple(args.seeds) if args.seeds is not None else None
        results = run_wm_formal_evaluations(
            args.output_root,
            variants=args.variants,
            semantic_source=args.semantic_source,
            formal_seeds=seeds,
        )
        report = write_wm_report(
            results, args.output_root, semantic_source=args.semantic_source, formal_seeds=seeds
        )
        print(json.dumps({"command": "run", "variants": list(results), "summary_rows": len(report["summary_rows"])}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "ablate":
        seeds = tuple(args.seeds) if args.seeds is not None else None
        results = run_wm_formal_evaluations(
            args.output_root,
            variants=args.variants,
            semantic_source=args.semantic_source,
            formal_seeds=seeds,
        )
        rows = summarize_wm_results(
            results, args.output_root, semantic_source=args.semantic_source, formal_seeds=seeds
        )
        print(json.dumps({"command": "ablate", "variants": list(results), "summary_rows": len(rows)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "report":
        seeds = tuple(args.seeds) if args.seeds is not None else None
        results = _load_written_results(
            args.output_root,
            tuple(VARIANTS),
            semantic_source=args.semantic_source,
            formal_seeds=seeds,
        )
        report = write_wm_report(
            results, args.output_root, semantic_source=args.semantic_source, formal_seeds=seeds
        )
        print(json.dumps({"command": "report", "summary_rows": len(report["summary_rows"])}, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
