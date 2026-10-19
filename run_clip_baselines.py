"""Run frozen CLIP and support-calibrated WM-811K experiments."""

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

from wm811k.baseline_experiment import discover_tasks
from wm811k.clip_experiment import (
    run_domain_geometry_fusion_tasks,
    run_fixed_fusion_tasks,
    run_llm_reliability_fusion_tasks,
    run_llm_structured_fixed_fusion_tasks,
    run_visual_prototype_tasks,
    run_zero_shot_tasks,
)
from wm811k.clip_features import FrozenClipEncoder
from wm811k.experiment_reporting import claim_output_root
from wm811k.resnet18_features import resolve_device


class ClipArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args=args, namespace=namespace)
        # Zero-shot has no support-dependent seed protocol and remains tied
        # to the original five-seed WM-811K manifest grid.  Visual and
        # fusion modes may consume the later ten-seed extension (0--9).
        if parsed.mode == "zero_shot":
            if parsed.seed is not None and parsed.seed >= 5:
                self.error("zero_shot supports seeds 0--4 only")
            if parsed.seeds is not None and any(seed >= 5 for seed in parsed.seeds):
                self.error("zero_shot supports seeds 0--4 only")
        if parsed.run_all and (parsed.shot is not None or parsed.seed is not None):
            self.error("--all cannot be combined with --shot or --seed")
        if parsed.run_all and parsed.limit_query is not None:
            self.error("--limit-query is debug-only and cannot be combined with --all")
        if parsed.shot is None:
            parsed.shot = 5
        if parsed.seed is None:
            parsed.seed = 0
        return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def default_output_root(mode: str) -> Path:
    if mode == "zero_shot":
        return PROJECT_ROOT / "results" / "clip_zero_shot"
    if mode == "visual_prototype":
        return PROJECT_ROOT / "results" / "clip_visual_prototype"
    if mode == "fixed_fusion":
        return PROJECT_ROOT / "results" / "clip_fixed_fusion"
    if mode == "domain_geometry_fusion":
        return PROJECT_ROOT / "results" / "clip_domain_geometry_fusion"
    if mode == "llm_structured_fixed_fusion":
        return PROJECT_ROOT / "results" / "clip_llm_structured_fixed_fusion"
    if mode == "llm_reliability_fusion":
        return PROJECT_ROOT / "results" / "clip_llm_reliability_fusion"
    raise ValueError(f"unsupported CLIP baseline mode: {mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = ClipArgumentParser(
        description=(
            "Frozen OpenAI CLIP ViT-B/32 baselines for the WM-811K "
            "8-way protocol"
        )
    )
    parser.add_argument(
        "--mode",
        choices=(
            "zero_shot",
            "visual_prototype",
            "fixed_fusion",
            "domain_geometry_fusion",
            "llm_structured_fixed_fusion",
            "llm_reliability_fusion",
        ),
        required=True,
        help=(
            "Zero-shot text prototypes, K-shot visual prototypes, or fixed "
            "0.5/0.5 visual-text score fusion, structured LLM text ablation, "
            "or support-only reliability fusion"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "WM811K",
        help="prepared WM-811K directory containing images/ and splits/",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="optional shared image directory when manifests live elsewhere",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="generated result directory; defaults to a mode-specific folder",
    )
    parser.add_argument("--shot", type=int, choices=(1, 5, 10), default=None)
    # The original WM-811K baseline protocol is five-seed (0--4); visual and
    # fusion re-evaluations may additionally use the ten-seed extension (0--9).
    parser.add_argument("--seed", type=int, choices=range(10), default=None)
    parser.add_argument("--seeds", type=int, nargs="+", choices=range(10), default=None)
    parser.add_argument(
        "--all",
        dest="run_all",
        action="store_true",
        help=(
            "validate all tasks for Zero-shot, or run all 1/5/10-shot "
            "seeds for visual prototypes or fixed fusion"
        ),
    )
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--limit-query",
        type=_positive_int,
        default=None,
        help=(
            "debug only: deterministic class-balanced total query limit; "
            "results are marked non-paper and never aggregated"
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _compact_result(result: dict) -> dict:
    return {
        "shot": result["shot"],
        "seed": result["seed"],
        "debug_run": result["debug_run"],
        "query_count": result["query_count"],
        "accuracy": result["metrics"]["accuracy"],
        "macro_f1": result["metrics"]["macro_f1"],
        "balanced_accuracy": result["metrics"]["balanced_accuracy"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    output_root = args.output_root or default_output_root(args.mode)
    claim_output_root(output_root, args.mode)
    formal_seeds = tuple(args.seeds) if args.seeds is not None else ((0, 1, 2, 3, 4) if args.run_all else (args.seed,))
    discover_kwargs = {}
    if args.seeds is not None or args.seed is not None and args.seed >= 5:
        discover_kwargs["formal_seeds"] = formal_seeds
    if args.image_root is not None:
        discover_kwargs["image_root"] = args.image_root
    tasks = discover_tasks(
        args.data_root,
        shot=args.shot,
        seed=args.seed,
        run_all=args.run_all,
        **discover_kwargs,
    )
    encoder = FrozenClipEncoder.create(device)
    if args.mode == "zero_shot":
        result = run_zero_shot_tasks(
            tasks,
            encoder=encoder,
            output_root=output_root,
            batch_size=args.batch_size,
            limit_query=args.limit_query,
            formal=args.run_all,
            formal_seeds=formal_seeds,
            show_progress=not args.no_progress,
        )
        compact = [_compact_result(result)]
    elif args.mode == "visual_prototype":
        results = run_visual_prototype_tasks(
            tasks,
            encoder=encoder,
            output_root=output_root,
            batch_size=args.batch_size,
            limit_query=args.limit_query,
            aggregate=args.run_all,
            expected_seeds=formal_seeds,
            show_progress=not args.no_progress,
        )
        compact = [_compact_result(result) for result in results]
    elif args.mode == "fixed_fusion":
        results = run_fixed_fusion_tasks(
            tasks,
            encoder=encoder,
            output_root=output_root,
            batch_size=args.batch_size,
            limit_query=args.limit_query,
            aggregate=args.run_all,
            expected_seeds=formal_seeds,
            show_progress=not args.no_progress,
        )
        compact = [_compact_result(result) for result in results]
    elif args.mode == "domain_geometry_fusion":
        results = run_domain_geometry_fusion_tasks(
            tasks,
            encoder=encoder,
            output_root=output_root,
            batch_size=args.batch_size,
            limit_query=args.limit_query,
            aggregate=args.run_all,
            expected_seeds=formal_seeds,
            show_progress=not args.no_progress,
        )
        compact = [_compact_result(result) for result in results]
    elif args.mode == "llm_structured_fixed_fusion":
        results = run_llm_structured_fixed_fusion_tasks(
            tasks,
            encoder=encoder,
            output_root=output_root,
            batch_size=args.batch_size,
            limit_query=args.limit_query,
            aggregate=args.run_all,
            expected_seeds=formal_seeds,
            show_progress=not args.no_progress,
        )
        compact = [_compact_result(result) for result in results]
    else:
        results = run_llm_reliability_fusion_tasks(
            tasks,
            encoder=encoder,
            output_root=output_root,
            batch_size=args.batch_size,
            limit_query=args.limit_query,
            aggregate=args.run_all,
            expected_seeds=formal_seeds,
            show_progress=not args.no_progress,
        )
        compact = [_compact_result(result) for result in results]

    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"Results written to: {output_root.resolve()}")
    if args.limit_query is not None:
        print("WARNING: debug_run=true; this result must not be used in the paper table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
