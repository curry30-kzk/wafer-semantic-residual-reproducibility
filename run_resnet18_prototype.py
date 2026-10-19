"""Run the frozen ResNet18 visual-prototype WM-811K baseline."""

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

from wm811k.baseline_experiment import discover_tasks, run_tasks
from wm811k.resnet18_features import resolve_device


class BaselineArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args=args, namespace=namespace)
        if parsed.run_all and (parsed.shot is not None or parsed.seed is not None):
            self.error("--all cannot be combined with --shot or --seed")
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


def build_parser() -> argparse.ArgumentParser:
    parser = BaselineArgumentParser(
        description=(
            "Frozen ImageNet ResNet18 + cosine visual prototypes for the "
            "WM-811K 8-way few-shot protocol"
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "WM811K",
        help="prepared WM-811K directory containing images/ and splits/",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "resnet18_prototype",
        help="generated result directory",
    )
    parser.add_argument("--shot", type=int, choices=(1, 5, 10), default=None)
    parser.add_argument("--seed", type=int, choices=range(5), default=None)
    parser.add_argument(
        "--all",
        dest="run_all",
        action="store_true",
        help="run all 1/5/10-shot tasks for seeds 0-4 and aggregate them",
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_all and args.limit_query is not None:
        raise SystemExit("--limit-query is debug-only and cannot be combined with --all")
    device = resolve_device(args.device)
    tasks = discover_tasks(
        args.data_root,
        shot=args.shot,
        seed=args.seed,
        run_all=args.run_all,
    )
    results = run_tasks(
        tasks,
        output_root=args.output_root,
        batch_size=args.batch_size,
        device=device,
        limit_query=args.limit_query,
        aggregate=args.run_all,
        show_progress=not args.no_progress,
    )
    compact = [
        {
            "shot": result["shot"],
            "seed": result["seed"],
            "debug_run": result["debug_run"],
            "query_count": result["query_count"],
            "accuracy": result["metrics"]["accuracy"],
            "macro_f1": result["metrics"]["macro_f1"],
            "balanced_accuracy": result["metrics"]["balanced_accuracy"],
        }
        for result in results
    ]
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"Results written to: {args.output_root.resolve()}")
    if args.limit_query is not None:
        print("WARNING: debug_run=true; this result must not be used in the paper table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
