"""Run paper-grade mainstream few-shot CLIP adaptation baselines."""

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
from wm811k.clip_adaptation_experiment import (
    ADAPTATION_MODES,
    run_clip_adaptation_tasks,
)
from wm811k.clip_features import FrozenClipEncoder
from wm811k.experiment_reporting import claim_output_root
from wm811k.resnet18_features import resolve_device


class AdaptationArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args=args, namespace=namespace)
        if parsed.run_all and (parsed.shot is not None or parsed.seed is not None):
            self.error("--all cannot be combined with --shot or --seed")
        if parsed.run_all and parsed.limit_query is not None:
            self.error("--limit-query is debug-only and cannot be combined with --all")
        if parsed.mode == "all" and parsed.output_root is not None:
            self.error("--output-root requires one explicit mode, not --mode all")
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
    directories = {
        "linear_probe": "clip_linear_probe",
        "tip_adapter": "clip_tip_adapter",
        "tip_adapter_f": "clip_tip_adapter_f",
        "lp_plus_plus": "clip_lp_plus_plus",
        "lp_plus_plus_fair": "clip_lp_plus_plus_fair",
        "zs_lp": "clip_zs_lp",
        "clap": "clip_clap",
        "rla": "clip_rla",
    }
    try:
        directory = directories[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported adaptation mode: {mode}") from exc
    return PROJECT_ROOT / "results" / directory


def build_parser() -> argparse.ArgumentParser:
    parser = AdaptationArgumentParser(
        description=(
            "Linear Probe, Tip-Adapter, Tip-Adapter-F, LP++, ZS-LP, CLAP, and RLA on the frozen "
            "WM-811K 8-way few-shot protocol"
        )
    )
    parser.add_argument(
        "--mode",
        choices=(*ADAPTATION_MODES, "all"),
        required=True,
        help="one adaptation baseline or all methods with shared CLIP encoding",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "WM811K",
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
        help="custom result root for one explicit mode",
    )
    parser.add_argument("--shot", type=int, choices=(1, 5, 10), default=None)
    parser.add_argument("--seed", type=int, choices=range(10), default=None)
    parser.add_argument("--seeds", type=int, nargs="+", choices=range(10), default=None)
    parser.add_argument(
        "--all",
        dest="run_all",
        action="store_true",
        help="run and aggregate the formal 1/5/10-shot x seeds 0-4 grid",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--limit-query",
        type=_positive_int,
        default=None,
        help="debug only: deterministic class-balanced total query limit",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _compact_result(mode: str, result: dict) -> dict:
    selection = result.get("selection") or {}
    compact_selection_keys = (
        "strategy",
        "reason",
        "C",
        "epoch",
        "beta",
        "alpha",
        "validation_macro_f1",
    )
    return {
        "mode": mode,
        "shot": result["shot"],
        "seed": result["seed"],
        "debug_run": result["debug_run"],
        "query_count": result["query_count"],
        "selected": {
            key: selection[key]
            for key in compact_selection_keys
            if key in selection
        },
        "accuracy": result["metrics"]["accuracy"],
        "macro_f1": result["metrics"]["macro_f1"],
        "balanced_accuracy": result["metrics"]["balanced_accuracy"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    modes = ADAPTATION_MODES if args.mode == "all" else (args.mode,)
    output_roots = {
        mode: (
            args.output_root
            if args.output_root is not None
            else default_output_root(mode)
        )
        for mode in modes
    }
    for mode, output_root in output_roots.items():
        claim_output_root(output_root, mode)
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
    device = resolve_device(args.device)
    encoder = FrozenClipEncoder.create(device)
    results = run_clip_adaptation_tasks(
        tasks,
        encoder=encoder,
        modes=modes,
        output_roots=output_roots,
        batch_size=args.batch_size,
        support_augment_views=(
            1
            if len(modes) == 1
            and modes[0] in ("lp_plus_plus", "lp_plus_plus_fair")
            else 10
        ),
        limit_query=args.limit_query,
        aggregate=args.run_all,
        expected_seeds=formal_seeds,
        show_progress=not args.no_progress,
    )
    compact = [
        _compact_result(mode, result)
        for mode in modes
        for result in results[mode]
    ]
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    for mode, output_root in output_roots.items():
        print(f"{mode} results written to: {Path(output_root).resolve()}")
    if args.limit_query is not None:
        print("WARNING: debug_run=true; this result must not be used in the paper table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
