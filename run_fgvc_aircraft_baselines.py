"""Run paper-auditable few-shot baselines on FGVC-Aircraft."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wm811k.baseline_experiment import discover_tasks, run_tasks
from wm811k.clip_adaptation_experiment import (
    ADAPTATION_MODES,
    run_clip_adaptation_tasks,
)
from wm811k.clip_experiment import (
    run_visual_prototype_tasks,
    run_zero_shot_tasks,
)
from wm811k.clip_features import FrozenClipEncoder
from wm811k.experiment_reporting import claim_output_root
from wm811k.resnet18_features import (
    AIRCRAFT_PREPROCESS_IDENTIFIER,
    extract_features,
    resnet18_preprocess_metadata,
    resolve_device,
)


DEFAULT_DATA_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "FGVC-Aircraft"
)
DEFAULT_IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "FGVC-Aircraft"
    / "fgvc-aircraft-2013b"
    / "data"
)
SINGLE_MODES = (
    "resnet18_prototype",
    "zero_shot",
    "visual_prototype",
    *ADAPTATION_MODES,
)
ALL_VISUAL_MODES = (
    "resnet18_prototype",
    "visual_prototype",
    "linear_probe",
    "tip_adapter",
    "tip_adapter_f",
    "lp_plus_plus",
    "lp_plus_plus_fair",
    "zs_lp",
    "clap",
    "rla",
)
CLAP_FROZEN_MODES = ("zs_lp", "clap")
CLAP_FROZEN_SOURCE_PATHS = (
    "src/wm811k/clap.py",
    "src/wm811k/clip_adaptation_experiment.py",
    "run_fgvc_aircraft_baselines.py",
)
RLA_FROZEN_SOURCE_PATHS = (
    "src/wm811k/rla.py",
    "src/wm811k/clip_adaptation_experiment.py",
    "run_fgvc_aircraft_baselines.py",
)


class AircraftArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args=args, namespace=namespace)
        if parsed.run_all and (parsed.shot is not None or parsed.seed is not None):
            self.error("--all cannot be combined with --shot or --seed")
        if parsed.seeds is not None and len(set(parsed.seeds)) != len(parsed.seeds):
            self.error("--seeds must contain unique values")
        if parsed.seeds is not None and not parsed.run_all:
            self.error("--seeds requires --all")
        if parsed.run_all and parsed.limit_query is not None:
            self.error("--limit-query is debug-only and cannot be combined with --all")
        if parsed.mode == "all_visual" and parsed.output_root is not None:
            self.error("--output-root requires one explicit mode")
        if parsed.query_split == "test" and parsed.run_all:
            if parsed.frozen_config is None or not parsed.confirm_test_once:
                self.error(
                    "formal Test evaluation requires --frozen-config and "
                    "--confirm-test-once"
                )
            if not parsed.frozen_config.is_file():
                self.error(f"frozen config does not exist: {parsed.frozen_config}")
        if parsed.shot is None:
            parsed.shot = 5
        if parsed.seed is None:
            parsed.seed = 0
        parsed.formal_seeds = (
            tuple(parsed.seeds) if parsed.seeds is not None else tuple(range(5))
        )
        return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = AircraftArgumentParser(
        description="FGVC-Aircraft 100 类小样本正式基线"
    )
    parser.add_argument(
        "--mode",
        choices=(*SINGLE_MODES, "all_visual"),
        required=True,
        help="运行一个基线，或运行全部支持图像视觉基线",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument(
        "--query-split", choices=("validation", "test"), default="validation"
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--shot", type=int, choices=(1, 5, 10), default=None)
    parser.add_argument("--seed", type=int, choices=range(10), default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        choices=range(10),
        nargs="+",
        default=None,
        help="正式运行的随机种子列表；默认兼容历史 seed 0-4",
    )
    parser.add_argument(
        "--all",
        dest="run_all",
        action="store_true",
        help="运行 1/5/10-shot × 指定随机种子的正式网格",
    )
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--limit-query", type=_positive_int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--frozen-config", type=Path, default=None)
    parser.add_argument("--confirm-test-once", action="store_true")
    return parser


def default_output_root(query_split: str, mode: str) -> Path:
    directories = {
        "resnet18_prototype": "resnet18_prototype",
        "zero_shot": "clip_zero_shot",
        "visual_prototype": "clip_visual_prototype",
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
        raise ValueError(f"unsupported aircraft baseline mode: {mode}") from exc
    return (
        PROJECT_ROOT
        / "results"
        / f"fgvc_aircraft_{query_split}"
        / directory
    )


def _validate_frozen_test_config(
    path: Path,
    *,
    expected_mode: str,
    expected_seeds: Sequence[int],
) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid frozen Test config: {path}") from exc
    if payload.get("query_split") != "test":
        raise ValueError("frozen Test config must declare query_split='test'")
    if payload.get("mode") != expected_mode:
        raise ValueError(
            "frozen Test config mode does not match the requested mode"
        )
    if tuple(payload.get("shots", ())) != (1, 5, 10):
        raise ValueError("frozen Test config must declare shots [1, 5, 10]")
    if tuple(payload.get("seeds", ())) != tuple(expected_seeds):
        raise ValueError(
            "frozen Test config seeds do not match the requested formal grid"
        )
    if payload.get("query_labels_used_for_selection") is not False:
        raise ValueError(
            "frozen Test config must prohibit query-label model selection"
        )
    if expected_mode in CLAP_FROZEN_MODES:
        required = {
            "schema_version": f"fgvc_aircraft_{expected_mode}_baseline_frozen_v1",
            "development_split": "validation",
            "support_augmentation_view_count": 10,
            "epochs": 300,
            "optimizer": "SGD",
            "learning_rate": 0.1,
            "batch_size": 256,
            "warmup_epochs": 1,
            "warmup_learning_rate": 1e-5,
            "learning_rate_scheduler": "cosine",
            "weight_decay": 0.0,
            "logit_scale": 100.0,
            "reference_commit": "dbdf4095b3d824546474c725539b9f4efff3ca0e",
            "test_run_before_freeze": False,
        }
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise ValueError(
                    f"frozen Test config has invalid {key!r}: "
                    f"expected {expected!r}"
                )
        hashes = payload.get("implementation_sha256")
        if not isinstance(hashes, dict):
            raise ValueError("frozen Test config must include implementation_sha256")
        for relative_path in CLAP_FROZEN_SOURCE_PATHS:
            source_path = PROJECT_ROOT / relative_path
            actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if hashes.get(relative_path) != actual:
                raise ValueError(
                    f"frozen Test config implementation hash mismatch: {relative_path}"
                )
    if expected_mode == "rla":
        required = {
            "schema_version": "fgvc_aircraft_rla_baseline_frozen_v1",
            "development_split": "validation",
            "support_augmentation_view_count": 10,
            "epochs": 300,
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "adam_eps": 1e-4,
            "learning_rate_scheduler": "cosine_annealing",
            "regularizer": "frobenius_distance_from_identity",
            "regularization_lambda_policy": "one_over_shot",
            "logit_scale": 100.0,
            "initialization": "identity",
            "reference_commit": "b43e795cbf0b53a9b0aa83cf4f47d9755896ba6f",
            "test_run_before_freeze": False,
        }
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise ValueError(
                    f"frozen Test config has invalid {key!r}: expected {expected!r}"
                )
        hashes = payload.get("implementation_sha256")
        if not isinstance(hashes, dict):
            raise ValueError("frozen Test config must include implementation_sha256")
        for relative_path in RLA_FROZEN_SOURCE_PATHS:
            source_path = PROJECT_ROOT / relative_path
            actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if hashes.get(relative_path) != actual:
                raise ValueError(
                    f"frozen Test config implementation hash mismatch: {relative_path}"
                )
    return payload


def _assert_fresh_formal_test_root(output_root: Path) -> None:
    """Refuse a second formal Test run or a root containing prior artifacts."""
    if not output_root.exists():
        return
    existing = [
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != ".experiment_method.json"
    ]
    if existing:
        raise FileExistsError(
            f"formal Test output root already contains result artifacts: {output_root}"
        )


def _compact(mode: str, result: dict) -> dict:
    selection = result.get("selection") or {}
    return {
        "mode": mode,
        "shot": result.get("shot"),
        "seed": result.get("seed"),
        "debug_run": result["debug_run"],
        "query_count": result["query_count"],
        "selected": {
            key: selection[key]
            for key in (
                "strategy",
                "reason",
                "C",
                "epoch",
                "beta",
                "alpha",
                "validation_macro_f1",
            )
            if key in selection
        },
        "accuracy": result["metrics"]["accuracy"],
        "macro_f1": result["metrics"]["macro_f1"],
        "balanced_accuracy": result["metrics"]["balanced_accuracy"],
    }


def _result_root(args, mode: str) -> Path:
    if args.output_root is not None:
        return args.output_root
    return default_output_root(args.query_split, mode)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.query_split == "test" and args.run_all:
        _validate_frozen_test_config(
            args.frozen_config,
            expected_mode=args.mode,
            expected_seeds=args.formal_seeds,
        )
    tasks = discover_tasks(
        args.data_root,
        shot=args.shot,
        seed=args.seed,
        run_all=args.run_all,
        query_split=args.query_split,
        n_way=100,
        image_root=args.image_root,
        formal_seeds=args.formal_seeds,
    )
    device = resolve_device(args.device)
    requested_modes = (
        ALL_VISUAL_MODES if args.mode == "all_visual" else (args.mode,)
    )
    compact: list[dict] = []

    if "resnet18_prototype" in requested_modes:
        output_root = _result_root(args, "resnet18_prototype")
        aircraft_extractor = partial(extract_features, dataset="fgvc_aircraft")
        results = run_tasks(
            tasks,
            output_root=output_root,
            batch_size=args.batch_size,
            device=device,
            limit_query=args.limit_query,
            aggregate=args.run_all,
            expected_seeds=args.formal_seeds,
            feature_extractor=aircraft_extractor,
            preprocess_identifier=AIRCRAFT_PREPROCESS_IDENTIFIER,
            preprocess_metadata=resnet18_preprocess_metadata("fgvc_aircraft"),
            show_progress=not args.no_progress,
        )
        compact.extend(_compact("resnet18_prototype", result) for result in results)

    clip_modes = tuple(
        mode for mode in requested_modes if mode != "resnet18_prototype"
    )
    if clip_modes:
        encoder = FrozenClipEncoder.create(device, dataset="fgvc_aircraft")
        if "zero_shot" in clip_modes:
            output_root = _result_root(args, "zero_shot")
            claim_output_root(output_root, "zero_shot")
            result = run_zero_shot_tasks(
                tasks,
                encoder=encoder,
                output_root=output_root,
                batch_size=args.batch_size,
                limit_query=args.limit_query,
                formal=args.run_all,
                formal_seeds=args.formal_seeds,
                show_progress=not args.no_progress,
            )
            compact.append(_compact("zero_shot", result))
        if "visual_prototype" in clip_modes:
            output_root = _result_root(args, "visual_prototype")
            claim_output_root(output_root, "visual_prototype")
            results = run_visual_prototype_tasks(
                tasks,
                encoder=encoder,
                output_root=output_root,
                batch_size=args.batch_size,
                limit_query=args.limit_query,
                aggregate=args.run_all,
                expected_seeds=args.formal_seeds,
                show_progress=not args.no_progress,
            )
            compact.extend(
                _compact("visual_prototype", result) for result in results
            )
        adaptation_modes = tuple(
            mode for mode in clip_modes if mode in ADAPTATION_MODES
        )
        if adaptation_modes:
            output_roots = {
                mode: _result_root(args, mode) for mode in adaptation_modes
            }
            for mode, output_root in output_roots.items():
                if args.query_split == "test" and args.run_all:
                    _assert_fresh_formal_test_root(output_root)
                claim_output_root(output_root, mode)
            results_by_mode = run_clip_adaptation_tasks(
                tasks,
                encoder=encoder,
                modes=adaptation_modes,
                output_roots=output_roots,
                batch_size=args.batch_size,
                limit_query=args.limit_query,
                aggregate=args.run_all,
                expected_seeds=args.formal_seeds,
                show_progress=not args.no_progress,
                support_augment_views=(
                    1
                    if len(adaptation_modes) == 1
                    and adaptation_modes[0] in ("lp_plus_plus", "lp_plus_plus_fair")
                    else 10
                ),
            )
            compact.extend(
                _compact(mode, result)
                for mode in adaptation_modes
                for result in results_by_mode[mode]
            )

    print(json.dumps(compact, ensure_ascii=False, indent=2))
    if args.limit_query is not None:
        print("警告：debug_run=true，该结果不能进入论文正式表格。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
