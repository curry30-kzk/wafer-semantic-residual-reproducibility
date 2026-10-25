"""Beginner-facing command line entry point for WM-811K preparation."""

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

from wm811k.preparation import PreparationConfig, run_preparation  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "WM811K" / "LSWMD.pkl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "WM811K"


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"LSWMD.pkl 路径（默认: {DEFAULT_INPUT}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"处理结果目录（默认: {DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--image-size", type=int, default=224, help="输出 PNG 边长（默认: 224）"
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="要生成的 K-shot 列表（默认: 1 5 10）",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="随机种子列表；prepare 默认使用论文种子 0 1 2 3 4",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="重新生成已存在且有效的 PNG",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "WM-811K 晶圆缺陷小样本分类数据准备工具。"
            "第一次运行请先选择 inspect。"
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="检查数据结构、统计标签并导出少量示例图"
    )
    _add_common_arguments(inspect_parser)
    inspect_parser.add_argument(
        "--samples-per-class",
        type=int,
        default=5,
        help="每类导出的检查图数量（默认: 5）",
    )

    prepare_parser = subparsers.add_parser(
        "prepare", help="导出八类缺陷图像并生成论文小样本划分"
    )
    _add_common_arguments(prepare_parser)
    prepare_parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help=(
            "仅限调试：每类最多导出多少张 PNG。"
            "不会缩小 JSON 划分，正式论文运行不得使用。"
        ),
    )
    prepare_parser.add_argument(
        "--no-progress", action="store_true", help="关闭完整导出的进度条"
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PreparationConfig:
    if args.seeds is None:
        seeds = (0,) if args.mode == "inspect" else (0, 1, 2, 3, 4)
    else:
        seeds = tuple(args.seeds)
    return PreparationConfig(
        mode=args.mode,
        input_path=args.input,
        output_root=args.output,
        image_size=args.image_size,
        shots=tuple(args.shots),
        seeds=seeds,
        samples_per_class=getattr(args, "samples_per_class", 5),
        limit_per_class=getattr(args, "limit_per_class", None),
        overwrite_images=args.overwrite_images,
        show_progress=not getattr(args, "no_progress", False),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    try:
        report = run_preparation(config)
    except (FileNotFoundError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f"\n处理失败：{exc}\n")

    print("\n处理完成，运行摘要：")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if config.mode == "inspect":
        print("\n请先检查 samples 和 summaries，确认无误后再运行 prepare。")
    elif not report["image_export_complete"]:
        print("\n注意：本次使用了 --limit-per-class，只能作为调试结果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
