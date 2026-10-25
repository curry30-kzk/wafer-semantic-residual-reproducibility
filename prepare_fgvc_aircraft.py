"""Download, inspect, and prepare the official FGVC-Aircraft dataset."""

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

from fgvc_aircraft.preparation import (
    OFFICIAL_ARCHIVE_MD5,
    OFFICIAL_ARCHIVE_NAME,
    OFFICIAL_ARCHIVE_URL,
    download_archive,
    inspect_aircraft_dataset,
    prepare_aircraft_dataset,
    safe_extract_archive,
)


DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "FGVC-Aircraft"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "FGVC-Aircraft"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="下载、校验并准备 FGVC-Aircraft 100 类小样本协议"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    download = subparsers.add_parser("download", help="下载并校验官方 2.75 GB 压缩包")
    download.add_argument("--url", default=OFFICIAL_ARCHIVE_URL)
    download.add_argument(
        "--archive-path",
        type=Path,
        default=DEFAULT_RAW_ROOT / OFFICIAL_ARCHIVE_NAME,
    )
    download.add_argument("--expected-md5", default=OFFICIAL_ARCHIVE_MD5)
    download.add_argument("--no-progress", action="store_true")

    for name, help_text in (
        ("inspect", "检查已经解压的官方数据"),
        ("prepare", "生成验证集和测试集小样本任务清单"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
        command.add_argument(
            "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
        )
    return parser


def _official_data_root(raw_root: Path) -> Path:
    return raw_root / "fgvc-aircraft-2013b" / "data"


def _ensure_extracted(raw_root: Path) -> Path:
    data_root = _official_data_root(raw_root)
    if data_root.is_dir():
        return data_root
    archive = raw_root / OFFICIAL_ARCHIVE_NAME
    if not archive.is_file():
        raise FileNotFoundError(
            f"未找到解压数据或官方压缩包；请先运行 download：{archive}"
        )
    safe_extract_archive(archive, raw_root)
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"解压完成后仍未找到官方 data 目录：{data_root}"
        )
    return data_root


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "download":
        archive = download_archive(
            args.url,
            args.archive_path,
            expected_md5=args.expected_md5,
            show_progress=not args.no_progress,
        )
        result = {
            "mode": "download",
            "archive_path": str(archive.resolve()),
            "md5": args.expected_md5,
        }
    else:
        data_root = _ensure_extracted(args.raw_root)
        if args.mode == "inspect":
            result = inspect_aircraft_dataset(data_root)
            result = {**result, "mode": "inspect"}
        else:
            result = prepare_aircraft_dataset(data_root, args.output_root)
            result = {**result, "mode": "prepare"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
