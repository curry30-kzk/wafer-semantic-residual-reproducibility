"""Verify and export the frozen semantic-source matching experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wm811k.semantic_source_matching import (
    build_report,
    export_compact_artifacts,
    load_and_validate_frozen_config,
    write_report,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "semantic_source_matching_frozen_v1.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="冻结语义来源匹配实验的校验、汇总与紧凑证据导出"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("verify", "校验冻结语义包和任务清单哈希"),
        ("analyze", "从现有缓存重新生成完整本地分析"),
        ("export", "导出可提交版本控制的紧凑证据"),
        ("all", "依次校验、分析并导出"),
    ):
        subparsers.add_parser(command, help=help_text)
    return parser


def _paths(config: dict) -> tuple[Path, Path]:
    paths = config.get("paths", {})
    result_root = PROJECT_ROOT / paths.get(
        "result_root", "results/semantic_source_matching"
    )
    artifact_root = PROJECT_ROOT / paths.get(
        "artifact_root", "docs/experiments/artifacts/semantic_source_matching"
    )
    return result_root, artifact_root


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_and_validate_frozen_config(args.config, project_root=PROJECT_ROOT)
    result_root, artifact_root = _paths(config)
    report_path = result_root / "report.json"

    if args.command in {"analyze", "all"}:
        report = build_report(project_root=PROJECT_ROOT)
        report["frozen_config"] = {
            "schema_version": config["schema_version"],
            "common_config": config["common_config"],
            "semantic_bundles": config["validation"]["semantic_bundles"],
            "manifest_sets": config["validation"]["manifest_sets"],
        }
        write_report(report, result_root)
    if args.command in {"export", "all"}:
        if not report_path.is_file():
            raise FileNotFoundError(f"analysis report is missing: {report_path}")
        exported = export_compact_artifacts(report_path, artifact_root)
    else:
        exported = []

    audits = []
    if report_path.is_file() and args.command != "verify":
        report = json.loads(report_path.read_text(encoding="utf-8"))
        audits = [
            item
            for values in report.get("visual_cache_audits", {}).values()
            for item in values
        ]
    print(
        json.dumps(
            {
                "command": args.command,
                "config_valid": True,
                "cache_audits": len(audits),
                "all_cache_pairs_passed": all(item.get("passed") for item in audits),
                "exported_files": [path.as_posix() for path in exported],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
