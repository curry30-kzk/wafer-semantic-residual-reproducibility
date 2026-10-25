"""Prepare and verify the isolated WM-811K content-deduplicated v2 protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wm811k.dedup_v2 import (  # noqa: E402
    AUDIT_FILENAMES,
    attach_content_hashes,
    clean_training_candidates,
    sha256_file,
    write_audit_bundle,
)
from wm811k.labels import prepare_labeled_records  # noqa: E402
from wm811k.pickle_compat import load_wm811k_pickle  # noqa: E402
from wm811k.splits_v2 import (  # noqa: E402
    PROTOCOL_NAME,
    build_v2_manifest_grid,
    manifest_set_sha256,
    validate_manifest_grid,
    write_v2_manifest,
)


DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "WM811K" / "LSWMD.pkl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "WM811K_dedup_v2"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "data" / "processed" / "WM811K" / "images"
DEFAULT_AUDIT_SCRIPT = PROJECT_ROOT / "tools" / "audit_wm811k_dedup_v2.py"
FORMAL_SHOTS = (1, 5, 10)
FORMAL_SEEDS = tuple(range(10))
WM811K_EXPECTED_COUNTS = {
    "valid_records": 25519,
    "unique_contents": 25425,
    "duplicate_content_groups": 93,
    "cross_split_duplicate_groups": 92,
    "cross_split_training_records": 93,
    "cross_split_test_records": 92,
    "label_conflict_groups": 3,
    "training_internal_duplicate_groups": 2,
    "test_internal_duplicate_groups": 0,
    "fixed_test_records": 7894,
}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _assert_expected_counts(
    actual: Mapping[str, Any], expected: Mapping[str, int] | None
) -> None:
    if expected is None:
        return
    mismatches = {
        key: {"expected": int(value), "actual": int(actual.get(key, -1))}
        for key, value in expected.items()
        if int(actual.get(key, -1)) != int(value)
    }
    if mismatches:
        raise ValueError(f"full-audit counts do not match the frozen design: {mismatches}")


def audit_v2(
    *,
    input_path: str | Path,
    output_root: str | Path,
    frame: pd.DataFrame | None = None,
    audit_script_path: str | Path = DEFAULT_AUDIT_SCRIPT,
    git_commit: str | None = None,
    expected_counts: Mapping[str, int] | None = WM811K_EXPECTED_COUNTS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run the full raw-matrix audit and return the hashed records."""

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"WM-811K source pickle does not exist: {source}")
    raw_frame = frame if frame is not None else load_wm811k_pickle(source)
    records = attach_content_hashes(prepare_labeled_records(raw_frame))
    report = write_audit_bundle(
        records,
        Path(output_root) / "audit",
        source_path=source,
        script_path=audit_script_path,
        git_commit=git_commit or _git_commit(),
    )
    _assert_expected_counts(report["counts"], expected_counts)
    return report, records


def prepare_v2(
    *,
    input_path: str | Path,
    output_root: str | Path,
    image_root: str | Path,
    shots: Sequence[int] = FORMAL_SHOTS,
    seeds: Sequence[int] = FORMAL_SEEDS,
    image_size: int = 224,
    frame: pd.DataFrame | None = None,
    audit_script_path: str | Path = DEFAULT_AUDIT_SCRIPT,
    git_commit: str | None = None,
    expected_counts: Mapping[str, int] | None = WM811K_EXPECTED_COUNTS,
) -> dict[str, Any]:
    """Audit, clean, and freeze v2 manifests without copying image files."""

    output = Path(output_root)
    audit_report, records = audit_v2(
        input_path=input_path,
        output_root=output,
        frame=frame,
        audit_script_path=audit_script_path,
        git_commit=git_commit,
        expected_counts=expected_counts,
    )
    audit_report_path = output / "audit" / "audit_report.json"
    audit_hash = sha256_file(audit_report_path)
    pool_hash = str(audit_report["csv_sha256"]["clean_training_pool.csv"])
    cleaned = clean_training_candidates(records)
    active_records = pd.concat(
        [cleaned.training, cleaned.query], ignore_index=True, sort=False
    )
    manifests = build_v2_manifest_grid(
        active_records,
        shots=shots,
        seeds=seeds,
        image_size=int(image_size),
        audit_hash=audit_hash,
        pool_hash=pool_hash,
        source_metadata=audit_report["source"],
    )
    splits_root = output / "splits"
    expected_names: set[str] = set()
    for (shot, seed), manifest in sorted(manifests.items()):
        name = f"8way_{shot}shot_seed{seed}.json"
        expected_names.add(name)
        write_v2_manifest(manifest, splits_root / name)
    unexpected = [
        path for path in splits_root.glob("8way_*shot_seed*.json")
        if path.name not in expected_names
    ]
    if unexpected:
        raise ValueError(f"v2 splits directory contains stale manifests: {unexpected[0]}")

    report: dict[str, Any] = {
        "schema_version": "wm811k_dedup_v2_preparation_v1",
        "protocol": PROTOCOL_NAME,
        "input_path": str(Path(input_path).resolve()),
        "output_root": str(output.resolve()),
        "image_root": str(Path(image_root).resolve()),
        "shots": [int(value) for value in shots],
        "seeds": [int(value) for value in seeds],
        "manifest_count": int(len(manifests)),
        "manifest_set_sha256": manifest_set_sha256(manifests),
        "dedup_audit_sha256": audit_hash,
        "candidate_pool_sha256": pool_hash,
        "audit_counts": dict(audit_report["counts"]),
        "class_counts_after": dict(audit_report["class_counts_after"]),
        "images_copied": False,
    }
    _atomic_write_json(output / "preparation_report.json", report)
    return report


def verify_v2(
    *,
    output_root: str | Path,
    image_root: str | Path,
    shots: Sequence[int] = FORMAL_SHOTS,
    seeds: Sequence[int] = FORMAL_SEEDS,
    verify_source: bool = True,
    verify_images: bool = True,
) -> dict[str, Any]:
    """Read back every audit and manifest artifact without modifying it."""

    output = Path(output_root)
    audit_root = output / "audit"
    audit_report_path = audit_root / "audit_report.json"
    preparation_path = output / "preparation_report.json"
    audit_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    for name in AUDIT_FILENAMES:
        actual = sha256_file(audit_root / name)
        if actual != audit_report["csv_sha256"][name]:
            raise ValueError(f"audit CSV hash mismatch: {name}")
    source = Path(audit_report["source"]["path"])
    if verify_source:
        if not source.is_file() or sha256_file(source) != audit_report["source"]["sha256"]:
            raise ValueError("source pickle identity does not match the audit")

    manifests: dict[tuple[int, int], dict[str, Any]] = {}
    for shot in shots:
        for seed in seeds:
            path = output / "splits" / f"8way_{int(shot)}shot_seed{int(seed)}.json"
            manifests[(int(shot), int(seed))] = json.loads(
                path.read_text(encoding="utf-8")
            )
    validate_manifest_grid(manifests, shots=shots, seeds=seeds)
    if manifest_set_sha256(manifests) != preparation["manifest_set_sha256"]:
        raise ValueError("manifest set hash mismatch")
    audit_hash = sha256_file(audit_report_path)
    if audit_hash != preparation["dedup_audit_sha256"]:
        raise ValueError("dedup audit hash mismatch")
    if audit_report["csv_sha256"]["clean_training_pool.csv"] != preparation["candidate_pool_sha256"]:
        raise ValueError("candidate pool hash mismatch")

    missing_images: list[str] = []
    if verify_images:
        seen: set[str] = set()
        for manifest in manifests.values():
            for record in (*manifest["support"], *manifest["query"]):
                relative = str(record["image_path"])
                if relative in seen:
                    continue
                seen.add(relative)
                if not (Path(image_root) / Path(relative)).is_file():
                    missing_images.append(relative)
        if missing_images:
            raise FileNotFoundError(f"v2 image is missing: {missing_images[0]}")
    return {
        "protocol": PROTOCOL_NAME,
        "audit_verified": True,
        "manifest_count": len(manifests),
        "manifest_set_sha256": preparation["manifest_set_sha256"],
        "missing_image_count": len(missing_images),
    }


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WM-811K 内容去重协议 v2")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="全量审计原始八类记录")
    _add_paths(audit)
    prepare = commands.add_parser("prepare", help="审计并生成冻结 v2 清单")
    _add_paths(prepare)
    prepare.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    prepare.add_argument("--image-size", type=int, default=224)
    prepare.add_argument("--shots", type=int, nargs="+", default=list(FORMAL_SHOTS))
    prepare.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    verify = commands.add_parser("verify", help="只读验证全部 v2 证据与清单")
    verify.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    verify.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    verify.add_argument("--shots", type=int, nargs="+", default=list(FORMAL_SHOTS))
    verify.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            report, _ = audit_v2(input_path=args.input, output_root=args.output)
        elif args.command == "prepare":
            report = prepare_v2(
                input_path=args.input,
                output_root=args.output,
                image_root=args.image_root,
                shots=args.shots,
                seeds=args.seeds,
                image_size=args.image_size,
            )
        elif args.command == "verify":
            report = verify_v2(
                output_root=args.output,
                image_root=args.image_root,
                shots=args.shots,
                seeds=args.seeds,
            )
        else:
            raise AssertionError(f"unsupported command: {args.command}")
    except (FileNotFoundError, KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, f"\nWM-811K v2 处理失败：{exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
