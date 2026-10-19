"""Verify count-only reproduction after copying four files outside the repository."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from recompute import read_json, sha256


def main():
    folder = Path(__file__).resolve().parent
    names = ("recompute.py", "metrics_inputs.json", "manifest.json", "reference_checks.json")
    with tempfile.TemporaryDirectory(prefix="v4_count_reproduction_") as temporary:
        independent = Path(temporary)
        for name in names:
            shutil.copyfile(folder / name, independent / name)
        process = subprocess.run([sys.executable, "-X", "utf8", str(independent / "recompute.py")],
                                 cwd=independent, capture_output=True, text=True, encoding="utf-8", timeout=60)
        if process.returncode:
            raise RuntimeError(f"Standalone copy failed: {process.stdout}\n{process.stderr}")
        verification = read_json(independent / "recomputed/verification.json")
        if not verification["all_passed"] or verification["source_audit"]["status"] != "not_requested":
            raise ValueError("Standalone copy did not pass all checks without original source access.")
        statistics = read_json(independent / "recomputed/statistics.json")
        evidence = {"status": "passed", "copied_files": list(names),
                    "copied_file_sha256": {name: sha256(independent / name) for name in names},
                    "original_repository_access_required": False, "source_audit_requested": False,
                    "all_reference_checks_passed": verification["all_passed"],
                    **{name: verification[name] for name in
                       ("checks_count", "input_record_count", "summary_cell_count", "input_sha256")},
                    "recompute_sha256": sha256(folder / "recompute.py"),
                    "shared_text_control_summary_cells": sum(row["scope"] == "shared_text_control" for row in statistics["summary"]),
                    "shared_text_control_paired_comparisons": sum(row["scope"] == "shared_text_control" for row in statistics["paired_comparisons"]),
                    "runtime": {"numpy": verification["numpy_version"], "scipy": verification["scipy_version"]}}
    (folder / "portability_check.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
