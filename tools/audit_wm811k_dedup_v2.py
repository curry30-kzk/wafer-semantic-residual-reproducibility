"""Formal full-dataset audit entry point for WM-811K deduplication v2."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prepare_wm811k_v2 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["audit", *sys.argv[1:]]))
