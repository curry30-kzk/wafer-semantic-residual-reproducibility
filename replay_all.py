"""Replay saved counts through the exact-freeze portable wrapper.

Original scientific inputs and manifests are unchanged. Only generated copies
may restore original LF/CRLF bytes, and every restored copy must match its
historical freeze SHA-256.
"""
from pathlib import Path
import runpy


def main():
    runpy.run_path(
        str(Path(__file__).resolve().parent / 'docs/manuscript_v19/replay_portable.py'),
        run_name='__main__',
    )


if __name__ == '__main__':
    main()
