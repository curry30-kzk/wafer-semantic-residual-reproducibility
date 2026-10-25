"""Regression for the public root entrypoint in a clean LF checkout."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublicReplayEntrypoint(unittest.TestCase):
    def test_root_replay_restores_exact_frozen_input_copies(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'replayed'
            proc = subprocess.run(
                [sys.executable, '-X', 'utf8', 'replay_all.py', '--output-dir', str(output)],
                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            report = json.loads((output / 'verification.json').read_text(encoding='utf-8'))
            self.assertTrue(report['all_passed'])
            self.assertIn('Generated input copies match', report['input_line_ending_policy'])


if __name__ == '__main__':
    unittest.main()
