"""Replay the three saved-count evidence bundles without raw images or models."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path,
                   default=Path('recomputed') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args = p.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commands = [
        ('original', ['docs/paper/revision_v6_20261015/reproduction/recompute.py',
                      '--output-dir', str(output / 'original')]),
        ('clap', ['docs/paper/revision_v9_20261015/bridge/recompute_bridge.py',
                  '--root', 'results/teacher_review_v9_20261015/clap_bridge',
                  '--output-dir', str(output / 'clap')]),
        ('global', ['docs/paper/revision_v11_20261017/global_control/recompute_control.py',
                    '--root', 'results/teacher_review_v11_20261017/global_control']),
    ]
    audits = {}
    for label, command in commands:
        proc = subprocess.run([sys.executable, '-X', 'utf8', *command], cwd=ROOT,
                              capture_output=True, text=True, encoding='utf-8')
        if proc.returncode:
            (output / f'{label}_failure.log').write_text(proc.stdout + proc.stderr, encoding='utf-8')
            raise SystemExit(f'{label} replay failed; see its output log')
        audits[label] = json.loads(proc.stdout)
        if not audits[label]['all_passed']:
            raise SystemExit(f'{label} replay reported a failed check')
        (output / f'{label}_audit.json').write_text(json.dumps(audits[label], indent=2) + '\n', encoding='utf-8')
        print(f'{label}: passed', flush=True)
    report = {'all_passed': True, 'audits': audits,
              'scope': 'Saved confusion-count arithmetic and selection replay only; no encoder or model rerun, no new independent test.'}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('All three saved-count replays passed.')


if __name__ == '__main__':
    main()
