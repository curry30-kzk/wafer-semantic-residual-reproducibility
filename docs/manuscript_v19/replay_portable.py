"""Replay the three saved-count evidence bundles without raw images or models."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def frozen_count_input(source, expected_sha256, destination):
    """Copy an input only if its bytes, or LF/CRLF normalization, match its freeze hash."""
    payload = source.read_bytes()
    candidates = (payload, payload.replace(b'\r\n', b'\n'),
                  payload.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
    for candidate in candidates:
        if hashlib.sha256(candidate).hexdigest() == expected_sha256:
            destination.write_bytes(candidate)
            return destination
    raise ValueError(f'Frozen SHA-256 mismatch after LF/CRLF checks: {source.name}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path,
                   default=Path('recomputed') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args = p.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Git can normalize CRLF JSON to LF. Restore exact frozen bytes in generated
    # copies only; never edit the scientific source inputs or bypass their checks.
    original_folder = ROOT / 'docs/paper/revision_v6_20261015/reproduction'
    inventory = json.loads((original_folder / 'manifest.json').read_text(encoding='utf-8'))
    input_copies = output / 'frozen_input_copies'
    input_copies.mkdir()
    copied = {}
    for name in ('metrics_inputs.json', 'reference_checks.json'):
        copied[name] = frozen_count_input(
            original_folder / name, inventory['bundle_files'][name]['sha256'],
            input_copies / name)
    commands = [
        ('original', ['docs/paper/revision_v6_20261015/reproduction/recompute.py',
                      '--inputs', str(copied['metrics_inputs.json']),
                      '--references', str(copied['reference_checks.json']),
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
              'input_line_ending_policy': 'Generated input copies match the original frozen SHA-256; only LF/CRLF candidates are accepted. Source files are unchanged.',
              'scope': 'Saved confusion-count arithmetic and selection replay only; no encoder or model rerun, no new independent test.'}
    (output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('All three saved-count replays passed.')


if __name__ == '__main__':
    main()
