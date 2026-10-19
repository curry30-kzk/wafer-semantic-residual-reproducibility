"""Check release file hashes and detect common secret/private-path patterns.

This bounded, deterministic scan is not a guarantee that every possible secret
format is detectable. It does not access GitHub or any credential store.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
PATTERNS = {
    'personal_windows_path': re.compile(r'(?i)[A-Z]:(?:/|\\+)(?:Users|用户)(?:/|\\+)[^"\r\n,]+'),
    'windows_absolute_path': re.compile(r'(?<![A-Za-z])([A-Za-z]:(?:/|\\+)[^"\r\n,]+)'),
    'unix_personal_path': re.compile(r'/(?:home|Users)/[A-Za-z0-9_.-]+/'),
    'private_ipv4': re.compile(r'\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b'),
    'credential_value': re.compile(r'(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)'),
}


def public_files():
    for f in ROOT.rglob('*'):
        if not f.is_file():
            continue
        relative = f.relative_to(ROOT)
        if any(p in {'.git', '__pycache__', '.pytest_cache', '.venv', 'recomputed'} for p in relative.parts):
            continue
        if f.name in {'PUBLIC_MANIFEST.json', 'PUBLIC_AUDIT.json', 'prepare_public_export.py'}:
            continue
        yield f, relative.as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true', help='Create the released-file SHA inventory and audit report')
    args = parser.parse_args()
    rows, findings = [], []
    for f, name in public_files():
        payload = f.read_bytes()
        rows.append({'path': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
        if f.suffix.lower() in {'.py', '.json', '.md', '.txt', '.csv'} or f.name == '.gitignore':
            text = payload.decode('utf-8-sig')
            for kind, pattern in PATTERNS.items():
                for _ in pattern.finditer(text):
                    findings.append({'path': name, 'kind': kind})
        if f.suffix.lower() in {'.npz', '.npy', '.pt', '.pth', '.ckpt', '.pkl', '.pickle', '.png', '.jpg', '.jpeg', '.pdf', '.zip'}:
            findings.append({'path': name, 'kind': 'excluded_binary_type'})
        if f.name.lower() in {'.env', 'credentials', 'kubeconfig', 'config.json'} and '.git' in f.parts:
            findings.append({'path': name, 'kind': 'credential_file_name'})
    rows.sort(key=lambda x: x['path'])
    if args.write_manifest:
        manifest = {'algorithm': 'sha256', 'scope': 'Public files excluding this inventory, PUBLIC_AUDIT.json and ignored/generated caches.',
                    'files': rows}
        (ROOT / 'PUBLIC_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    else:
        manifest = json.loads((ROOT / 'PUBLIC_MANIFEST.json').read_text(encoding='utf-8'))
        if rows != manifest['files']:
            findings.append({'path': 'PUBLIC_MANIFEST.json', 'kind': 'inventory_or_hash_mismatch'})
    report = {'all_passed': not findings, 'public_file_count_excluding_audit_and_manifest': len(rows),
              'public_file_bytes_excluding_audit_and_manifest': sum(r['bytes'] for r in rows),
              'largest_file': max(rows, key=lambda x: x['bytes']), 'findings': findings,
              'scanner_scope': list(PATTERNS) + ['excluded_binary_type'],
              'limitation': 'Pattern scan and allowlisted export, not a guarantee against every credential format. No image-level model rerun or test-independence claim.'}
    if args.write_manifest:
        (ROOT / 'PUBLIC_AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
