"""Check a new public release identity while retaining original freeze receipts.

The public-byte SHA is a separate identity from the original historical SHA.
Only exact LF/CRLF transport candidates for UTF-8 text may match the declared
public identity. Neither source files nor historical protocols are rewritten.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json

TEXT_SUFFIXES = {'.json', '.csv', '.md', '.txt', '.py', '.bib', '.tex', '.yml', '.yaml'}


def checked_path(root, name):
    root = Path(root).resolve()
    path = PurePosixPath(name)
    if not name or '\\' in name or path.is_absolute() or '..' in path.parts or ':' in path.parts[0]:
        raise ValueError(f'Unsafe release path: {name}')
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f'Release path escapes root: {name}')
    return result


def verified_bytes(root, entry):
    path = checked_path(root, entry['path'])
    payload = path.read_bytes()
    expected = entry['public_sha256']
    candidates = [('exact', payload)]
    if path.suffix.lower() in TEXT_SUFFIXES:
        try:
            payload.decode('utf-8-sig')
        except UnicodeDecodeError:
            pass
        else:
            lf = payload.replace(b'\r\n', b'\n')
            candidates.extend([('LF', lf), ('CRLF', lf.replace(b'\n', b'\r\n'))])
    for transport, candidate in candidates:
        if hashlib.sha256(candidate).hexdigest() == expected:
            if len(candidate) != entry['public_bytes']:
                raise ValueError(f'Declared public length differs: {entry["path"]}')
            return candidate, transport
    raise ValueError(f'Public SHA-256 mismatch: {entry["path"]}')


def verify_entries(root, entries, groups=('evidence',)):
    selected = set(groups)
    seen = set()
    transports = []
    verified = []
    skipped = []
    for entry in entries:
        name = entry['path']
        checked_path(root, name)
        if name in seen:
            raise ValueError(f'Duplicate release path: {name}')
        seen.add(name)
        if entry['group'] not in selected:
            skipped.append(name)
            continue
        _, transport = verified_bytes(root, entry)
        verified.append(name)
        if transport != 'exact':
            transports.append(name)
    return {'all_passed': True, 'requested_groups': sorted(selected),
            'verified_count': len(verified), 'line_ending_transports': transports,
            'unrequested_count': len(skipped),
            'scope': 'Public release byte integrity only; not model rerun or original-freeze byte equivalence.'}


def verify_release(root, with_caches=False):
    root = Path(root)
    manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'public_reproduction_release_v1':
        raise ValueError('Unknown public release manifest schema')
    groups = ('evidence', 'cache') if with_caches else ('evidence',)
    return verify_entries(root, manifest['files'], groups=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-caches', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify_release(Path(__file__).resolve().parent, args.with_caches), indent=2))


if __name__ == '__main__':
    main()
