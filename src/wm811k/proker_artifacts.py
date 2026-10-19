"""Hash-bound, atomic artifacts for the new ProKeR comparison."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import numpy as np


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temp, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def require_binding(payload, expected):
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f'artifact binding mismatch: {field}')


def write_receipt(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    with path.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(sha256_file(path) + '\n')


def verify_receipt(path, *, base_dir=None):
    path = Path(path)
    if sha256_file(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('freeze receipt hash mismatch')
    receipt = load_json(path)
    if receipt.get('schema_version') != 'proker_freeze_receipt_v1':
        raise ValueError('unsupported freeze receipt')
    for name, digest in receipt['file_hashes'].items():
        target = Path(name) if base_dir is None else Path(base_dir) / name
        if sha256_file(target) != digest:
            raise ValueError(f'frozen input hash mismatch: {name}')
    return receipt
