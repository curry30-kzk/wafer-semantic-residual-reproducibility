"""Release verification must detect changes without rewriting frozen inputs."""
import hashlib
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


class ReleaseIntegrity(unittest.TestCase):
    def verifier(self):
        if importlib.util.find_spec('release_integrity') is None:
            self.fail('The explicit public-release integrity verifier is not implemented')
        return importlib.import_module('release_integrity').verify_entries

    @staticmethod
    def row(name, payload, group='evidence'):
        return {'path': name, 'public_sha256': hashlib.sha256(payload).hexdigest(),
                'public_bytes': len(payload), 'group': group}

    def test_numeric_edit_is_rejected(self):
        verify = self.verifier()
        original = b'{"macro_f1": 0.4}\n'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'result.json').write_bytes(b'{"macro_f1": 0.5}\n')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                verify(root, [self.row('result.json', original)])

    def test_crlf_transport_matches_only_declared_public_text_identity(self):
        verify = self.verifier()
        payload = b'{\n"macro_f1": 0.4\n}\n'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            actual = payload.replace(b'\n', b'\r\n')
            (root / 'result.json').write_bytes(actual)
            result = verify(root, [self.row('result.json', payload)])
            self.assertTrue(result['all_passed'])
            self.assertEqual(result['line_ending_transports'], ['result.json'])
            self.assertEqual((root / 'result.json').read_bytes(), actual)

    def test_parent_escape_is_rejected(self):
        verify = self.verifier()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'path'):
                verify(Path(folder), [self.row('../secret.txt', b'private')])

    def test_citation_cff_crlf_transport_matches_declared_identity(self):
        verify = self.verifier()
        payload = b'cff-version: 1.2.0\nmessage: Please cite this version.\n'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'CITATION.cff').write_bytes(payload.replace(b'\n', b'\r\n'))
            result = verify(root, [self.row('CITATION.cff', payload)])
            self.assertTrue(result['all_passed'])
            self.assertEqual(result['line_ending_transports'], ['CITATION.cff'])

    def test_required_cache_absence_is_not_reported_as_success(self):
        verify = self.verifier()
        with tempfile.TemporaryDirectory() as folder:
            row = self.row('arrays.npz', b'PK\x00\x01', 'cache')
            with self.assertRaises(FileNotFoundError):
                verify(Path(folder), [row], groups=('evidence', 'cache'))

    def test_binary_changes_are_rejected_without_line_ending_repair(self):
        verify = self.verifier()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = b'PK\r\n\x00\xff'
            (root / 'arrays.npz').write_bytes(original.replace(b'\r\n', b'\n'))
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                verify(root, [self.row('arrays.npz', original, 'cache')], groups=('cache',))


if __name__ == '__main__':
    unittest.main()
