import hashlib
import tempfile
import unittest
from pathlib import Path

from fretwork_core.checkpoints import verify_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_verifies_the_expected_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.pth'
            path.write_bytes(b'fixture')
            self.assertEqual(verify_checkpoint(path, hashlib.sha256(b'fixture').hexdigest()), path)

    def test_rejects_untrusted_weights_without_deleting_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.pth'
            path.write_bytes(b'untrusted')
            with self.assertRaises(RuntimeError): verify_checkpoint(path)
            self.assertEqual(path.read_bytes(), b'untrusted')
