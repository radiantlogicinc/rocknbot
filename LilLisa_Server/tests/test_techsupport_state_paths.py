"""
Unit tests for where the techsupport pipeline keeps its runtime files.

Run from LilLisa_Server:
    PYTHONPATH=. python3 tests/test_techsupport_state_paths.py
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.techsupport_state_paths import resolve_state_dir, resolve_verified_qa_dir  # noqa: E402

ROOT = Path("/srv/LilLisa_Server")


class ResolveTests(unittest.TestCase):
    def test_default_sits_next_to_absolute_lancedb_folder(self):
        env = {"LANCEDB_FOLDERPATH": "/data/lancedb"}
        self.assertEqual(resolve_state_dir(env, ROOT), Path("/data/techsupport_state"))
        self.assertEqual(
            resolve_verified_qa_dir(env, ROOT),
            Path("/data/techsupport_state/verified_techsupport"),
        )

    def test_relative_lancedb_folder_resolves_against_server_root(self):
        env = {"LANCEDB_FOLDERPATH": "./lancedb"}
        self.assertEqual(resolve_state_dir(env, ROOT), ROOT / "techsupport_state")

    def test_missing_lancedb_folder_uses_the_example_default(self):
        self.assertEqual(resolve_state_dir({}, ROOT), ROOT / "techsupport_state")

    def test_state_dir_override(self):
        env = {"LANCEDB_FOLDERPATH": "/data/lancedb", "TECHSUPPORT_STATE_DIR": "/mnt/state"}
        self.assertEqual(resolve_state_dir(env, ROOT), Path("/mnt/state"))
        self.assertEqual(resolve_verified_qa_dir(env, ROOT), Path("/mnt/state/verified_techsupport"))

    def test_relative_state_dir_override_resolves_against_server_root(self):
        env = {"TECHSUPPORT_STATE_DIR": "state"}
        self.assertEqual(resolve_state_dir(env, ROOT), ROOT / "state")

    def test_verified_qa_override_is_independent(self):
        env = {"LANCEDB_FOLDERPATH": "/data/lancedb", "VERIFIED_TECHSUPPORT_QA_FOLDERPATH": "/mnt/md"}
        self.assertEqual(resolve_verified_qa_dir(env, ROOT), Path("/mnt/md"))
        self.assertEqual(resolve_state_dir(env, ROOT), Path("/data/techsupport_state"))

    def test_blank_values_fall_back_to_defaults(self):
        env = {"LANCEDB_FOLDERPATH": "/data/lancedb", "TECHSUPPORT_STATE_DIR": "  "}
        self.assertEqual(resolve_state_dir(env, ROOT), Path("/data/techsupport_state"))


if __name__ == "__main__":
    unittest.main()
