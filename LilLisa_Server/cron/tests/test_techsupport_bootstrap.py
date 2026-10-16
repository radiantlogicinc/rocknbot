"""
Unit tests for techsupport_bootstrap.run_bootstrap(): the startup step that
clones techsupport_qa_pairs.md from GITHUB_REPO_URL (the source of truth)
over the local copy, then rebuilds TECHSUPPORT_QA_PAIRS from it when the
table's row count doesn't already match the markdown's entry count.

No git, no LLM, no network, no real LanceDB -- sync_verified_qa_pairs(),
_lancedb_row_count(), and run_reembed_if_due() are all patched out; these
tests only exercise the decision logic around them.

Run from LilLisa_Server/cron:
    PYTHONPATH=. python3 tests/test_techsupport_bootstrap.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

# techsupport_bootstrap -> techsupport_contextual_reembed -> paths ->
# src.techsupport_state_paths -> src.utils, whose module-level
# LILLISA_SERVER_ENV_DICT["LOG_LEVEL"] lookup is the one hard env requirement
# in that chain -- same fix as test_techsupport_correct_entry.py's setUp
# comment describes.
os.environ.setdefault("LOG_LEVEL", "INFO")

import techsupport_bootstrap as bootstrap  # noqa: E402


class RunBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.markdown_path = Path(self.tmpdir.name) / "techsupport_qa_pairs.md"
        self._enter(patch.object(bootstrap, "MARKDOWN_PATH", self.markdown_path))
        self.run_reembed_if_due = self._enter(
            patch.object(bootstrap, "run_reembed_if_due", return_value={"ran": True, "forced": True})
        )

    def _enter(self, cm):
        value = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return value

    def _write_markdown(self, entry_count: int) -> None:
        self.markdown_path.write_text(
            "".join(f"## Entry {i}\n\nBody {i}.\n\n" for i in range(entry_count)),
            encoding="utf-8",
        )

    def test_no_local_markdown_and_nothing_pulled_skips_rebuild(self):
        nothing_to_sync = {"direction": "none", "pull": {"pulled": False, "reason": "remote_missing"}}
        with patch.object(bootstrap, "sync_verified_qa_pairs", return_value=nothing_to_sync):
            result = bootstrap.run_bootstrap()
        self.assertEqual(result["markdown_sync"]["direction"], "none")
        self.assertEqual(result["lancedb_rebuild"], {"ran": False, "reason": "no_markdown_entries_yet"})
        self.run_reembed_if_due.assert_not_called()

    def test_pull_populates_markdown_then_rebuild_runs(self):
        def fake_sync():
            self._write_markdown(5)
            return {"direction": "pull", "pull": {"pulled": True, "entry_count": 5}}

        with patch.object(bootstrap, "sync_verified_qa_pairs", side_effect=fake_sync), patch.object(
            bootstrap, "_lancedb_row_count", return_value=0
        ):
            result = bootstrap.run_bootstrap()

        self.assertEqual(result["markdown_sync"]["direction"], "pull")
        self.assertEqual(result["lancedb_rebuild"], {"ran": True, "forced": True})
        self.run_reembed_if_due.assert_called_once_with(force=True)

    def test_push_direction_still_checks_lancedb_against_the_local_markdown(self):
        # Local was ahead and got pushed; the local markdown is unchanged and
        # is still what TECHSUPPORT_QA_PAIRS must match.
        self._write_markdown(640)
        pushed = {"direction": "push", "pull": {"pulled": False, "reason": "local_ahead"}, "push": {"pushed": True}}
        with patch.object(bootstrap, "sync_verified_qa_pairs", return_value=pushed), patch.object(
            bootstrap, "_lancedb_row_count", return_value=640
        ):
            result = bootstrap.run_bootstrap()

        self.assertEqual(result["markdown_sync"]["direction"], "push")
        self.assertEqual(result["lancedb_rebuild"]["reason"], "already_in_sync")
        self.run_reembed_if_due.assert_not_called()

    def test_matching_row_count_skips_rebuild(self):
        self._write_markdown(5)
        with patch.object(bootstrap, "sync_verified_qa_pairs", return_value={"direction": "none", "pull": {"pulled": False}}), patch.object(
            bootstrap, "_lancedb_row_count", return_value=5
        ):
            result = bootstrap.run_bootstrap()

        self.assertEqual(
            result["lancedb_rebuild"],
            {"ran": False, "reason": "already_in_sync", "entry_count": 5, "row_count": 5},
        )
        self.run_reembed_if_due.assert_not_called()

    def test_mismatched_row_count_triggers_forced_rebuild(self):
        self._write_markdown(638)
        with patch.object(bootstrap, "sync_verified_qa_pairs", return_value={"direction": "none", "pull": {"pulled": False}}), patch.object(
            bootstrap, "_lancedb_row_count", return_value=0
        ):
            result = bootstrap.run_bootstrap()

        self.assertEqual(result["lancedb_rebuild"], {"ran": True, "forced": True})
        self.run_reembed_if_due.assert_called_once_with(force=True)

    def test_sync_failure_is_captured_and_skips_the_rebuild(self):
        # Even with a populated local markdown, a failed sync means the
        # baseline is unreconciled -- the caller aborts startup, so rebuilding
        # LanceDB from it first would be wasted work.
        self._write_markdown(3)
        with patch.object(bootstrap, "sync_verified_qa_pairs", side_effect=RuntimeError("no credentials")):
            result = bootstrap.run_bootstrap()

        self.assertIn("no credentials", result["markdown_sync"]["error"])
        self.assertEqual(result["lancedb_rebuild"], {"ran": False, "reason": "markdown_sync_failed"})
        self.run_reembed_if_due.assert_not_called()

    def test_lancedb_check_failure_is_captured_not_raised(self):
        self._write_markdown(3)
        with patch.object(bootstrap, "sync_verified_qa_pairs", return_value={"direction": "none", "pull": {"pulled": False}}), patch.object(
            bootstrap, "_lancedb_row_count", side_effect=RuntimeError("lancedb unreachable")
        ):
            result = bootstrap.run_bootstrap()

        self.assertIn("lancedb unreachable", result["lancedb_rebuild"]["error"])
        self.run_reembed_if_due.assert_not_called()

    def test_rebuild_failure_is_captured_not_raised(self):
        self._write_markdown(3)
        self.run_reembed_if_due.side_effect = RuntimeError("rebuild blew up")
        with patch.object(bootstrap, "sync_verified_qa_pairs", return_value={"direction": "none", "pull": {"pulled": False}}), patch.object(
            bootstrap, "_lancedb_row_count", return_value=0
        ):
            result = bootstrap.run_bootstrap()

        self.assertIn("rebuild blew up", result["lancedb_rebuild"]["error"])


if __name__ == "__main__":
    unittest.main()
