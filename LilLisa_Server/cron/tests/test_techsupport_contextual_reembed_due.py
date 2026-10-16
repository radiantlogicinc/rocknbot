"""
Unit tests for techsupport_contextual_reembed.py's state-gating logic:
is_reembed_due() / run_if_due() / the --force override.

No LLM, no LanceDB, no network -- run_reembed() (the actual rebuild) is
patched out entirely; these tests only exercise the gating decision around
it, since that decision -- NOT auto-rebuilding on a missing/reset state
file -- is the thing a lost techsupport_reembed_state.json must not bypass
(see PR42 Release and Deployment Notes.md's incident writeup).

Run from LilLisa_Server/cron:
    PYTHONPATH=. python3 tests/test_techsupport_contextual_reembed_due.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

# techsupport_contextual_reembed -> paths -> src.techsupport_state_paths ->
# src.utils, whose module-level LILLISA_SERVER_ENV_DICT["LOG_LEVEL"] lookup is
# the one hard env requirement in that chain -- same fix as
# test_techsupport_correct_entry.py's setUp comment describes.
os.environ.setdefault("LOG_LEVEL", "INFO")

import techsupport_contextual_reembed as reembed  # noqa: E402


class IsReembedDueTests(unittest.TestCase):
    """A missing/empty "last_reembed_timestamp" must mean "not due" --
    load_state() already returns exactly that shape when the state file
    doesn't exist (see its own docstring/behavior)."""

    def test_missing_timestamp_is_not_due(self):
        self.assertFalse(reembed.is_reembed_due({"last_reembed_timestamp": None}))
        self.assertFalse(reembed.is_reembed_due({}))

    def test_recent_timestamp_is_not_due(self):
        with patch.object(reembed, "get_reembed_interval_days", return_value=7):
            state = {"last_reembed_timestamp": f"{reembed.time.time():.6f}"}
            self.assertFalse(reembed.is_reembed_due(state))

    def test_old_timestamp_is_due(self):
        with patch.object(reembed, "get_reembed_interval_days", return_value=7):
            eight_days_ago = reembed.time.time() - 8 * 86400
            state = {"last_reembed_timestamp": f"{eight_days_ago:.6f}"}
            self.assertTrue(reembed.is_reembed_due(state))


class RunIfDueTests(unittest.TestCase):
    """run_if_due() wraps is_reembed_due() with the actual rebuild + state
    save, plus the force override."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.state_path = Path(self.tmpdir.name) / "techsupport_reembed_state.json"
        self._enter(patch.object(reembed, "STATE_PATH", self.state_path))
        self.run_reembed = self._enter(
            patch.object(reembed, "run_reembed", return_value={"table_name": "T", "row_count": 3})
        )

    def _enter(self, cm):
        value = cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)
        return value

    def test_missing_state_file_does_not_auto_rebuild(self):
        self.assertFalse(self.state_path.exists())
        result = reembed.run_if_due()
        self.assertFalse(result["ran"])
        self.assertEqual(result["reason"], "no_prior_reembed_on_record")
        self.run_reembed.assert_not_called()
        # No state file should be created either -- a no-op must stay a no-op.
        self.assertFalse(self.state_path.exists())

    def test_interval_not_elapsed_does_not_rebuild(self):
        reembed.save_state({"last_reembed_timestamp": f"{reembed.time.time():.6f}"})
        result = reembed.run_if_due()
        self.assertFalse(result["ran"])
        self.assertEqual(result["reason"], "interval_not_elapsed")
        self.run_reembed.assert_not_called()

    def test_elapsed_interval_rebuilds_and_updates_state(self):
        eight_days_ago = reembed.time.time() - 8 * 86400
        reembed.save_state({"last_reembed_timestamp": f"{eight_days_ago:.6f}"})
        result = reembed.run_if_due()
        self.assertTrue(result["ran"])
        self.assertFalse(result["forced"])
        self.run_reembed.assert_called_once()
        new_state = reembed.load_state()
        self.assertGreater(float(new_state["last_reembed_timestamp"]), eight_days_ago + 7 * 86400)

    def test_force_rebuilds_despite_missing_state(self):
        self.assertFalse(self.state_path.exists())
        result = reembed.run_if_due(force=True)
        self.assertTrue(result["ran"])
        self.assertTrue(result["forced"])
        self.run_reembed.assert_called_once()
        # A forced run still records state, starting the normal interval clock.
        self.assertTrue(self.state_path.exists())

    def test_force_rebuilds_despite_interval_not_elapsed(self):
        reembed.save_state({"last_reembed_timestamp": f"{reembed.time.time():.6f}"})
        result = reembed.run_if_due(force=True)
        self.assertTrue(result["ran"])
        self.run_reembed.assert_called_once()


if __name__ == "__main__":
    unittest.main()
