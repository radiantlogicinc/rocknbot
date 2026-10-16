"""
Unit tests for src/techsupport_cron.run_product_channel_scan_once (pr42-w1x.8):
the in-process driver behind POST /run_product_channel_scan/.

What matters here is that the manual product-channel scan and the nightly
pipeline serialise against ONE lock. They write the same
techsupport_sync_state.json and the same verified-entry store, so overlapping
them in either direction would corrupt the dedup bookkeeping. Overlaps are
dropped, never queued, exactly like two nightly triggers.

src.main is deliberately NOT imported here (it pulls the whole server), so
POST /run_product_channel_scan/ itself is not covered by a test; the handler is
a thin JWT check plus a BackgroundTask over the function tested below.

Run from LilLisa_Server:
    PYTHONPATH=. python3 tests/test_techsupport_cron_product_scan.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import techsupport_cron  # noqa: E402


class RunProductChannelScanOnceTests(unittest.TestCase):
    def setUp(self):
        self.assertFalse(techsupport_cron.is_running(), "a previous test left the run lock held")

    def test_force_is_passed_through_and_the_summary_is_returned(self):
        with patch.object(techsupport_cron, "run_product_channel_scan", return_value={"product_channels": {}}) as scan:
            result = techsupport_cron.run_product_channel_scan_once(force=True)
        scan.assert_called_once_with(force=True)
        self.assertTrue(result["ran"])
        self.assertEqual(result["summary"], {"product_channels": {}})

    def test_force_defaults_to_false(self):
        with patch.object(techsupport_cron, "run_product_channel_scan", return_value={}) as scan:
            techsupport_cron.run_product_channel_scan_once()
        scan.assert_called_once_with(force=False)

    def test_a_failing_scan_is_reported_not_raised(self):
        # The caller is a FastAPI BackgroundTask, which would swallow a raise.
        with patch.object(techsupport_cron, "run_product_channel_scan", side_effect=RuntimeError("slack down")):
            result = techsupport_cron.run_product_channel_scan_once()
        self.assertFalse(result["ran"])
        self.assertEqual(result["error"], "slack down")
        self.assertFalse(techsupport_cron.is_running(), "the lock must be released on failure")

    def test_unavailable_cron_package_is_reported(self):
        with patch.object(techsupport_cron, "run_product_channel_scan", None):
            result = techsupport_cron.run_product_channel_scan_once()
        self.assertEqual(result, {"ran": False, "reason": "unavailable", "error": techsupport_cron.IMPORT_ERROR})

    def test_a_scan_in_flight_blocks_the_nightly_run(self):
        nested = {}

        def scan(force=False):
            nested["is_running"] = techsupport_cron.is_running()
            nested["nightly"] = techsupport_cron.run_once()
            return {}

        with (
            patch.object(techsupport_cron, "run_product_channel_scan", side_effect=scan),
            patch.object(techsupport_cron, "run_pipeline") as pipeline,
        ):
            result = techsupport_cron.run_product_channel_scan_once()

        self.assertTrue(result["ran"])
        self.assertTrue(nested["is_running"])
        self.assertEqual(nested["nightly"], {"ran": False, "reason": "already_running"})
        pipeline.assert_not_called()

    def test_a_nightly_run_in_flight_blocks_the_scan(self):
        nested = {}

        def pipeline(force=False):
            nested["scan"] = techsupport_cron.run_product_channel_scan_once(force=True)
            return {}

        with (
            patch.object(techsupport_cron, "run_pipeline", side_effect=pipeline),
            patch.object(techsupport_cron, "run_product_channel_scan") as scan,
        ):
            result = techsupport_cron.run_once()

        self.assertTrue(result["ran"])
        self.assertEqual(nested["scan"], {"ran": False, "reason": "already_running"})
        scan.assert_not_called()


if __name__ == "__main__":
    unittest.main()
