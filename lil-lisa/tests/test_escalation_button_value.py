"""
Unit tests for Slack escalate button JSON (2000-char cap).

Run from lil-lisa:
    PYTHONPATH=src python3 tests/test_escalation_button_value.py
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import (  # noqa: E402
    SLACK_ACTION_VALUE_MAX,
    build_escalation_button_value,
    warn_if_escalate_body_channel_mismatch,
)


class EscalationButtonValueTests(unittest.TestCase):
    def test_short_title_round_trips(self):
        blob = build_escalation_button_value(
            "how do I bind?",
            "C111",
            "123.4",
            "sess",
            "Uasker",
            primary_techsupport_match_title="LDAP bind timeout",
        )
        payload = json.loads(blob)
        self.assertEqual(payload["primary_techsupport_match_title"], "LDAP bind timeout")
        self.assertLessEqual(len(blob), SLACK_ACTION_VALUE_MAX)

    def test_long_title_is_truncated_to_fit(self):
        title = "T" * 2000
        blob = build_escalation_button_value(
            "q" * 1500,
            "C111",
            "123.4",
            "sess",
            "Uasker",
            primary_techsupport_match_title=title,
        )
        self.assertLessEqual(len(blob), SLACK_ACTION_VALUE_MAX)
        payload = json.loads(blob)
        self.assertIn("query", payload)
        if "primary_techsupport_match_title" in payload:
            self.assertLess(len(payload["primary_techsupport_match_title"]), 2000)


class BodyChannelMismatchTests(unittest.TestCase):
    def test_warns_when_body_channel_differs(self):
        with self.assertLogs("RL_Logger", level="WARNING") as captured:
            warned = warn_if_escalate_body_channel_mismatch(
                {"channel": {"id": "Cforwarded"}},
                "Cpayload",
            )
        self.assertTrue(warned)
        self.assertTrue(any("Cforwarded" in line and "Cpayload" in line for line in captured.output))

    def test_no_warn_when_channels_match(self):
        with self.assertNoLogs("RL_Logger", level="WARNING"):
            self.assertFalse(
                warn_if_escalate_body_channel_mismatch({"channel": {"id": "Csame"}}, "Csame")
            )

    def test_no_warn_when_body_channel_missing(self):
        with self.assertNoLogs("RL_Logger", level="WARNING"):
            self.assertFalse(warn_if_escalate_body_channel_mismatch({}, "Cpayload"))
            self.assertFalse(warn_if_escalate_body_channel_mismatch({"channel": {}}, "Cpayload"))


if __name__ == "__main__":
    unittest.main()
