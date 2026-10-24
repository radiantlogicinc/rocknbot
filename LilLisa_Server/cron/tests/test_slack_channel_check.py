"""
Unit tests for slack_channel_check.log_configured_channels(): the startup log
of which Slack channel each configured channel ID names. No live Slack calls:
conversations.info is injected, or requests.get is mocked.

Run from LilLisa_Server/cron:
    PYTHONPATH=. python3 tests/test_slack_channel_check.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import slack_channel_check as scc  # noqa: E402


def channel(name, **flags):
    return {"ok": True, "channel": {"name": name, **flags}}


class FakeInfo:
    def __init__(self, payloads):
        self.payloads = dict(payloads)
        self.calls = []

    def __call__(self, channel_id):
        self.calls.append(channel_id)
        outcome = self.payloads[channel_id]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class LogConfiguredChannelsTests(unittest.TestCase):
    def test_each_configured_channel_is_logged_with_its_name(self):
        info = FakeInfo({"C1": channel("techsupport", is_member=True), "C2": channel("rocknbot-ida")})
        env = {"TECHSUPPORT_CHANNEL_ID": "C1", "CHANNEL_ID_IDA": " C2 ", "CHANNEL_ID_IDDM": ""}

        with self.assertLogs(scc.logger, level="INFO") as logs:
            result = scc.log_configured_channels(env, fetch_channel_info=info)

        self.assertEqual(result, {"TECHSUPPORT_CHANNEL_ID": "#techsupport", "CHANNEL_ID_IDA": "#rocknbot-ida"})
        self.assertEqual(info.calls, ["C1", "C2"])
        self.assertIn("TECHSUPPORT_CHANNEL_ID = C1 (#techsupport)", logs.output[0])

    def test_problems_are_warnings_and_never_raise(self):
        info = FakeInfo(
            {
                "C1": {"ok": False, "error": "channel_not_found"},
                "C2": channel("old", is_archived=True),
                "C3": channel("ida", is_private=True, is_member=False),
                "C4": RuntimeError("network down"),
            }
        )
        env = {
            "TECHSUPPORT_CHANNEL_ID": "C1",
            "ADMIN_CHANNEL_ID": "C2",
            "CHANNEL_ID_IDA": "C3",
            "CHANNEL_ID_IDDM": "C4",
        }

        with self.assertLogs(scc.logger, level="WARNING") as logs:
            result = scc.log_configured_channels(env, fetch_channel_info=info)

        self.assertEqual(len(logs.output), 4)
        self.assertIn("channel_not_found", result["TECHSUPPORT_CHANNEL_ID"])
        self.assertIn("ARCHIVED", result["ADMIN_CHANNEL_ID"])
        self.assertIn("bot is NOT a member", result["CHANNEL_ID_IDA"])
        self.assertIn("network down", result["CHANNEL_ID_IDDM"])

    def test_missing_scope_names_the_scopes(self):
        ok, text = scc.describe_channel({"ok": False, "error": "missing_scope"})
        self.assertFalse(ok)
        self.assertIn("groups:read", text)

    def test_nothing_configured_makes_no_calls(self):
        info = FakeInfo({})
        self.assertEqual(scc.log_configured_channels({}, fetch_channel_info=info), {})
        self.assertEqual(info.calls, [])

    def test_no_token_skips_with_a_warning(self):
        with self.assertLogs(scc.logger, level="WARNING"):
            self.assertEqual(scc.log_configured_channels({"TECHSUPPORT_CHANNEL_ID": "C1"}), {})

    def test_default_fetcher_calls_conversations_info(self):
        env = {"TECHSUPPORT_CHANNEL_ID": "C1", "SLACK_BOT_TOKEN": "xoxb-test"}
        with patch.object(scc.requests, "get", return_value=FakeResponse(channel("techsupport"))) as get:
            result = scc.log_configured_channels(env)

        self.assertEqual(result, {"TECHSUPPORT_CHANNEL_ID": "#techsupport"})
        self.assertTrue(get.call_args.args[0].endswith("/conversations.info"))
        self.assertEqual(get.call_args.kwargs["params"], {"channel": "C1"})
        self.assertEqual(get.call_args.kwargs["headers"], {"Authorization": "Bearer xoxb-test"})


if __name__ == "__main__":
    unittest.main()
