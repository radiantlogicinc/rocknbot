"""
Unit tests for channel_check.log_configured_channels(): the startup log of
which Slack channel each configured channel ID names. No live Slack calls:
conversations.info is injected.

Run from lil-lisa:
    PYTHONPATH=src python3 tests/test_channel_check.py
"""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import channel_check  # noqa: E402


def channel(name, **flags):
    return {"ok": True, "channel": {"name": name, **flags}}


class FakeInfo:
    def __init__(self, payloads):
        self.payloads = dict(payloads)
        self.calls = []

    async def __call__(self, channel_id):
        self.calls.append(channel_id)
        outcome = self.payloads[channel_id]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def run(coro):
    return asyncio.run(coro)


class LogConfiguredChannelsTests(unittest.TestCase):
    def test_each_configured_channel_is_logged_with_its_name(self):
        info = FakeInfo({"C1": channel("rocknbot-ida", is_member=True), "C2": channel("techsupport", is_private=True)})
        env = {"CHANNEL_ID_IDA": "C1", "TECHSUPPORT_CHANNEL_ID_IDA": " C2 ", "CHANNEL_ID_IDO": None}

        with patch.object(channel_check.logger, "info") as info_log, patch.object(channel_check.logger, "warning") as warn:
            result = run(channel_check.log_configured_channels(env, info))

        self.assertEqual(result, {"CHANNEL_ID_IDA": "#rocknbot-ida", "TECHSUPPORT_CHANNEL_ID_IDA": "#techsupport [private]"})
        self.assertEqual(info.calls, ["C1", "C2"])
        self.assertIn("CHANNEL_ID_IDA = C1 (#rocknbot-ida)", info_log.call_args_list[0].args[0])
        warn.assert_not_called()

    def test_problems_are_warnings_and_never_raise(self):
        info = FakeInfo(
            {
                "C1": {"ok": False, "error": "channel_not_found"},
                "C2": channel("old", is_archived=True),
                "C3": channel("iddm", is_member=False),
                "C4": RuntimeError("network down"),
            }
        )
        env = {
            "CHANNEL_ID_IDA": "C1",
            "ADMIN_CHANNEL_ID_IDA": "C2",
            "CHANNEL_ID_IDDM": "C3",
            "ADMIN_CHANNEL_ID_IDDM": "C4",
        }

        with patch.object(channel_check.logger, "warning") as warn:
            result = run(channel_check.log_configured_channels(env, info))

        self.assertEqual(warn.call_count, 4)
        self.assertIn("channel_not_found", result["CHANNEL_ID_IDA"])
        self.assertIn("ARCHIVED", result["ADMIN_CHANNEL_ID_IDA"])
        self.assertIn("bot is NOT a member", result["CHANNEL_ID_IDDM"])
        self.assertIn("network down", result["ADMIN_CHANNEL_ID_IDDM"])

    def test_missing_scope_names_the_scopes(self):
        ok, text = channel_check.describe_channel({"ok": False, "error": "missing_scope"})
        self.assertFalse(ok)
        self.assertIn("groups:read", text)

    def test_nothing_configured_makes_no_calls(self):
        info = FakeInfo({})
        self.assertEqual(run(channel_check.log_configured_channels({}, info)), {})
        self.assertEqual(info.calls, [])


if __name__ == "__main__":
    unittest.main()
