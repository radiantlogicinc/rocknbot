"""
Unit tests for nightly techsupport thread-update detection (hot window,
catch-up age cap, parent latest_reply lookups) and for the channel-keyed
state file (v2), which is required rather than repaired.

Run from LilLisa_Server/cron:
    PYTHONPATH=. python3 tests/test_nightly_techsupport_sync.py
"""

import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import nightly_techsupport_sync as sync_mod  # noqa: E402


class SelectParentLookupsTests(unittest.TestCase):
    def setUp(self):
        self.now = 2_000_000_000.0
        day = sync_mod.SECONDS_PER_DAY
        self.threads = {
            "hot": {"last_seen_reply_ts": str(self.now - 5 * day)},
            "warm": {"last_seen_reply_ts": str(self.now - 45 * day)},
            "cold": {"last_seen_reply_ts": str(self.now - 120 * day)},
            "hottest": {"last_seen_reply_ts": str(self.now - 1 * day)},
        }
        self.ids = ["hot", "warm", "cold", "hottest"]

    def test_hot_window_excludes_warm_and_cold(self):
        chosen, capped = sync_mod.select_parent_lookups(
            self.ids, self.threads, self.now, window_days=30, max_lookups=200
        )
        self.assertEqual(chosen, ["hottest", "hot"])
        self.assertEqual(capped, 0)

    def test_catchup_age_cap_includes_warm_excludes_cold(self):
        chosen, capped = sync_mod.select_parent_lookups(
            self.ids, self.threads, self.now, window_days=90, max_lookups=200
        )
        self.assertEqual(chosen, ["hottest", "hot", "warm"])
        self.assertEqual(capped, 0)
        self.assertNotIn("cold", chosen)

    def test_cap_keeps_hottest_first(self):
        chosen, capped = sync_mod.select_parent_lookups(
            self.ids, self.threads, self.now, window_days=90, max_lookups=2
        )
        self.assertEqual(chosen, ["hottest", "hot"])
        self.assertEqual(capped, 1)

    def test_exclude_skips_already_checked_hot_ids(self):
        chosen, capped = sync_mod.select_parent_lookups(
            self.ids,
            self.threads,
            self.now,
            window_days=90,
            max_lookups=200,
            exclude=["hottest", "hot"],
        )
        self.assertEqual(chosen, ["warm"])
        self.assertEqual(capped, 0)

    def test_missing_last_seen_uses_parent_ts(self):
        parent_ts = str(self.now - 10 * sync_mod.SECONDS_PER_DAY)
        chosen, _ = sync_mod.select_parent_lookups([parent_ts], {}, self.now, window_days=30, max_lookups=10)
        self.assertEqual(chosen, [parent_ts])


class CatchupDueTests(unittest.TestCase):
    def test_missing_timestamp_is_due(self):
        self.assertTrue(sync_mod.is_catchup_due({}, now=100.0, interval_days=7))

    def test_recent_catchup_is_not_due(self):
        now = 1_000_000.0
        state = {"last_catchup_timestamp": str(now - 2 * sync_mod.SECONDS_PER_DAY)}
        self.assertFalse(sync_mod.is_catchup_due(state, now=now, interval_days=7))

    def test_stale_catchup_is_due(self):
        now = 1_000_000.0
        state = {"last_catchup_timestamp": str(now - 8 * sync_mod.SECONDS_PER_DAY)}
        self.assertTrue(sync_mod.is_catchup_due(state, now=now, interval_days=7))


class FetchParentLatestReplyTsTests(unittest.TestCase):
    def test_uses_history_not_replies_and_reads_latest_reply(self):
        with patch.object(sync_mod, "slack_api_call") as mock_call:
            mock_call.return_value = {
                "ok": True,
                "messages": [{"ts": "100.0", "latest_reply": "150.0"}],
            }
            newest = sync_mod.fetch_parent_latest_reply_ts("token", "C1", "100.0")
        self.assertEqual(newest, "150.0")
        mock_call.assert_called_once()
        args, _kwargs = mock_call.call_args
        self.assertEqual(args[0], "conversations.history")
        self.assertEqual(args[2]["oldest"], "100.0")
        self.assertEqual(args[2]["limit"], 1)
        self.assertEqual(args[2]["inclusive"], "true")

    def test_falls_back_to_parent_ts_when_no_replies(self):
        with patch.object(sync_mod, "slack_api_call") as mock_call:
            mock_call.return_value = {"ok": True, "messages": [{"ts": "100.0"}]}
            newest = sync_mod.fetch_parent_latest_reply_ts("token", "C1", "100.0")
        self.assertEqual(newest, "100.0")


class SyncHybridTests(unittest.TestCase):
    def test_nightly_only_looks_up_hot_threads_and_skips_replies(self):
        now = 2_000_000_000.0
        day = sync_mod.SECONDS_PER_DAY
        threads = {
            "hot": {"last_seen_reply_ts": str(now - 2 * day), "added_to_verified_db": True},
            "warm": {"last_seen_reply_ts": str(now - 45 * day)},
            "cold": {"last_seen_reply_ts": str(now - 120 * day)},
        }
        state = {
            "version": 2,
            "channels": {
                "C": {
                    "last_run_timestamp": str(now - day),
                    "last_catchup_timestamp": str(now - day),
                    "threads": threads,
                }
            },
        }
        parent_calls: List[str] = []

        def fake_fetch(_token, _channel, thread_ts):
            parent_calls.append(thread_ts)
            if thread_ts == "hot":
                return str(now - 1)
            return threads[thread_ts]["last_seen_reply_ts"]

        with (
            patch.object(sync_mod, "load_env", return_value={"SLACK_BOT_TOKEN": "t", "TECHSUPPORT_CHANNEL_ID": "C"}),
            patch.object(sync_mod, "load_state", return_value=state),
            patch.object(sync_mod, "save_state") as mock_save,
            patch.object(sync_mod, "find_new_threads", return_value=[]),
            patch.object(sync_mod, "fetch_parent_latest_reply_ts", side_effect=fake_fetch),
            patch.object(sync_mod, "paginate_messages") as mock_pages,
            patch.object(sync_mod, "get_hot_days", return_value=30.0),
            patch.object(sync_mod, "get_catchup_age_days", return_value=90.0),
            patch.object(sync_mod, "get_catchup_interval_days", return_value=7.0),
            patch.object(sync_mod, "get_max_parent_lookups", return_value=200),
            patch.object(sync_mod.time, "time", return_value=now),
            patch.object(sync_mod.time, "sleep"),
        ):
            result = sync_mod.sync()

        self.assertEqual(parent_calls, ["hot"])
        self.assertEqual(result["updated_thread_ids"], ["hot"])
        self.assertEqual(result["new_thread_ids"], [])
        mock_pages.assert_not_called()
        saved = mock_save.call_args[0][0]["channels"]["C"]
        self.assertTrue(saved["threads"]["hot"]["added_to_verified_db"])
        self.assertEqual(saved["threads"]["cold"]["last_seen_reply_ts"], str(now - 120 * day))
        self.assertEqual(saved["last_catchup_timestamp"], str(now - day))

    def test_catchup_looks_up_warm_but_not_cold(self):
        now = 2_000_000_000.0
        day = sync_mod.SECONDS_PER_DAY
        threads = {
            "hot": {"last_seen_reply_ts": str(now - 2 * day)},
            "warm": {"last_seen_reply_ts": str(now - 45 * day)},
            "cold": {"last_seen_reply_ts": str(now - 120 * day)},
        }
        state = {
            "version": 2,
            "channels": {"C": {"last_run_timestamp": str(now - day), "threads": threads}},
        }
        parent_calls: List[str] = []

        def fake_fetch(_token, _channel, thread_ts):
            parent_calls.append(thread_ts)
            if thread_ts == "warm":
                return str(now - 40 * day)
            return threads[thread_ts]["last_seen_reply_ts"]

        with (
            patch.object(sync_mod, "load_env", return_value={"SLACK_BOT_TOKEN": "t", "TECHSUPPORT_CHANNEL_ID": "C"}),
            patch.object(sync_mod, "load_state", return_value=state),
            patch.object(sync_mod, "save_state") as mock_save,
            patch.object(sync_mod, "find_new_threads", return_value=[]),
            patch.object(sync_mod, "fetch_parent_latest_reply_ts", side_effect=fake_fetch),
            patch.object(sync_mod, "get_hot_days", return_value=30.0),
            patch.object(sync_mod, "get_catchup_age_days", return_value=90.0),
            patch.object(sync_mod, "get_catchup_interval_days", return_value=7.0),
            patch.object(sync_mod, "get_max_parent_lookups", return_value=200),
            patch.object(sync_mod.time, "time", return_value=now),
            patch.object(sync_mod.time, "sleep"),
        ):
            result = sync_mod.sync()

        self.assertEqual(set(parent_calls), {"hot", "warm"})
        self.assertNotIn("cold", parent_calls)
        self.assertEqual(result["updated_thread_ids"], ["warm"])
        saved = mock_save.call_args[0][0]["channels"]["C"]
        self.assertEqual(saved["last_catchup_timestamp"], f"{now:.6f}")
        self.assertEqual(saved["threads"]["cold"]["last_seen_reply_ts"], str(now - 120 * day))


class StateFileVersionTests(unittest.TestCase):
    """The v2 file is the only shape load_state() accepts. The pre-v2 flat file
    (last_run_timestamp/last_catchup_timestamp/threads at the top level) never
    shipped, so it is rejected instead of being repaired."""

    FLAT = {
        "last_run_timestamp": "1700.0",
        "last_catchup_timestamp": "1600.0",
        "threads": {
            "100.0": {"last_seen_reply_ts": "150.0", "added_to_verified_db": True},
            "200.0": {"last_seen_reply_ts": "260.0"},
        },
    }

    V2 = {
        "version": 2,
        "channels": {
            "C_TECH": {
                "last_run_timestamp": "1700.0",
                "last_catchup_timestamp": "1600.0",
                "threads": {"100.0": {"last_seen_reply_ts": "150.0", "added_to_verified_db": True}},
            }
        },
    }

    @contextlib.contextmanager
    def _state_file(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "techsupport_sync_state.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(sync_mod, "STATE_PATH", path):
                yield path

    def test_missing_file_is_a_fresh_v2_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(sync_mod, "STATE_PATH", Path(tmp) / "nope.json"):
                self.assertEqual(sync_mod.load_state(), {"version": 2, "channels": {}})

    def test_flat_legacy_file_raises_naming_the_path(self):
        with self._state_file(self.FLAT) as path:
            with self.assertRaises(RuntimeError) as ctx:
                sync_mod.load_state()
        message = str(ctx.exception)
        self.assertIn(str(path), message)
        self.assertIn("channel-keyed", message)
        self.assertIn("Delete the file", message)

    def test_unknown_version_raises(self):
        with self._state_file({"version": 99, "channels": {}}):
            with self.assertRaises(RuntimeError):
                sync_mod.load_state()

    def test_v2_file_loads_unchanged(self):
        with self._state_file(self.V2):
            self.assertEqual(sync_mod.load_state(), self.V2)

    def test_channel_state_creates_an_empty_slice(self):
        state = {"version": 2, "channels": {}}
        slice_ = sync_mod.channel_state(state, "C_NEW")
        self.assertEqual(slice_, {"last_run_timestamp": "0", "threads": {}})
        self.assertIs(state["channels"]["C_NEW"], slice_)


class SyncPerChannelTests(unittest.TestCase):
    def test_sync_channel_id_writes_only_that_channel(self):
        now = 2_000_000_000.0
        state = {
            "version": 2,
            "channels": {
                "C_TECH": {
                    "last_run_timestamp": str(now - 100),
                    "threads": {"tech1": {"last_seen_reply_ts": str(now - 100)}},
                },
                "C_IDA": {
                    "last_run_timestamp": str(now - 200),
                    "threads": {"ida1": {"last_seen_reply_ts": str(now - 200)}},
                },
            },
        }

        with (
            patch.object(
                sync_mod,
                "load_env",
                return_value={"SLACK_BOT_TOKEN": "t", "TECHSUPPORT_CHANNEL_ID": "C_TECH"},
            ),
            patch.object(sync_mod, "load_state", return_value=state),
            patch.object(sync_mod, "save_state") as mock_save,
            patch.object(sync_mod, "find_new_threads", return_value=[{"ts": "300.0"}]) as mock_find,
            patch.object(sync_mod, "fetch_parent_latest_reply_ts", return_value=str(now - 1)),
            patch.object(sync_mod, "get_hot_days", return_value=30.0),
            patch.object(sync_mod, "get_catchup_age_days", return_value=90.0),
            patch.object(sync_mod, "get_catchup_interval_days", return_value=7.0),
            patch.object(sync_mod, "get_max_parent_lookups", return_value=200),
            patch.object(sync_mod.time, "time", return_value=now),
            patch.object(sync_mod.time, "sleep"),
        ):
            result = sync_mod.sync(channel_id="C_IDA")

        # find_new_threads was asked about the product channel, from that
        # channel's own last_run_timestamp.
        args, _kwargs = mock_find.call_args
        self.assertEqual(args[1], "C_IDA")
        self.assertEqual(args[2], str(now - 200))
        self.assertEqual(result["new_thread_ids"], ["300.0"])
        saved = mock_save.call_args[0][0]
        self.assertEqual(saved["channels"]["C_IDA"]["last_run_timestamp"], f"{now:.6f}")
        self.assertIn("300.0", saved["channels"]["C_IDA"]["threads"])
        # The techsupport channel's slice is untouched.
        self.assertEqual(saved["channels"]["C_TECH"], state["channels"]["C_TECH"])
        self.assertEqual(saved["channels"]["C_TECH"]["last_run_timestamp"], str(now - 100))
        self.assertNotIn("300.0", saved["channels"]["C_TECH"]["threads"])


class ProductChannelIdsTests(unittest.TestCase):
    def test_only_configured_products_are_returned(self):
        channels = sync_mod.product_channel_ids(
            {"CHANNEL_ID_IDA": "C_IDA", "CHANNEL_ID_IDDM": "  ", "OTHER": "x"}
        )
        self.assertEqual(channels, {"IDA": "C_IDA"})

    def test_no_product_channels_configured(self):
        self.assertEqual(sync_mod.product_channel_ids({}), {})

    def test_sync_env_fills_channel_ids_from_lil_lisa_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = Path(tmp) / "lil-lisa.env"
            sync_file = Path(tmp) / "techsupport_sync.env"
            bot.write_text("CHANNEL_ID_IDA=C_IDA\nCHANNEL_ID_IDDM=C_IDDM\nSLACK_BOT_TOKEN=xoxb-bot\n")
            sync_file.write_text("SLACK_BOT_TOKEN=xoxb-sync\nTECHSUPPORT_CHANNEL_ID=C_TS\n")
            with patch.object(sync_mod, "ENV_PATH", sync_file), patch(
                "paths.bot_env_path", return_value=bot
            ), patch.dict("os.environ", {}, clear=True):
                env = sync_mod._sync_env()
        self.assertEqual(env["CHANNEL_ID_IDA"], "C_IDA")
        self.assertEqual(env["CHANNEL_ID_IDDM"], "C_IDDM")
        self.assertEqual(env["SLACK_BOT_TOKEN"], "xoxb-sync")
        self.assertNotIn("CHANNEL_ID_IDO", env)

    def test_process_environment_overrides_lil_lisa_channel_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            bot = Path(tmp) / "lil-lisa.env"
            sync_file = Path(tmp) / "techsupport_sync.env"
            bot.write_text("CHANNEL_ID_IDA=C_FROM_BOT\n")
            sync_file.write_text("TECHSUPPORT_CHANNEL_ID=C_TS\n")
            with patch.object(sync_mod, "ENV_PATH", sync_file), patch(
                "paths.bot_env_path", return_value=bot
            ), patch.dict("os.environ", {"CHANNEL_ID_IDA": "C_FROM_ENV"}, clear=True):
                env = sync_mod._sync_env()
        self.assertEqual(env["CHANNEL_ID_IDA"], "C_FROM_ENV")


if __name__ == "__main__":
    unittest.main()
