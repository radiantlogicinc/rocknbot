"""The bot's @handle lookup walks every usergroups.list page.

Run from lil-lisa:
    PYTHONPATH=src .venv/bin/python -m unittest tests.test_usergroup_handle_lookup
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from slack_sdk.errors import SlackApiError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

os.environ.update(
    {
        "LIL_LISA_SLACK_USERID": "Ubot",
        "SLACK_BOT_TOKEN": "xoxb-test",
        "SLACK_APP_TOKEN": "xapp-test",
        "CHANNEL_ID_IDDM": "Ciddm",
        "ADMIN_CHANNEL_ID_IDDM": "Cadmin-iddm",
        "CHANNEL_ID_IDA": "Cida",
        "ADMIN_CHANNEL_ID_IDA": "Cadmin-ida",
        "AUTHENTICATION_KEY": "test-authentication-key",
        "MAX_LENGTH": "4000",
        "LIL_LISA_SERVER_URL": "http://server.test",
        "EXPERT_GROUP_ID_IDA": "Sida",
        "EXPERT_GROUP_ID_IDDM": "Siddm",
    }
)

import slack  # noqa: E402
from expert_group import RateLimited  # noqa: E402


class _SlackErrorResponse(dict):
    def __init__(self, payload, headers=None):
        super().__init__(payload)
        self.headers = headers or {}


class UsergroupHandleLookupTests(unittest.IsolatedAsyncioTestCase):
    async def test_find_usergroup_id_follows_the_cursor(self):
        pages = [
            {
                "usergroups": [{"id": "S1", "handle": "other"}],
                "response_metadata": {"next_cursor": "page-2"},
            },
            {
                "usergroups": [{"id": "S_IDA", "handle": "ida-experts"}],
                "response_metadata": {"next_cursor": "page-3"},
            },
        ]
        list_groups = AsyncMock(side_effect=pages)
        with patch.object(slack.app.client, "usergroups_list", new=list_groups):
            self.assertEqual(await slack._find_usergroup_id("ida-experts"), "S_IDA")
        self.assertEqual(list_groups.await_count, 2)
        self.assertEqual(list_groups.await_args_list[0].kwargs, {})
        self.assertEqual(list_groups.await_args_list[1].kwargs, {"cursor": "page-2"})

    async def test_find_usergroup_id_unknown_after_the_last_page(self):
        pages = [
            {
                "usergroups": [{"id": "S1", "handle": "other"}],
                "response_metadata": {"next_cursor": "page-2"},
            },
            {"usergroups": [{"id": "S2", "handle": "later"}]},
        ]
        list_groups = AsyncMock(side_effect=pages)
        with patch.object(slack.app.client, "usergroups_list", new=list_groups):
            with self.assertRaises(RuntimeError) as ctx:
                await slack._find_usergroup_id("missing")
        self.assertIn("@missing", str(ctx.exception))
        self.assertEqual(list_groups.await_count, 2)

    async def test_find_usergroup_id_rate_limit_on_a_later_page(self):
        pages = [
            {
                "usergroups": [{"id": "S1", "handle": "other"}],
                "response_metadata": {"next_cursor": "page-2"},
            },
            SlackApiError(
                "ratelimited",
                _SlackErrorResponse({"ok": False, "error": "ratelimited"}, {"Retry-After": "4"}),
            ),
        ]
        list_groups = AsyncMock(side_effect=pages)
        with patch.object(slack.app.client, "usergroups_list", new=list_groups):
            with self.assertRaises(RateLimited) as ctx:
                await slack._find_usergroup_id("ida-experts")
        self.assertEqual(ctx.exception.retry_after, 4.0)


if __name__ == "__main__":
    unittest.main()
