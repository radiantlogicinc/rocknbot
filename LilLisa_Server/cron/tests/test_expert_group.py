"""
Unit tests for cron-side expert user group resolution (membership, TTL cache,
rate-limit retry, required-group validation, and the fail-loud behaviour when a
group cannot be read). No live Slack calls: the fetcher is injected, and the
requests-based default fetcher is exercised against a mocked requests.get.

Run from LilLisa_Server/cron:
    PYTHONPATH=. python3 tests/test_expert_group.py
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import expert_group as eg  # noqa: E402


class FakeClock:
    """Monotonic clock the tests advance by hand."""

    def __init__(self, now=1000.0):
        self.now = float(now)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


class FakeFetcher:
    """Injected `usergroups.users.list` stand-in.

    `results` is a list of per-call outcomes: a list of user ids to return, or
    an exception instance to raise.
    """

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, usergroup_id):
        self.calls.append(usergroup_id)
        outcome = self.results[min(len(self.calls), len(self.results)) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return list(outcome)


class FakeResponse:
    def __init__(self, payload, headers=None):
        self.payload = payload
        self.headers = headers or {}

    def json(self):
        return self.payload


def make_resolver(fetcher=None, clock=None, sleeps=None, **kwargs):
    kwargs.setdefault("group_ids", {"IDA": "S_IDA", "IDDM": "S_IDDM", "IDO": None})
    kwargs.setdefault("cache_seconds", 300.0)
    return eg.ExpertResolver(
        fetch_members=fetcher,
        time_source=clock or FakeClock(),
        sleep=(sleeps.append if sleeps is not None else (lambda _seconds: None)),
        **kwargs,
    )


class MembershipTests(unittest.TestCase):
    def test_group_member_is_expert(self):
        resolver = make_resolver(FakeFetcher([["U1", "U2"]]))
        self.assertTrue(resolver.is_expert("U2", "IDA"))
        self.assertEqual(resolver.expert_user_ids("IDA"), ["U1", "U2"])

    def test_non_member_is_not_expert(self):
        resolver = make_resolver(FakeFetcher([["U1", "U2"]]))
        self.assertFalse(resolver.is_expert("Ustranger", "IDA"))

    def test_membership_is_whitespace_cleaned_and_empties_dropped(self):
        resolver = make_resolver(FakeFetcher([[" U1 ", "", None, "U2"]]))
        self.assertEqual(resolver.expert_user_ids("IDA"), ["U1", "U2"])

    def test_unknown_product_and_empty_user_are_not_experts(self):
        resolver = make_resolver(FakeFetcher([["U1"]]))
        self.assertEqual(resolver.expert_user_ids(None), [])
        self.assertFalse(resolver.is_expert("U1", None))
        self.assertFalse(resolver.is_expert(None, "IDA"))

    def test_primary_expert_id_is_first_member(self):
        resolver = make_resolver(FakeFetcher([["U1", "U2"]]))
        self.assertEqual(resolver.primary_expert_id("IDA"), "U1")


class RequiredGroupTests(unittest.TestCase):
    def test_missing_required_group_raises_naming_the_variable(self):
        with self.assertRaises(ValueError) as ctx:
            eg.ExpertResolver(group_ids={"IDA": "S_IDA", "IDDM": None, "IDO": None})
        self.assertIn("EXPERT_GROUP_ID_IDDM", str(ctx.exception))
        self.assertNotIn("EXPERT_GROUP_ID_IDA", str(ctx.exception))

    def test_from_env_raises_when_a_required_group_is_missing(self):
        with self.assertRaises(ValueError) as ctx:
            eg.ExpertResolver.from_env({"EXPERT_GROUP_ID_IDA": "S_IDA"}, fetch_members=FakeFetcher([["U1"]]))
        self.assertIn("EXPERT_GROUP_ID_IDDM", str(ctx.exception))

    def test_unconfigured_ido_is_allowed_and_nobody_is_an_ido_expert(self):
        fetcher = FakeFetcher([["U1"]])
        resolver = make_resolver(fetcher)
        self.assertEqual(resolver.expert_user_ids("IDO"), [])
        self.assertFalse(resolver.is_expert("U1", "IDO"))
        self.assertEqual(fetcher.calls, [], "no group configured means no Slack call")


class LookupFailureTests(unittest.TestCase):
    def test_fetch_failure_with_no_cache_raises(self):
        resolver = make_resolver(FakeFetcher([RuntimeError("boom")]))
        with self.assertRaises(eg.ExpertLookupError) as ctx:
            resolver.expert_user_ids("IDA")
        self.assertEqual(ctx.exception.product, "IDA")
        self.assertEqual(ctx.exception.group_id, "S_IDA")

    def test_fetch_failure_keeps_serving_cached_membership(self):
        clock = FakeClock()
        fetcher = FakeFetcher([["U1", "U2"], RuntimeError("slack down")])
        resolver = make_resolver(fetcher, clock=clock)

        self.assertEqual(resolver.expert_user_ids("IDA"), ["U1", "U2"])
        clock.advance(301)
        self.assertEqual(resolver.expert_user_ids("IDA"), ["U1", "U2"])
        self.assertEqual(len(fetcher.calls), 2)
        self.assertTrue(resolver.is_expert("U1", "IDA"))

    def test_failed_lookup_re_stamps_the_cache_instead_of_refetching(self):
        clock = FakeClock()
        fetcher = FakeFetcher([["U1"], RuntimeError("slack down")])
        resolver = make_resolver(fetcher, clock=clock)

        self.assertEqual(resolver.expert_user_ids("IDA"), ["U1"])
        clock.advance(301)
        self.assertEqual(resolver.expert_user_ids("IDA"), ["U1"])
        self.assertEqual(resolver.expert_user_ids("IDA"), ["U1"])
        self.assertEqual(len(fetcher.calls), 2, "the failure must not be retried on every call")

    def test_missing_fetcher_raises(self):
        resolver = make_resolver(None)
        with self.assertRaises(eg.ExpertLookupError):
            resolver.expert_user_ids("IDA")


class CacheTests(unittest.TestCase):
    def test_second_call_within_ttl_does_not_refetch(self):
        clock = FakeClock()
        fetcher = FakeFetcher([["U1"]])
        resolver = make_resolver(fetcher, clock=clock)

        resolver.expert_user_ids("IDA")
        clock.advance(299)
        resolver.expert_user_ids("IDA")
        self.assertEqual(len(fetcher.calls), 1)

    def test_call_after_ttl_refetches(self):
        clock = FakeClock()
        fetcher = FakeFetcher([["U1"], ["U1", "U2"]])
        resolver = make_resolver(fetcher, clock=clock)

        self.assertEqual(resolver.expert_user_ids("IDA"), ["U1"])
        clock.advance(301)
        self.assertEqual(resolver.expert_user_ids("IDA"), ["U1", "U2"])
        self.assertEqual(len(fetcher.calls), 2)

    def test_products_are_cached_independently(self):
        fetcher = FakeFetcher([["U1"], ["U9"]])
        resolver = make_resolver(fetcher)

        self.assertEqual(resolver.expert_user_ids("IDA"), ["U1"])
        self.assertEqual(resolver.expert_user_ids("IDDM"), ["U9"])
        self.assertEqual(fetcher.calls, ["S_IDA", "S_IDDM"])


class RateLimitTests(unittest.TestCase):
    def test_rate_limit_is_retried_once_after_retry_after(self):
        sleeps = []
        fetcher = FakeFetcher([eg.RateLimited(7), ["U1", "U2"]])
        resolver = make_resolver(fetcher, sleeps=sleeps)

        self.assertEqual(resolver.expert_user_ids("IDA"), ["U1", "U2"])
        self.assertEqual(sleeps, [7.0])
        self.assertEqual(len(fetcher.calls), 2)

    def test_rate_limit_twice_gives_up_and_raises(self):
        sleeps = []
        fetcher = FakeFetcher([eg.RateLimited(2), eg.RateLimited(2)])
        resolver = make_resolver(fetcher, sleeps=sleeps)

        with self.assertRaises(eg.ExpertLookupError):
            resolver.expert_user_ids("IDA")
        self.assertEqual(sleeps, [2.0], "only one retry")
        self.assertEqual(len(fetcher.calls), 2)

    def test_rate_limit_twice_serves_cached_membership_when_there_is_one(self):
        clock = FakeClock()
        sleeps = []
        fetcher = FakeFetcher([["U1"], eg.RateLimited(2), eg.RateLimited(2)])
        resolver = make_resolver(fetcher, clock=clock, sleeps=sleeps)

        self.assertEqual(resolver.expert_user_ids("IDA"), ["U1"])
        clock.advance(301)
        self.assertEqual(resolver.expert_user_ids("IDA"), ["U1"])
        self.assertEqual(sleeps, [2.0])


class FakeFinder:
    """Injected `usergroups.list` handle lookup stand-in."""

    def __init__(self, ids):
        self.ids = dict(ids)
        self.calls = []

    def __call__(self, handle):
        self.calls.append(handle)
        return eg.group_id_for_handle([{"id": i, "handle": h} for h, i in self.ids.items()], handle)


class HandleTests(unittest.TestCase):
    def test_handle_is_resolved_to_its_id_before_fetching_members(self):
        fetcher = FakeFetcher([["U1"]])
        finder = FakeFinder({"ida-experts": "S_IDA"})
        resolver = make_resolver(fetcher, group_ids={"IDA": "@ida-experts", "IDDM": "S_IDDM"}, find_group_id=finder)

        self.assertTrue(resolver.is_expert("U1", "IDA"))
        self.assertEqual(finder.calls, ["ida-experts"])
        self.assertEqual(fetcher.calls, ["S_IDA"])

    def test_handle_is_looked_up_once_across_refreshes(self):
        clock = FakeClock()
        fetcher = FakeFetcher([["U1"], ["U2"]])
        finder = FakeFinder({"ida-experts": "S_IDA"})
        resolver = make_resolver(
            fetcher, clock=clock, group_ids={"IDA": "@ida-experts", "IDDM": "S_IDDM"}, find_group_id=finder
        )

        resolver.expert_user_ids("IDA")
        clock.advance(301)
        self.assertEqual(resolver.expert_user_ids("IDA"), ["U2"])
        self.assertEqual(finder.calls, ["ida-experts"])

    def test_id_values_never_call_the_finder(self):
        finder = FakeFinder({})
        resolver = make_resolver(FakeFetcher([["U1"]]), find_group_id=finder)

        resolver.expert_user_ids("IDA")
        self.assertEqual(finder.calls, [])

    def test_unknown_handle_fails_loudly(self):
        fetcher = FakeFetcher([["U1"]])
        resolver = make_resolver(
            fetcher, group_ids={"IDA": "@nope", "IDDM": "S_IDDM"}, find_group_id=FakeFinder({"ida-experts": "S_IDA"})
        )

        with self.assertRaises(eg.ExpertLookupError) as ctx:
            resolver.expert_user_ids("IDA")
        self.assertEqual(ctx.exception.group_id, "@nope")
        self.assertIn("@nope", str(ctx.exception))
        self.assertEqual(fetcher.calls, [])

    def test_handle_without_a_finder_fails_loudly(self):
        resolver = make_resolver(FakeFetcher([["U1"]]), group_ids={"IDA": "@ida-experts", "IDDM": "S_IDDM"})
        with self.assertRaises(eg.ExpertLookupError):
            resolver.expert_user_ids("IDA")

    def test_handle_match_is_case_insensitive(self):
        groups = [{"id": "S1", "handle": "Other"}, {"id": "S2", "handle": "RockNBot-IDA-Experts"}]
        self.assertEqual(eg.group_id_for_handle(groups, "rocknbot-ida-experts"), "S2")

    def test_slack_group_finder(self):
        find = eg.make_slack_group_finder("xoxb-test")
        payload = {"ok": True, "usergroups": [{"id": "S_IDA", "handle": "ida-experts"}]}
        with patch.object(eg.requests, "get", return_value=FakeResponse(payload)) as get:
            self.assertEqual(find("ida-experts"), "S_IDA")
        self.assertTrue(get.call_args.args[0].endswith("/usergroups.list"))
        self.assertEqual(get.call_args.kwargs["headers"], {"Authorization": "Bearer xoxb-test"})

    def test_slack_group_finder_errors(self):
        find = eg.make_slack_group_finder("xoxb-test")
        limited = FakeResponse({"ok": False, "error": "ratelimited"}, headers={"Retry-After": "4"})
        with patch.object(eg.requests, "get", return_value=limited):
            with self.assertRaises(eg.RateLimited):
                find("ida-experts")
        with patch.object(eg.requests, "get", return_value=FakeResponse({"ok": False, "error": "missing_scope"})):
            with self.assertRaises(RuntimeError):
                find("ida-experts")

    def test_slack_group_finder_follows_cursor_until_the_handle_is_found(self):
        find = eg.make_slack_group_finder("xoxb-test")
        pages = [
            FakeResponse({
                "ok": True,
                "usergroups": [{"id": "S_OTHER", "handle": "other"}],
                "response_metadata": {"next_cursor": "page-2"},
            }),
            FakeResponse({
                "ok": True,
                "usergroups": [{"id": "S_IDA", "handle": "IDA-Experts"}],
                "response_metadata": {"next_cursor": "page-3"},
            }),
            FakeResponse({"ok": True, "usergroups": [{"id": "S_LATER", "handle": "later"}]}),
        ]
        with patch.object(eg.requests, "get", side_effect=pages) as get:
            self.assertEqual(find("ida-experts"), "S_IDA")
        self.assertEqual(get.call_count, 2, "stops once the handle is found")
        self.assertIsNone(get.call_args_list[0].kwargs["params"])
        self.assertEqual(get.call_args_list[1].kwargs["params"], {"cursor": "page-2"})

    def test_slack_group_finder_unknown_handle_after_the_last_page(self):
        find = eg.make_slack_group_finder("xoxb-test")
        pages = [
            FakeResponse({
                "ok": True,
                "usergroups": [{"id": "S_OTHER", "handle": "other"}],
                "response_metadata": {"next_cursor": "page-2"},
            }),
            FakeResponse({"ok": True, "usergroups": [{"id": "S_LATER", "handle": "later"}]}),
        ]
        with patch.object(eg.requests, "get", side_effect=pages) as get:
            with self.assertRaises(RuntimeError) as ctx:
                find("ida-experts")
        self.assertEqual(get.call_count, 2)
        self.assertIn("@ida-experts", str(ctx.exception))

    def test_slack_group_finder_stops_when_a_cursor_repeats(self):
        find = eg.make_slack_group_finder("xoxb-test")
        page = FakeResponse({
            "ok": True,
            "usergroups": [{"id": "S_OTHER", "handle": "other"}],
            "response_metadata": {"next_cursor": "again"},
        })
        with patch.object(eg.requests, "get", return_value=page) as get:
            with self.assertRaises(RuntimeError):
                find("ida-experts")
        # First page, then the repeated cursor once, then stop.
        self.assertEqual(get.call_count, 2)

    def test_slack_group_finder_rate_limit_on_a_later_page(self):
        find = eg.make_slack_group_finder("xoxb-test")
        pages = [
            FakeResponse({
                "ok": True,
                "usergroups": [{"id": "S_OTHER", "handle": "other"}],
                "response_metadata": {"next_cursor": "page-2"},
            }),
            FakeResponse({"ok": False, "error": "ratelimited"}, headers={"Retry-After": "4"}),
        ]
        with patch.object(eg.requests, "get", side_effect=pages):
            with self.assertRaises(eg.RateLimited) as ctx:
                find("ida-experts")
        self.assertEqual(ctx.exception.retry_after, 4.0)

    def test_from_env_wires_the_slack_group_finder(self):
        payload = {"ok": True, "usergroups": [{"id": "S_IDA", "handle": "ida-experts"}]}
        resolver = eg.ExpertResolver.from_env(
            {"EXPERT_GROUP_ID_IDA": "@ida-experts", "EXPERT_GROUP_ID_IDDM": "S_IDDM", "SLACK_BOT_TOKEN": "xoxb-test"},
            fetch_members=FakeFetcher([["U1"]]),
        )
        with patch.object(eg.requests, "get", return_value=FakeResponse(payload)):
            self.assertTrue(resolver.is_expert("U1", "IDA"))


class SlackFetcherTests(unittest.TestCase):
    def test_ok_response_returns_users(self):
        fetch = eg.make_slack_fetcher("xoxb-test")
        with patch.object(eg.requests, "get", return_value=FakeResponse({"ok": True, "users": ["U1", "U2"]})) as get:
            self.assertEqual(fetch("S_IDA"), ["U1", "U2"])
        self.assertEqual(get.call_args.kwargs["params"], {"usergroup": "S_IDA"})
        self.assertEqual(get.call_args.kwargs["headers"], {"Authorization": "Bearer xoxb-test"})

    def test_ratelimited_raises_rate_limited_with_retry_after(self):
        fetch = eg.make_slack_fetcher("xoxb-test")
        response = FakeResponse({"ok": False, "error": "ratelimited"}, headers={"Retry-After": "3"})
        with patch.object(eg.requests, "get", return_value=response):
            with self.assertRaises(eg.RateLimited) as ctx:
                fetch("S_IDA")
        self.assertEqual(ctx.exception.retry_after, 3.0)

    def test_other_slack_error_raises_runtime_error(self):
        fetch = eg.make_slack_fetcher("xoxb-test")
        with patch.object(eg.requests, "get", return_value=FakeResponse({"ok": False, "error": "missing_scope"})):
            with self.assertRaises(RuntimeError):
                fetch("S_IDA")

    def test_resolver_surfaces_slack_errors(self):
        resolver = make_resolver(eg.make_slack_fetcher("xoxb-test"))
        with patch.object(eg.requests, "get", return_value=FakeResponse({"ok": False, "error": "missing_scope"})):
            with self.assertRaises(eg.ExpertLookupError):
                resolver.expert_user_ids("IDA")


class ConfigTests(unittest.TestCase):
    def test_env_readers(self):
        env = {
            "EXPERT_GROUP_ID_IDA": " S_IDA ",
            "EXPERT_GROUP_ID_IDDM": "S_IDDM",
            "EXPERT_GROUP_ID_IDO": "",
            "EXPERT_GROUP_CACHE_SECONDS": "60",
        }
        self.assertEqual(eg.group_ids_from_env(env), {"IDA": "S_IDA", "IDDM": "S_IDDM", "IDO": None})
        self.assertEqual(eg.cache_seconds_from_env(env), 60.0)

    def test_cache_seconds_defaults_when_unset_or_bad(self):
        self.assertEqual(eg.cache_seconds_from_env({}), eg.DEFAULT_CACHE_SECONDS)
        self.assertEqual(eg.cache_seconds_from_env({"EXPERT_GROUP_CACHE_SECONDS": "soon"}), eg.DEFAULT_CACHE_SECONDS)

    def test_from_env_builds_configured_resolver(self):
        fetcher = FakeFetcher([["U1"], ["U2"]])
        resolver = eg.ExpertResolver.from_env(
            {
                "EXPERT_GROUP_ID_IDA": "S_IDA",
                "EXPERT_GROUP_ID_IDDM": "S_IDDM",
                "EXPERT_GROUP_CACHE_SECONDS": "60",
            },
            fetch_members=fetcher,
        )
        self.assertEqual(resolver.cache_seconds, 60.0)
        self.assertTrue(resolver.is_expert("U1", "IDA"))
        self.assertEqual(resolver.unconfigured_products(("IDA", "IDDM", "IDO")), ["IDO"])

    def test_module_level_helpers_use_the_default_resolver(self):
        resolver = make_resolver(FakeFetcher([["U1"]]))
        eg.set_default_resolver(resolver)
        try:
            self.assertEqual(eg.expert_user_ids("IDA"), ["U1"])
            self.assertTrue(eg.is_expert("U1", "IDA"))
            self.assertFalse(eg.is_expert("U2", "IDA"))
        finally:
            eg.set_default_resolver(None)


if __name__ == "__main__":
    unittest.main()
