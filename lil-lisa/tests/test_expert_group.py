"""
Unit tests for expert user group resolution (membership, TTL cache, rate-limit
retry, required-group validation, and the fail-loud behaviour when a group
cannot be read). No live Slack calls: the fetcher is injected.

Run from lil-lisa:
    PYTHONPATH=src python3 tests/test_expert_group.py
"""

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from expert_group import (  # noqa: E402
    DEFAULT_CACHE_SECONDS,
    ExpertLookupError,
    ExpertResolver,
    RateLimited,
    cache_seconds_from_env,
    group_id_for_handle,
    group_id_for_handle_across_pages,
    group_ids_from_env,
)


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

    async def __call__(self, usergroup_id):
        self.calls.append(usergroup_id)
        outcome = self.results[min(len(self.calls), len(self.results)) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return list(outcome)


def run(coro):
    return asyncio.run(coro)


def make_resolver(fetcher=None, clock=None, sleeps=None, **kwargs):
    async def record_sleep(seconds):
        if sleeps is not None:
            sleeps.append(seconds)

    kwargs.setdefault("group_ids", {"IDA": "S_IDA", "IDDM": "S_IDDM", "IDO": None})
    kwargs.setdefault("cache_seconds", 300.0)
    return ExpertResolver(
        fetch_members=fetcher,
        time_source=clock or FakeClock(),
        sleep=record_sleep,
        **kwargs,
    )


class MembershipTests(unittest.TestCase):
    def test_group_member_is_expert(self):
        resolver = make_resolver(FakeFetcher([["U1", "U2"]]))
        self.assertTrue(run(resolver.is_expert("U2", "IDA")))
        self.assertEqual(run(resolver.expert_user_ids("IDA")), ["U1", "U2"])

    def test_non_member_is_not_expert(self):
        resolver = make_resolver(FakeFetcher([["U1", "U2"]]))
        self.assertFalse(run(resolver.is_expert("Ustranger", "IDA")))

    def test_membership_is_whitespace_cleaned_and_empties_dropped(self):
        resolver = make_resolver(FakeFetcher([[" U1 ", "", None, "U2"]]))
        self.assertEqual(run(resolver.expert_user_ids("IDA")), ["U1", "U2"])

    def test_unknown_product_and_empty_user_are_not_experts(self):
        resolver = make_resolver(FakeFetcher([["U1"]]))
        self.assertEqual(run(resolver.expert_user_ids(None)), [])
        self.assertFalse(run(resolver.is_expert("U1", None)))
        self.assertFalse(run(resolver.is_expert(None, "IDA")))

    def test_primary_expert_id_is_first_member(self):
        resolver = make_resolver(FakeFetcher([["U1", "U2"]]))
        self.assertEqual(run(resolver.primary_expert_id("IDA")), "U1")


class RequiredGroupTests(unittest.TestCase):
    def test_missing_required_group_raises_naming_the_variable(self):
        with self.assertRaises(ValueError) as ctx:
            ExpertResolver(group_ids={"IDA": "S_IDA", "IDDM": None, "IDO": None})
        self.assertIn("EXPERT_GROUP_ID_IDDM", str(ctx.exception))
        self.assertNotIn("EXPERT_GROUP_ID_IDA", str(ctx.exception))

    def test_every_missing_required_group_is_named(self):
        with self.assertRaises(ValueError) as ctx:
            ExpertResolver(group_ids={})
        self.assertIn("EXPERT_GROUP_ID_IDA", str(ctx.exception))
        self.assertIn("EXPERT_GROUP_ID_IDDM", str(ctx.exception))

    def test_unconfigured_ido_is_allowed_and_nobody_is_an_ido_expert(self):
        fetcher = FakeFetcher([["U1"]])
        resolver = make_resolver(fetcher)
        self.assertEqual(run(resolver.expert_user_ids("IDO")), [])
        self.assertFalse(run(resolver.is_expert("U1", "IDO")))
        self.assertEqual(fetcher.calls, [], "no group configured means no Slack call")


class LookupFailureTests(unittest.TestCase):
    def test_fetch_failure_with_no_cache_raises(self):
        resolver = make_resolver(FakeFetcher([RuntimeError("boom")]))
        with self.assertRaises(ExpertLookupError) as ctx:
            run(resolver.expert_user_ids("IDA"))
        self.assertEqual(ctx.exception.product, "IDA")
        self.assertEqual(ctx.exception.group_id, "S_IDA")

    def test_fetch_failure_keeps_serving_cached_membership(self):
        clock = FakeClock()
        fetcher = FakeFetcher([["U1", "U2"], RuntimeError("slack down")])
        resolver = make_resolver(fetcher, clock=clock)

        self.assertEqual(run(resolver.expert_user_ids("IDA")), ["U1", "U2"])
        clock.advance(301)
        self.assertEqual(run(resolver.expert_user_ids("IDA")), ["U1", "U2"])
        self.assertEqual(len(fetcher.calls), 2)
        self.assertTrue(run(resolver.is_expert("U1", "IDA")))

    def test_failed_lookup_re_stamps_the_cache_instead_of_refetching(self):
        clock = FakeClock()
        fetcher = FakeFetcher([["U1"], RuntimeError("slack down")])
        resolver = make_resolver(fetcher, clock=clock)

        self.assertEqual(run(resolver.expert_user_ids("IDA")), ["U1"])
        clock.advance(301)
        self.assertEqual(run(resolver.expert_user_ids("IDA")), ["U1"])
        self.assertEqual(run(resolver.expert_user_ids("IDA")), ["U1"])
        self.assertEqual(len(fetcher.calls), 2, "the failure must not be retried on every call")

    def test_missing_fetcher_raises(self):
        resolver = make_resolver(None)
        with self.assertRaises(ExpertLookupError):
            run(resolver.expert_user_ids("IDA"))


class CacheTests(unittest.TestCase):
    def test_second_call_within_ttl_does_not_refetch(self):
        clock = FakeClock()
        fetcher = FakeFetcher([["U1"]])
        resolver = make_resolver(fetcher, clock=clock)

        run(resolver.expert_user_ids("IDA"))
        clock.advance(299)
        run(resolver.expert_user_ids("IDA"))
        self.assertEqual(len(fetcher.calls), 1)

    def test_call_after_ttl_refetches(self):
        clock = FakeClock()
        fetcher = FakeFetcher([["U1"], ["U1", "U2"]])
        resolver = make_resolver(fetcher, clock=clock)

        self.assertEqual(run(resolver.expert_user_ids("IDA")), ["U1"])
        clock.advance(301)
        self.assertEqual(run(resolver.expert_user_ids("IDA")), ["U1", "U2"])
        self.assertEqual(len(fetcher.calls), 2)

    def test_products_are_cached_independently(self):
        fetcher = FakeFetcher([["U1"], ["U9"]])
        resolver = make_resolver(fetcher)

        self.assertEqual(run(resolver.expert_user_ids("IDA")), ["U1"])
        self.assertEqual(run(resolver.expert_user_ids("IDDM")), ["U9"])
        self.assertEqual(fetcher.calls, ["S_IDA", "S_IDDM"])


class RateLimitTests(unittest.TestCase):
    def test_rate_limit_is_retried_once_after_retry_after(self):
        sleeps = []
        fetcher = FakeFetcher([RateLimited(7), ["U1", "U2"]])
        resolver = make_resolver(fetcher, sleeps=sleeps)

        self.assertEqual(run(resolver.expert_user_ids("IDA")), ["U1", "U2"])
        self.assertEqual(sleeps, [7.0])
        self.assertEqual(len(fetcher.calls), 2)

    def test_rate_limit_twice_gives_up_and_raises(self):
        sleeps = []
        fetcher = FakeFetcher([RateLimited(2), RateLimited(2)])
        resolver = make_resolver(fetcher, sleeps=sleeps)

        with self.assertRaises(ExpertLookupError):
            run(resolver.expert_user_ids("IDA"))
        self.assertEqual(sleeps, [2.0], "only one retry")
        self.assertEqual(len(fetcher.calls), 2)

    def test_rate_limit_twice_serves_cached_membership_when_there_is_one(self):
        clock = FakeClock()
        sleeps = []
        fetcher = FakeFetcher([["U1"], RateLimited(2), RateLimited(2)])
        resolver = make_resolver(fetcher, clock=clock, sleeps=sleeps)

        self.assertEqual(run(resolver.expert_user_ids("IDA")), ["U1"])
        clock.advance(301)
        self.assertEqual(run(resolver.expert_user_ids("IDA")), ["U1"])
        self.assertEqual(sleeps, [2.0])


class FakeFinder:
    """Injected `usergroups.list` handle lookup stand-in."""

    def __init__(self, ids):
        self.ids = dict(ids)
        self.calls = []

    async def __call__(self, handle):
        self.calls.append(handle)
        return group_id_for_handle([{"id": i, "handle": h} for h, i in self.ids.items()], handle)


class HandleTests(unittest.TestCase):
    def test_handle_is_resolved_to_its_id_before_fetching_members(self):
        fetcher = FakeFetcher([["U1"]])
        finder = FakeFinder({"ida-experts": "S_IDA"})
        resolver = make_resolver(fetcher, group_ids={"IDA": "@ida-experts", "IDDM": "S_IDDM"}, find_group_id=finder)

        self.assertTrue(run(resolver.is_expert("U1", "IDA")))
        self.assertEqual(finder.calls, ["ida-experts"])
        self.assertEqual(fetcher.calls, ["S_IDA"])

    def test_handle_is_looked_up_once_across_refreshes(self):
        clock = FakeClock()
        fetcher = FakeFetcher([["U1"], ["U2"]])
        finder = FakeFinder({"ida-experts": "S_IDA"})
        resolver = make_resolver(
            fetcher, clock=clock, group_ids={"IDA": "@ida-experts", "IDDM": "S_IDDM"}, find_group_id=finder
        )

        run(resolver.expert_user_ids("IDA"))
        clock.advance(301)
        self.assertEqual(run(resolver.expert_user_ids("IDA")), ["U2"])
        self.assertEqual(finder.calls, ["ida-experts"])

    def test_id_values_never_call_the_finder(self):
        finder = FakeFinder({})
        resolver = make_resolver(FakeFetcher([["U1"]]), find_group_id=finder)

        run(resolver.expert_user_ids("IDA"))
        self.assertEqual(finder.calls, [])

    def test_unknown_handle_fails_loudly(self):
        fetcher = FakeFetcher([["U1"]])
        resolver = make_resolver(
            fetcher, group_ids={"IDA": "@nope", "IDDM": "S_IDDM"}, find_group_id=FakeFinder({"ida-experts": "S_IDA"})
        )

        with self.assertRaises(ExpertLookupError) as ctx:
            run(resolver.expert_user_ids("IDA"))
        self.assertEqual(ctx.exception.group_id, "@nope")
        self.assertIn("@nope", str(ctx.exception))
        self.assertEqual(fetcher.calls, [])

    def test_handle_without_a_finder_fails_loudly(self):
        resolver = make_resolver(FakeFetcher([["U1"]]), group_ids={"IDA": "@ida-experts", "IDDM": "S_IDDM"})
        with self.assertRaises(ExpertLookupError):
            run(resolver.expert_user_ids("IDA"))

    def test_handle_match_is_case_insensitive(self):
        groups = [{"id": "S1", "handle": "Other"}, {"id": "S2", "handle": "RockNBot-IDA-Experts"}]
        self.assertEqual(group_id_for_handle(groups, "rocknbot-ida-experts"), "S2")

    def test_handle_lookup_follows_cursor_until_the_handle_is_found(self):
        pages = [
            {
                "usergroups": [{"id": "S1", "handle": "other"}],
                "response_metadata": {"next_cursor": "page-2"},
            },
            {
                "usergroups": [{"id": "S2", "handle": "RockNBot-IDA-Experts"}],
                "response_metadata": {"next_cursor": "page-3"},
            },
            {"usergroups": [{"id": "S3", "handle": "later"}]},
        ]
        cursors = []

        async def fetch_page(cursor):
            cursors.append(cursor)
            return pages[len(cursors) - 1]

        self.assertEqual(run(group_id_for_handle_across_pages(fetch_page, "rocknbot-ida-experts")), "S2")
        self.assertEqual(cursors, ["", "page-2"])

    def test_handle_lookup_fails_after_the_last_page(self):
        pages = [
            {
                "usergroups": [{"id": "S1", "handle": "other"}],
                "response_metadata": {"next_cursor": "page-2"},
            },
            {"usergroups": [{"id": "S2", "handle": "later"}]},
        ]

        async def fetch_page(cursor):
            return pages[0] if cursor == "" else pages[1]

        with self.assertRaises(RuntimeError) as ctx:
            run(group_id_for_handle_across_pages(fetch_page, "missing"))
        self.assertIn("@missing", str(ctx.exception))

    def test_handle_lookup_stops_when_a_cursor_repeats(self):
        calls = []

        async def fetch_page(cursor):
            calls.append(cursor)
            return {
                "usergroups": [{"id": "S1", "handle": "other"}],
                "response_metadata": {"next_cursor": "again"},
            }

        with self.assertRaises(RuntimeError):
            run(group_id_for_handle_across_pages(fetch_page, "missing"))
        self.assertEqual(calls, ["", "again"])


class ConfigTests(unittest.TestCase):
    def test_env_readers(self):
        env = {
            "EXPERT_GROUP_ID_IDA": " S_IDA ",
            "EXPERT_GROUP_ID_IDDM": "S_IDDM",
            "EXPERT_GROUP_ID_IDO": "",
            "EXPERT_GROUP_CACHE_SECONDS": "60",
        }
        self.assertEqual(group_ids_from_env(env), {"IDA": "S_IDA", "IDDM": "S_IDDM", "IDO": None})
        self.assertEqual(cache_seconds_from_env(env), 60.0)

    def test_cache_seconds_defaults_when_unset_or_bad(self):
        self.assertEqual(cache_seconds_from_env({}), DEFAULT_CACHE_SECONDS)
        self.assertEqual(cache_seconds_from_env({"EXPERT_GROUP_CACHE_SECONDS": "soon"}), DEFAULT_CACHE_SECONDS)

    def test_is_configured_and_unconfigured_products(self):
        resolver = make_resolver(None)
        self.assertTrue(resolver.is_configured("IDA"))
        self.assertTrue(resolver.is_configured("IDDM"))
        self.assertFalse(resolver.is_configured("IDO"))
        self.assertEqual(resolver.unconfigured_products(("IDA", "IDDM", "IDO")), ["IDO"])


if __name__ == "__main__":
    unittest.main()
