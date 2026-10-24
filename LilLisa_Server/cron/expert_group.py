"""
expert_group.py
====================================
Resolves "who is an expert for product X" from a Slack user group. The user
group is the only source of expert identity: there is no single-ID setting and
no silent degradation when the group cannot be read.

This is the cron-side twin of lil-lisa/src/expert_group.py. LilLisa_Server and
lil-lisa deploy independently and cannot import each other, so the contract is
duplicated rather than shared -- keep the two aligned. The only differences are
mechanical: this one is synchronous and its default fetcher uses `requests`
(the bot's uses the Bolt AsyncApp client).

Configuration (LilLisa_Server/cron/env/techsupport_sync.env, overridden by the
process environment -- the same load_env pattern as nightly_techsupport_sync.py):
    EXPERT_GROUP_ID_IDA / _IDDM          Slack user group ID (e.g. S0123ABCD) or
                                         @handle (e.g. @ida-experts).
                                         REQUIRED: the resolver raises if unset.
    EXPERT_GROUP_ID_IDO                  Optional -- IDO is only deployed for
                                         some workspaces. With no group, nobody
                                         is an IDO expert.
    EXPERT_GROUP_CACHE_SECONDS           Membership cache TTL (default 300).
    SLACK_BOT_TOKEN                      Needs the `usergroups:read` scope.

Shared rules (identical in both copies):
  1. Membership comes from Slack `usergroups.users.list?usergroup=<id>` and is
     cached per product for EXPERT_GROUP_CACHE_SECONDS (default 300s) so hot
     paths (per-message scans) do not hit Slack.
  2. On `ratelimited`, the fetch is retried once after Retry-After seconds.
  3. A required product (IDA, IDDM) with no group id is a configuration error:
     the constructor raises ValueError naming the missing variable.
  4. On a failed lookup, the last successfully cached membership keeps being
     served (its expiry is re-stamped and a warning is logged) so a brief Slack
     outage is survivable. With nothing cached, ExpertLookupError is raised so
     the failure is loud instead of silently making everyone a non-expert.
  5. An unconfigured optional product resolves to [] without any Slack call.
  6. A value starting with "@" is a user group handle. It is resolved to its ID
     once via Slack `usergroups.list` (same `usergroups:read` scope), following
     `response_metadata.next_cursor` until the handle is found or the last page
     is reached. The ID is kept for the life of the process. An unknown handle
     is a failed lookup under rule 4. Any other value is used as the ID unchanged.

Usage:
    from expert_group import is_expert, expert_user_ids
    if is_expert(message["user"], "IDA"): ...
Tests inject their own fetcher:
    ExpertResolver(group_ids={...}, fetch_members=fake)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import requests
from dotenv import dotenv_values
from paths import PACKAGE_ROOT

ENV_PATH = PACKAGE_ROOT / "env" / "techsupport_sync.env"
SLACK_API_BASE = "https://slack.com/api"

PRODUCTS: Sequence[str] = ("IDA", "IDDM", "IDO")
# Products the pipeline cannot sensibly run without an expert group for.
REQUIRED_PRODUCTS: Sequence[str] = ("IDA", "IDDM")
DEFAULT_CACHE_SECONDS = 300.0
SLACK_TIMEOUT_SECONDS = 30

logger = logging.getLogger(__name__)

# The injected Slack call: usergroup id -> member user ids.
FetchMembers = Callable[[str], List[str]]
# The injected Slack call: usergroup handle (without "@") -> usergroup id.
FindGroupId = Callable[[str], str]

HANDLE_PREFIX = "@"


class RateLimited(Exception):
    """Raised by a fetcher when Slack answers `ratelimited`.

    `retry_after` is the Retry-After header value in seconds; the resolver
    sleeps that long and retries the fetch exactly once.
    """

    def __init__(self, retry_after: float = 1.0):
        super().__init__(f"Slack rate limited - retry after {retry_after}s")
        self.retry_after = float(retry_after)


class ExpertLookupError(RuntimeError):
    """Expert membership could not be resolved and nothing was cached.

    Raised into the caller on purpose: with no membership there is no safe
    answer to "is this user an expert", and guessing "no" would silently make
    the product-channel scan find no expert corrections at all.
    """

    def __init__(self, product: str, group_id: Optional[str], cause: BaseException):
        super().__init__(
            f"Could not resolve expert user group {group_id} for {product} and no cached "
            f"membership is available: {cause}"
        )
        self.product = product
        self.group_id = group_id
        self.cause = cause


class _Entry:
    """One product's cached membership (only successful fetches are cached)."""

    __slots__ = ("members", "expires_at")

    def __init__(self, members: List[str], expires_at: float):
        self.members = members
        self.expires_at = expires_at


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def load_env() -> Dict[str, str]:
    """Env-file values overridden by the process environment (same as
    nightly_techsupport_sync.load_env, minus its required-var check: the
    resolver does its own required-group check in __init__)."""
    env: Dict[str, str] = {k: v for k, v in dotenv_values(str(ENV_PATH)).items() if v is not None}
    env.update(os.environ)
    return env


def group_ids_from_env(env: Mapping[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """{"IDA": env["EXPERT_GROUP_ID_IDA"], ...} -- IDO's entry may be None."""
    return {product: _clean(env.get(f"EXPERT_GROUP_ID_{product}")) for product in PRODUCTS}


def cache_seconds_from_env(env: Mapping[str, Optional[str]]) -> float:
    """EXPERT_GROUP_CACHE_SECONDS, defaulting to 300s if unset or unparseable."""
    raw = _clean(env.get("EXPERT_GROUP_CACHE_SECONDS"))
    if not raw:
        return DEFAULT_CACHE_SECONDS
    try:
        return float(raw)
    except ValueError:
        logger.warning("EXPERT_GROUP_CACHE_SECONDS=%r is not a number - using %s", raw, DEFAULT_CACHE_SECONDS)
        return DEFAULT_CACHE_SECONDS


def make_slack_fetcher(token: str) -> FetchMembers:
    """Default fetcher: Slack `usergroups.users.list` over `requests`.

    Raises RateLimited on `ratelimited` so the resolver owns the retry policy;
    any other Slack error becomes a RuntimeError, which the resolver turns into
    either the last cached membership or an ExpertLookupError.
    """

    def fetch(usergroup_id: str) -> List[str]:
        resp = requests.get(
            f"{SLACK_API_BASE}/usergroups.users.list",
            headers={"Authorization": f"Bearer {token}"},
            params={"usergroup": usergroup_id},
            timeout=SLACK_TIMEOUT_SECONDS,
        )
        data = resp.json()
        if data.get("ok"):
            return list(data.get("users") or [])
        if data.get("error") == "ratelimited":
            raise RateLimited(float(resp.headers.get("Retry-After", "1")))
        raise RuntimeError(f"Slack API error calling usergroups.users.list: {data.get('error')}")

    return fetch


def make_slack_group_finder(token: str) -> FindGroupId:
    """Default handle lookup: Slack `usergroups.list` over `requests`.

    Follows `response_metadata.next_cursor` until the handle is found or Slack
    has no further page. Handles match case-insensitively. Raises RateLimited
    on `ratelimited`, and RuntimeError for any other Slack error or when no
    group has the handle.
    """

    def fetch_page(cursor: str) -> Mapping[str, object]:
        params = {"cursor": cursor} if cursor else None
        resp = requests.get(
            f"{SLACK_API_BASE}/usergroups.list",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=SLACK_TIMEOUT_SECONDS,
        )
        data = resp.json()
        if not data.get("ok"):
            if data.get("error") == "ratelimited":
                raise RateLimited(float(resp.headers.get("Retry-After", "1")))
            raise RuntimeError(f"Slack API error calling usergroups.list: {data.get('error')}")
        return data

    def find(handle: str) -> str:
        return group_id_for_handle_across_pages(fetch_page, handle)

    return find


def group_id_for_handle(usergroups: Iterable[Mapping[str, object]], handle: str) -> str:
    """The id of the group in one `usergroups.list` page whose handle is `handle`."""
    found = _group_id_in_page(usergroups, handle)
    if found:
        return found
    raise RuntimeError(f"No Slack user group has the handle {HANDLE_PREFIX}{handle}")


def usergroups_list_next_cursor(payload: Mapping[str, object]) -> str:
    """The next `usergroups.list` page cursor, or "" when this page is the last."""
    metadata = payload.get("response_metadata")
    if not isinstance(metadata, Mapping):
        return ""
    return str(metadata.get("next_cursor") or "").strip()


def group_id_for_handle_across_pages(
    fetch_page: Callable[[str], Mapping[str, object]],
    handle: str,
) -> str:
    """Resolve `handle` by walking `usergroups.list` pages.

    `fetch_page` takes the cursor ("" for the first page) and returns one
    successful page payload. Stops when the handle is found, when the next
    cursor is empty, or when Slack repeats a cursor.
    """
    cursor = ""
    seen = set()
    while cursor not in seen:
        seen.add(cursor)
        payload = fetch_page(cursor)
        found = _group_id_in_page(payload.get("usergroups") or [], handle)
        if found:
            return found
        cursor = usergroups_list_next_cursor(payload)
        if not cursor:
            break
    raise RuntimeError(f"No Slack user group has the handle {HANDLE_PREFIX}{handle}")


def _group_id_in_page(usergroups: Iterable[Mapping[str, object]], handle: str) -> Optional[str]:
    """The id of the group on this page whose handle is `handle`, else None."""
    wanted = handle.lower()
    for group in usergroups:
        if str(group.get("handle") or "").lower() == wanted and group.get("id"):
            return str(group["id"])
    return None


class ExpertResolver:
    """Per-product expert membership from Slack user groups, with a TTL cache.

    Args:
        group_ids: product -> Slack usergroup id or @handle (None for a product
            with no group; only allowed for products outside `required_products`).
        fetch_members: callable taking a usergroup id and returning its member
            user ids. Must raise `RateLimited` for Slack `ratelimited`.
        find_group_id: callable taking a handle (without "@") and returning the
            usergroup id. Only called for @handle values.
        cache_seconds: membership TTL in seconds (default 300).
        required_products: products that must have a group id (default IDA and
            IDDM). Tests pass () to build partial resolvers.
        time_source / sleep: injection points for tests.

    Raises:
        ValueError: a required product has no EXPERT_GROUP_ID_<PRODUCT>.
    """

    def __init__(
        self,
        group_ids: Optional[Mapping[str, Optional[str]]] = None,
        fetch_members: Optional[FetchMembers] = None,
        cache_seconds: float = DEFAULT_CACHE_SECONDS,
        required_products: Sequence[str] = REQUIRED_PRODUCTS,
        time_source: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        find_group_id: Optional[FindGroupId] = None,
    ):
        self.group_ids: Dict[str, Optional[str]] = dict(group_ids or {})
        self._fetch_members = fetch_members
        self._find_group_id = find_group_id
        self.cache_seconds = float(cache_seconds)
        self._now = time_source
        self._sleep = sleep
        self._cache: Dict[str, _Entry] = {}
        self._resolved_ids: Dict[str, str] = {}

        missing = self.unconfigured_products(required_products)
        if missing:
            raise ValueError(
                "Missing required Slack expert user group id(s): "
                + ", ".join(f"EXPERT_GROUP_ID_{product}" for product in missing)
            )

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, **kwargs) -> "ExpertResolver":
        """Build a resolver from techsupport_sync.env (+ process environment).

        Raises ValueError if EXPERT_GROUP_ID_IDA or _IDDM is missing.
        """
        env = env if env is not None else load_env()
        kwargs.setdefault("fetch_members", make_slack_fetcher(env.get("SLACK_BOT_TOKEN", "")))
        kwargs.setdefault("find_group_id", make_slack_group_finder(env.get("SLACK_BOT_TOKEN", "")))
        return cls(
            group_ids=group_ids_from_env(env),
            cache_seconds=cache_seconds_from_env(env),
            **kwargs,
        )

    # ---------------------------------------------------------------- config

    def group_id(self, product: Optional[str]) -> Optional[str]:
        return self.group_ids.get(product) if product else None

    def is_configured(self, product: Optional[str]) -> bool:
        """True if the product has a Slack expert user group."""
        return bool(self.group_id(product))

    def unconfigured_products(self, products: Iterable[str]) -> List[str]:
        """Which of `products` have no expert user group."""
        return [product for product in products if not self.is_configured(product)]

    # ------------------------------------------------------------ resolution

    def expert_user_ids(self, product: Optional[str]) -> List[str]:
        """Expert user ids for `product`, refreshing the cache when it is stale.

        Returns [] for an unknown product and for an optional product with no
        group configured. Raises ExpertLookupError if the group is configured
        but unreadable and nothing was ever cached for it.
        """
        if not product:
            return []
        group_id = self.group_id(product)
        if not group_id:
            return []

        entry = self._cache.get(product)
        if entry is not None and self._now() < entry.expires_at:
            return list(entry.members)
        return self._refresh(product, group_id)

    def is_expert(self, user_id: Optional[str], product: Optional[str]) -> bool:
        """True if `user_id` is a member of `product`'s expert user group."""
        if not user_id:
            return False
        return user_id in self.expert_user_ids(product)

    def primary_expert_id(self, product: Optional[str]) -> Optional[str]:
        """First member of the product's expert user group, else None."""
        experts = self.expert_user_ids(product)
        return experts[0] if experts else None

    # --------------------------------------------------------------- private

    def _refresh(self, product: str, group_id: str) -> List[str]:
        """Fetch membership, retrying once on rate limit."""
        try:
            if self._fetch_members is None:
                raise RuntimeError(f"No Slack fetcher configured for expert group {group_id} ({product})")
            usergroup_id = self._usergroup_id(group_id)
            try:
                members = self._fetch_once(usergroup_id)
            except RateLimited as exc:
                self._sleep(exc.retry_after)
                members = self._fetch_once(usergroup_id)
        except Exception as exc:  # pylint:disable=broad-except
            return self._serve_cached_or_raise(product, group_id, exc)

        self._cache[product] = _Entry(list(members), self._now() + self.cache_seconds)
        return list(members)

    def _usergroup_id(self, group_id: str) -> str:
        """`group_id` unchanged, or the id behind an @handle (looked up once)."""
        if not group_id.startswith(HANDLE_PREFIX):
            return group_id
        if group_id not in self._resolved_ids:
            if self._find_group_id is None:
                raise RuntimeError(f"No Slack handle lookup configured for expert group {group_id}")
            self._resolved_ids[group_id] = self._find_group_id(group_id[len(HANDLE_PREFIX):])
            logger.info("Resolved Slack expert user group %s to %s", group_id, self._resolved_ids[group_id])
        return self._resolved_ids[group_id]

    def _fetch_once(self, group_id: str) -> List[str]:
        members = self._fetch_members(group_id)  # type: ignore[misc]
        return [member for member in (_clean(m) for m in (members or [])) if member]

    def _serve_cached_or_raise(self, product: str, group_id: str, cause: BaseException) -> List[str]:
        """Keep serving the last good membership, or fail loudly if there is none."""
        entry = self._cache.get(product)
        if entry is None:
            raise ExpertLookupError(product, group_id, cause)
        logger.warning(
            "Slack usergroups.users.list failed for %s group %s: %s - serving the last known "
            "membership for another %ss",
            product,
            group_id,
            cause,
            self.cache_seconds,
        )
        # Re-stamp the expiry so a persistent outage is not a request storm.
        entry.expires_at = self._now() + self.cache_seconds
        return list(entry.members)


_DEFAULT_RESOLVER: Optional[ExpertResolver] = None


def get_default_resolver() -> ExpertResolver:
    """Process-wide resolver built from techsupport_sync.env on first use.

    Raises ValueError if a required EXPERT_GROUP_ID_* is missing.
    """
    global _DEFAULT_RESOLVER  # pylint:disable=global-statement
    if _DEFAULT_RESOLVER is None:
        _DEFAULT_RESOLVER = ExpertResolver.from_env()
    return _DEFAULT_RESOLVER


def set_default_resolver(resolver: Optional[ExpertResolver]) -> None:
    """Replace (or, with None, reset) the process-wide resolver -- used by tests."""
    global _DEFAULT_RESOLVER  # pylint:disable=global-statement
    _DEFAULT_RESOLVER = resolver


def expert_user_ids(product: Optional[str]) -> List[str]:
    """Expert user ids for `product` using the process-wide resolver."""
    return get_default_resolver().expert_user_ids(product)


def is_expert(user_id: Optional[str], product: Optional[str]) -> bool:
    """True if `user_id` is an expert for `product`, via the process-wide resolver."""
    return get_default_resolver().is_expert(user_id, product)
