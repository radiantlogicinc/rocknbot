"""
expert_group.py
====================================
Resolves "who is an expert for product X" from a Slack user group. The user
group is the only source of expert identity: there is no single-ID setting and
no silent degradation when the group cannot be read.

Configuration (read from lil-lisa/app_envfiles/lil-lisa.env, overridable in the
process environment):
    EXPERT_GROUP_ID_IDA / _IDDM          Slack user group ID (e.g. S0123ABCD) or
                                         @handle (e.g. @ida-experts).
                                         REQUIRED: the resolver raises if unset.
    EXPERT_GROUP_ID_IDO                  Optional -- IDO is only deployed for
                                         some workspaces. With no group, nobody
                                         is an IDO expert.
    EXPERT_GROUP_CACHE_SECONDS           Membership cache TTL (default 300).

Shared rules -- LilLisa_Server/cron/expert_group.py implements the same contract
with a blocking `requests` fetcher; keep the two aligned:
  1. Membership comes from Slack `usergroups.users.list?usergroup=<id>` and is
     cached per product for EXPERT_GROUP_CACHE_SECONDS (default 300s) so hot
     paths (every message, every reaction) do not hit Slack.
  2. On `ratelimited`, the fetch is retried once after Retry-After seconds.
  3. A required product (IDA, IDDM) with no group id is a configuration error:
     the constructor raises ValueError naming the missing variable.
  4. On a failed lookup, the last successfully cached membership keeps being
     served (its expiry is re-stamped and a warning is logged) so a brief Slack
     outage is survivable. With nothing cached, ExpertLookupError is raised so
     the failure is loud instead of silently making everyone a non-expert.
  5. An unconfigured optional product resolves to [] without any Slack call.
  6. A value starting with "@" is a user group handle. It is resolved to its ID
     once via Slack `usergroups.list` (same `usergroups:read` scope) and the ID
     is kept for the life of the process. An unknown handle is a failed lookup
     under rule 4. Any other value is used as the ID unchanged.

This module deliberately imports nothing from slack_bolt/slack_sdk: the Slack
calls are injected as `fetch_members` and `find_group_id`, so tests (and the
cron package) can supply their own. A fetcher must raise `RateLimited` for Slack's `ratelimited` error to
get the retry-once behaviour.
"""

import asyncio
import time
from typing import Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from utils import logger

PRODUCTS: Sequence[str] = ("IDA", "IDDM", "IDO")
# Products the bot cannot sensibly run without an expert group for.
REQUIRED_PRODUCTS: Sequence[str] = ("IDA", "IDDM")
DEFAULT_CACHE_SECONDS = 300.0

# The injected Slack call: usergroup id -> member user ids.
FetchMembers = Callable[[str], Awaitable[List[str]]]
# The injected Slack call: usergroup handle (without "@") -> usergroup id.
FindGroupId = Callable[[str], Awaitable[str]]

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
    answer to "is this user an expert", and guessing "no" would silently drop
    expert thumbs-up and #answer handling.
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
        logger.warning(f"EXPERT_GROUP_CACHE_SECONDS={raw!r} is not a number - using {DEFAULT_CACHE_SECONDS}")
        return DEFAULT_CACHE_SECONDS


def group_id_for_handle(usergroups: Iterable[Mapping[str, object]], handle: str) -> str:
    """The id of the group in a `usergroups.list` payload whose handle is `handle`."""
    wanted = handle.lower()
    for group in usergroups:
        if str(group.get("handle") or "").lower() == wanted and group.get("id"):
            return str(group["id"])
    raise RuntimeError(f"No Slack user group has the handle {HANDLE_PREFIX}{handle}")


class ExpertResolver:
    """Per-product expert membership from Slack user groups, with a TTL cache.

    Args:
        group_ids: product -> Slack usergroup id or @handle (None for a product
            with no group; only allowed for products outside `required_products`).
        fetch_members: async callable taking a usergroup id and returning its
            member user ids. Must raise `RateLimited` for Slack `ratelimited`.
        find_group_id: async callable taking a handle (without "@") and
            returning the usergroup id. Only called for @handle values.
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
        sleep: Optional[Callable[[float], Awaitable[None]]] = None,
        find_group_id: Optional[FindGroupId] = None,
    ):
        self.group_ids: Dict[str, Optional[str]] = dict(group_ids or {})
        self._fetch_members = fetch_members
        self._find_group_id = find_group_id
        self.cache_seconds = float(cache_seconds)
        self._now = time_source
        self._sleep = sleep or asyncio.sleep
        self._cache: Dict[str, _Entry] = {}
        self._resolved_ids: Dict[str, str] = {}

        missing = self.unconfigured_products(required_products)
        if missing:
            raise ValueError(
                "Missing required Slack expert user group id(s): "
                + ", ".join(f"EXPERT_GROUP_ID_{product}" for product in missing)
            )

    def set_fetcher(self, fetch_members: Optional[FetchMembers]) -> None:
        """Attach the Slack call after construction (slack.py needs `app` first)."""
        self._fetch_members = fetch_members

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

    async def expert_user_ids(self, product: Optional[str]) -> List[str]:
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
        return await self._refresh(product, group_id)

    async def is_expert(self, user_id: Optional[str], product: Optional[str]) -> bool:
        """True if `user_id` is a member of `product`'s expert user group."""
        if not user_id:
            return False
        return user_id in await self.expert_user_ids(product)

    async def primary_expert_id(self, product: Optional[str]) -> Optional[str]:
        """First member of the product's expert user group, else None."""
        experts = await self.expert_user_ids(product)
        return experts[0] if experts else None

    # --------------------------------------------------------------- private

    async def _refresh(self, product: str, group_id: str) -> List[str]:
        """Fetch membership, retrying once on rate limit."""
        try:
            if self._fetch_members is None:
                raise RuntimeError(f"No Slack fetcher configured for expert group {group_id} ({product})")
            usergroup_id = await self._usergroup_id(group_id)
            try:
                members = await self._fetch_once(usergroup_id)
            except RateLimited as exc:
                await self._sleep(exc.retry_after)
                members = await self._fetch_once(usergroup_id)
        except Exception as exc:  # pylint:disable=broad-except
            return self._serve_cached_or_raise(product, group_id, exc)

        self._cache[product] = _Entry(list(members), self._now() + self.cache_seconds)
        return list(members)

    async def _usergroup_id(self, group_id: str) -> str:
        """`group_id` unchanged, or the id behind an @handle (looked up once)."""
        if not group_id.startswith(HANDLE_PREFIX):
            return group_id
        if group_id not in self._resolved_ids:
            if self._find_group_id is None:
                raise RuntimeError(f"No Slack handle lookup configured for expert group {group_id}")
            self._resolved_ids[group_id] = await self._find_group_id(group_id[len(HANDLE_PREFIX):])
            logger.info(f"Resolved Slack expert user group {group_id} to {self._resolved_ids[group_id]}")
        return self._resolved_ids[group_id]

    async def _fetch_once(self, group_id: str) -> List[str]:
        members = await self._fetch_members(group_id)  # type: ignore[misc]
        return [member for member in (_clean(m) for m in (members or [])) if member]

    def _serve_cached_or_raise(self, product: str, group_id: str, cause: BaseException) -> List[str]:
        """Keep serving the last good membership, or fail loudly if there is none."""
        entry = self._cache.get(product)
        if entry is None:
            raise ExpertLookupError(product, group_id, cause)
        logger.warning(
            f"Slack usergroups.users.list failed for {product} group {group_id}: {cause} "
            f"- serving the last known membership for another {self.cache_seconds}s"
        )
        # Re-stamp the expiry so a persistent outage is not a request storm.
        entry.expires_at = self._now() + self.cache_seconds
        return list(entry.members)
