"""
nightly_techsupport_sync.py
====================================
Standalone script (no dependency on LilLisa_Server or lil-lisa's runtime code)
that detects which threads in a Slack techsupport channel are new or have new
reply activity since the last run.

Slack's conversations.history only returns new top-level messages (filtered by
an "oldest" timestamp) -- it does NOT surface new replies posted to existing
threads. Slack has no "threads updated since T" API; parent messages do carry
latest_reply when you fetch the parent itself.

Detection therefore:
  1. Finds new parents via conversations.history(oldest=last_run) and stores
     each message's latest_reply (no extra call).
  2. Refreshes latest_reply on a bounded set of already-known threads by
     fetching the parent only (conversations.history, inclusive, limit=1) --
     never conversations.replies. Nightly: threads whose last_seen_reply_ts
     is within TECHSUPPORT_SYNC_HOT_DAYS (default 30). Periodic catch-up:
     threads within TECHSUPPORT_SYNC_CATCHUP_AGE_DAYS (default 90). Both
     lists are capped at TECHSUPPORT_SYNC_MAX_PARENT_LOOKUPS. Threads quieter
     than the catch-up age cap are not polled (state is kept, including
     added_to_verified_db). The pipeline fetches full replies later, and only
     for new/updated ids.

This is step one of the nightly techsupport sync job: it only detects new /
updated threads. Classifying threads as resolved (or anything else) is a
separate, later step.

sync() is channel-parametric: sync() with no argument syncs
TECHSUPPORT_CHANNEL_ID (the historical behaviour nightly_pipeline.py's
techsupport loop relies on), sync(channel_id=...) syncs any other channel --
used by nightly_pipeline.py's product-channel pass (CHANNEL_ID_IDA /
_IDDM / _IDO, the same variables lil-lisa answers questions in), which looks
for expert corrections in the IDA/IDDM/IDO
channels. State is therefore keyed by channel:

    {"version": 2,
     "channels": {"<channel id>": {"last_run_timestamp": ...,
                                   "last_catchup_timestamp": ...,
                                   "threads": {...}}}}

load_state() accepts only this shape. A missing file starts a fresh v2
state; an existing file whose top-level "version" is not 2 (including the
pre-version-2 flat shape, which had last_run_timestamp /
last_catchup_timestamp / threads at the top level) raises RuntimeError
naming the path, so the run stops instead of silently starting over. The
operator deletes the file and re-runs; nothing shipped has the old shape.

Required env vars (see ./env/techsupport_sync.env):
    SLACK_BOT_TOKEN        - Slack bot token with channels:history / groups:history
                              (and the read scope for the channel type in use)
    TECHSUPPORT_CHANNEL_ID - Channel ID to sync (e.g. the test-techsupport channel)

Optional:
    CHANNEL_ID_IDA / _IDDM / _IDO - product channels scanned for expert
                              corrections. Defined in lil-lisa.env (or the
                              process environment). A missing one disables
                              that product. Do not copy them into
                              techsupport_sync.env.
    LIL_LISA_SLACK_USERID   - the bot's Slack user id, used to role-tag its turns.

Optional knobs (env/lillisa_server.env, overridable in the process environment):
    TECHSUPPORT_SYNC_HOT_DAYS                - nightly parent-lookup window (default 30)
    TECHSUPPORT_SYNC_CATCHUP_AGE_DAYS        - catch-up age cap (default 90)
    TECHSUPPORT_SYNC_CATCHUP_INTERVAL_DAYS   - how often catch-up runs (default 7)
    TECHSUPPORT_SYNC_MAX_PARENT_LOOKUPS      - cap per lookup set (default 200)

Usage:
    python nightly_techsupport_sync.py
"""

import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import requests
from dotenv import dotenv_values
from paths import (
    LILLISA_SERVER_ENV_PATH,
    LILLISA_SERVER_ROOT,
    PACKAGE_ROOT,
    SYNC_STATE_PATH,
    ensure_import_paths,
    read_lil_lisa_channel_ids,
)

ensure_import_paths()
SCRIPT_DIR = PACKAGE_ROOT
PROJECT_ROOT = LILLISA_SERVER_ROOT

from atomic_io import atomic_write_json  # noqa: E402

ENV_PATH = SCRIPT_DIR / "env" / "techsupport_sync.env"
STATE_PATH = SYNC_STATE_PATH

# Bump only for a shape change; load_state() rejects any other version.
STATE_VERSION = 2

PRODUCTS: Tuple[str, ...] = ("IDA", "IDDM", "IDO")
CHANNEL_ID_ENV_KEY = "CHANNEL_ID_{product}"

SLACK_API_BASE = "https://slack.com/api"
IGNORED_SUBTYPES = {"channel_join", "channel_leave"}
SECONDS_PER_DAY = 86400.0

DEFAULT_HOT_DAYS = 30.0
DEFAULT_CATCHUP_AGE_DAYS = 90.0
DEFAULT_CATCHUP_INTERVAL_DAYS = 7.0
DEFAULT_MAX_PARENT_LOOKUPS = 200

# Same propagation-delay quirk fixed the same way in
# lil-lisa/src/slack.py's conversations_replies_with_retry(): a channel/
# message can momentarily fail to resolve via the Web API an instant after
# it's posted, even though it's otherwise valid and a retry a few seconds
# later succeeds. Same delay/attempt counts as that fix.
CHANNEL_NOT_FOUND_RETRY_DELAY_SECONDS = 3.0
CHANNEL_NOT_FOUND_MAX_RETRIES = 2

# Parent-lookup loop fires one conversations.history call per selected
# thread. Throttle the same way pagination already is, so a burst of
# same-token requests does not trip transient channel_not_found.
PER_THREAD_THROTTLE_SECONDS = 0.3

# No basicConfig() call -- library module, imported into nightly_pipeline.py
# (which configures logging itself); this just needs a named logger that
# inherits whichever configuration is already in place. Standalone runs
# (main() below) don't configure logging either, matching this module's
# existing print()-based console output -- INFO-level retry logging here
# is aimed at nightly_pipeline.py's log stream, not standalone use.
logger = logging.getLogger(__name__)


def _file_env() -> Dict[str, str]:
    """techsupport_sync.env, with CHANNEL_ID_* filled from lil-lisa.env.

    The product question channels are defined once, in lil-lisa.env. A value
    in techsupport_sync.env still wins, and the process environment (applied
    by the callers) wins over both. Only CHANNEL_ID_* are taken from
    lil-lisa.env; the rest of that file stays with the Slack bot.
    """
    env = read_lil_lisa_channel_ids()
    env.update({k: v for k, v in dotenv_values(str(ENV_PATH)).items() if v is not None})
    return env


def load_env() -> Dict[str, str]:
    """Load env vars the same way lil-lisa/src/slack.py does: values from the
    env file, overridden by any matching variables already in the process
    environment. CHANNEL_ID_* also come from lil-lisa.env when this file
    does not set them."""
    env = _sync_env()

    required = ["SLACK_BOT_TOKEN", "TECHSUPPORT_CHANNEL_ID"]
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise RuntimeError(f"Missing required env var(s) {missing} - expected in {ENV_PATH}")
    return env


def _pipeline_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    if LILLISA_SERVER_ENV_PATH.exists():
        env.update({k: v for k, v in dotenv_values(str(LILLISA_SERVER_ENV_PATH)).items() if v is not None})
    env.update({k: v for k, v in os.environ.items() if v is not None})
    return env


def get_hot_days() -> float:
    return float(_pipeline_env().get("TECHSUPPORT_SYNC_HOT_DAYS", str(DEFAULT_HOT_DAYS)))


def get_catchup_age_days() -> float:
    return float(_pipeline_env().get("TECHSUPPORT_SYNC_CATCHUP_AGE_DAYS", str(DEFAULT_CATCHUP_AGE_DAYS)))


def get_catchup_interval_days() -> float:
    return float(_pipeline_env().get("TECHSUPPORT_SYNC_CATCHUP_INTERVAL_DAYS", str(DEFAULT_CATCHUP_INTERVAL_DAYS)))


def get_max_parent_lookups() -> int:
    return int(float(_pipeline_env().get("TECHSUPPORT_SYNC_MAX_PARENT_LOOKUPS", str(DEFAULT_MAX_PARENT_LOOKUPS))))


def product_channel_ids(env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """{"IDA": "C123", ...} for whichever CHANNEL_ID_* are configured.

    All three are optional; a product with no channel id is simply not scanned.
    """
    env = env if env is not None else _sync_env()
    configured: Dict[str, str] = {}
    for product in PRODUCTS:
        channel_id = (env.get(CHANNEL_ID_ENV_KEY.format(product=product)) or "").strip()
        if channel_id:
            configured[product] = channel_id
    return configured


def _sync_env() -> Dict[str, str]:
    """techsupport_sync.env overlaid by the process environment, with none of
    load_env()'s required-var enforcement (callers here only want optionals).

    CHANNEL_ID_* missing from this file are filled from lil-lisa.env.
    """
    env = _file_env()
    env.update({k: v for k, v in os.environ.items() if v is not None})
    return env


def default_channel_id() -> Optional[str]:
    """TECHSUPPORT_CHANNEL_ID, or None if it isn't configured."""
    return (_sync_env().get("TECHSUPPORT_CHANNEL_ID") or "").strip() or None


def new_state() -> Dict[str, Any]:
    return {"version": STATE_VERSION, "channels": {}}


def channel_state(state: Dict[str, Any], channel_id: str) -> Dict[str, Any]:
    """The per-channel sub-state (created empty if this channel is new)."""
    channels = state.setdefault("channels", {})
    per_channel = channels.setdefault(channel_id, {})
    per_channel.setdefault("last_run_timestamp", "0")
    per_channel.setdefault("threads", {})
    return per_channel


def load_state() -> Dict[str, Any]:
    """Read the channel-keyed state file, or start a fresh one if it is absent.

    Any existing file that is not version 2 is rejected rather than repaired:
    this file was introduced on this branch, so a foreign shape means the file
    is stale or hand-edited, and silently starting over would quietly lose
    every added_to_verified_db flag.
    """
    if not STATE_PATH.exists():
        return new_state()
    with open(STATE_PATH, encoding="utf-8") as f:
        state = json.load(f)
    if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
        raise RuntimeError(
            f"{STATE_PATH} predates the channel-keyed state format "
            f"(expected top-level \"version\": {STATE_VERSION}). Delete the file and re-run; "
            "the next run rebuilds it, and for the tech support channel that means the first "
            "sync starts from the beginning of the channel again."
        )
    return state


def save_state(state: Dict[str, Any]) -> None:
    atomic_write_json(STATE_PATH, state, indent=2, sort_keys=True)


def slack_api_call(method: str, token: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Call a Slack Web API method, retrying on rate limits and, a limited
    number of times, on "channel_not_found" -- see
    CHANNEL_NOT_FOUND_MAX_RETRIES above for why. Any other error is raised
    immediately, unretried, so it isn't masked."""
    url = f"{SLACK_API_BASE}/{method}"
    headers = {"Authorization": f"Bearer {token}"}
    channel_not_found_attempts = 0
    while True:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        data = resp.json()
        if data.get("ok"):
            return data
        if data.get("error") == "ratelimited":
            retry_after = int(resp.headers.get("Retry-After", "1"))
            time.sleep(retry_after)
            continue
        if data.get("error") == "channel_not_found" and channel_not_found_attempts < CHANNEL_NOT_FOUND_MAX_RETRIES:
            channel_not_found_attempts += 1
            logger.info(
                "[RETRY %s] channel_not_found for channel=%r (attempt %d/%d) -- retrying in %ss",
                method,
                params.get("channel"),
                channel_not_found_attempts,
                CHANNEL_NOT_FOUND_MAX_RETRIES,
                CHANNEL_NOT_FOUND_RETRY_DELAY_SECONDS,
            )
            time.sleep(CHANNEL_NOT_FOUND_RETRY_DELAY_SECONDS)
            continue
        if data.get("error") == "channel_not_found":
            logger.error(
                "[RETRY %s] Exhausted retries -- channel_not_found persisted for channel=%r",
                method,
                params.get("channel"),
            )
        raise RuntimeError(f"Slack API error calling {method}: {data.get('error')}")


def paginate_messages(method: str, token: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Collect the "messages" list across all pages of a cursor-paginated
    Slack conversations.* method."""
    messages: List[Dict[str, Any]] = []
    cursor = None
    while True:
        call_params = dict(params)
        if cursor:
            call_params["cursor"] = cursor
        data = slack_api_call(method, token, call_params)
        messages.extend(data.get("messages", []))
        cursor = data.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            return messages
        time.sleep(1)


def find_new_threads(token: str, channel_id: str, oldest: str, known_thread_ids: set) -> List[Dict[str, Any]]:
    """Find top-level messages posted since `oldest` that aren't already tracked."""
    history = paginate_messages(
        "conversations.history",
        token,
        {"channel": channel_id, "oldest": oldest, "limit": 200},
    )
    new_threads = []
    for msg in history:
        if msg.get("subtype") in IGNORED_SUBTYPES:
            continue
        thread_ts = msg["ts"]
        if thread_ts in known_thread_ids:
            continue
        new_threads.append(msg)
    return new_threads


def parent_activity_ts(thread_ts: str, info: Optional[Dict[str, Any]] = None) -> float:
    """Unix time of last known activity: stored last_seen_reply_ts, else parent ts."""
    info = info or {}
    raw = info.get("last_seen_reply_ts") or thread_ts
    return float(raw)


def select_parent_lookups(
    thread_ids: Sequence[str],
    threads: Dict[str, Any],
    now: float,
    window_days: float,
    max_lookups: int,
    exclude: Optional[Iterable[str]] = None,
) -> Tuple[List[str], int]:
    """Pick known threads to refresh via a cheap parent latest_reply fetch.

    Includes threads whose last_seen_reply_ts is within window_days of now,
    hottest first (most recent activity first). Caps at max_lookups. Does not
    delete or mutate state for threads that age out or exceed the cap.

    Returns (ids_to_check, skipped_over_cap).
    """
    excluded: Set[str] = set(exclude or [])
    window_seconds = window_days * SECONDS_PER_DAY
    eligible: List[Tuple[float, str]] = []
    for thread_ts in thread_ids:
        if thread_ts in excluded:
            continue
        activity = parent_activity_ts(thread_ts, threads.get(thread_ts) or {})
        if now - activity <= window_seconds:
            eligible.append((activity, thread_ts))
    eligible.sort(key=lambda item: item[0], reverse=True)
    chosen = [thread_ts for _, thread_ts in eligible[:max_lookups]]
    skipped_over_cap = max(0, len(eligible) - len(chosen))
    return chosen, skipped_over_cap


def is_catchup_due(channel_sub_state: Dict[str, Any], now: float, interval_days: float) -> bool:
    """`channel_sub_state` is one channel's slice of the state (channel_state())."""
    last = channel_sub_state.get("last_catchup_timestamp")
    if last is None or last == "":
        return True
    return (now - float(last)) >= interval_days * SECONDS_PER_DAY


def fetch_parent_latest_reply_ts(token: str, channel_id: str, thread_ts: str) -> str:
    """Return Slack's latest_reply on the parent message, or the parent ts.

    Uses conversations.history inclusive limit=1 (Slack's documented single-
    message lookup) so sync() never downloads the reply list. The pipeline
    fetches conversations.replies later for threads that actually changed.
    """
    data = slack_api_call(
        "conversations.history",
        token,
        {
            "channel": channel_id,
            "oldest": thread_ts,
            "inclusive": "true",
            "limit": 1,
        },
    )
    messages = data.get("messages") or []
    if not messages:
        raise RuntimeError(f"No parent message found for thread {thread_ts}")
    parent = messages[0]
    return parent.get("latest_reply") or parent.get("ts") or thread_ts


def _refresh_known_threads(
    token: str,
    channel_id: str,
    threads: Dict[str, Any],
    thread_ids: Sequence[str],
) -> List[str]:
    """Compare stored last_seen_reply_ts to Slack latest_reply; return updated ids."""
    updated_thread_ids: List[str] = []
    for index, thread_ts in enumerate(thread_ids):
        try:
            newest_ts = fetch_parent_latest_reply_ts(token, channel_id, thread_ts)
        except Exception as exc:  # noqa: BLE001 -- one missing parent must not abort the run
            logger.warning("Skipping parent lookup for thread %s: %s", thread_ts, exc)
            continue
        previous = threads.setdefault(thread_ts, {}).get("last_seen_reply_ts", thread_ts)
        if float(newest_ts) > float(previous):
            updated_thread_ids.append(thread_ts)
        threads[thread_ts]["last_seen_reply_ts"] = newest_ts
        if index < len(thread_ids) - 1:
            time.sleep(PER_THREAD_THROTTLE_SECONDS)
    return updated_thread_ids


def sync(channel_id: Optional[str] = None) -> Dict[str, List[str]]:
    """Detect new/updated threads in one Slack channel and update that
    channel's slice of the on-disk state file accordingly. Returns
    {"new_thread_ids": [...], "updated_thread_ids": [...]} so callers (e.g.
    nightly_pipeline.py) can process exactly the threads that changed since
    the last run, without re-implementing the state-tracking logic here.

    `channel_id` defaults to TECHSUPPORT_CHANNEL_ID, which is exactly the
    behaviour every pre-existing caller had. Pass a product channel id to run
    the same detection there; the hot/catch-up windows and the parent-lookup
    cap are per channel, using the same env knobs."""
    env = load_env()
    token = env["SLACK_BOT_TOKEN"]
    channel_id = channel_id or env["TECHSUPPORT_CHANNEL_ID"]

    state = load_state()
    per_channel = channel_state(state, channel_id)
    threads = per_channel["threads"]
    last_run_timestamp = per_channel.get("last_run_timestamp", "0")
    now = time.time()

    # Snapshot of threads we already knew about *before* this run, so we can
    # tell "brand new" apart from "existing, checked for new replies".
    previously_known_thread_ids = list(threads.keys())

    new_threads = find_new_threads(token, channel_id, last_run_timestamp, set(previously_known_thread_ids))
    for msg in new_threads:
        thread_ts = msg["ts"]
        # conversations.history already tells us the latest reply ts for
        # threads with replies, so no extra API call is needed here.
        existing = threads.get(thread_ts) or {}
        threads[thread_ts] = {**existing, "last_seen_reply_ts": msg.get("latest_reply", thread_ts)}

    hot_days = get_hot_days()
    catchup_age_days = get_catchup_age_days()
    catchup_interval_days = get_catchup_interval_days()
    max_lookups = get_max_parent_lookups()

    hot_ids, hot_capped = select_parent_lookups(previously_known_thread_ids, threads, now, hot_days, max_lookups)
    lookup_ids = list(hot_ids)
    catchup_due = is_catchup_due(per_channel, now, catchup_interval_days)
    catchup_ids: List[str] = []
    catchup_capped = 0
    if catchup_due:
        catchup_ids, catchup_capped = select_parent_lookups(
            previously_known_thread_ids,
            threads,
            now,
            catchup_age_days,
            max_lookups,
            exclude=hot_ids,
        )
        lookup_ids.extend(catchup_ids)

    logger.info(
        "Parent lookups for channel %s: hot=%d (window=%.0fd, capped=%d) catchup_due=%s catchup=%d "
        "(age_cap=%.0fd, capped=%d) skipped_older_than_catchup=%d",
        channel_id,
        len(hot_ids),
        hot_days,
        hot_capped,
        catchup_due,
        len(catchup_ids),
        catchup_age_days,
        catchup_capped,
        _count_older_than(previously_known_thread_ids, threads, now, catchup_age_days),
    )

    updated_thread_ids = _refresh_known_threads(token, channel_id, threads, lookup_ids)

    per_channel["last_run_timestamp"] = f"{now:.6f}"
    if catchup_due:
        per_channel["last_catchup_timestamp"] = f"{now:.6f}"
    save_state(state)

    new_thread_ids = [msg["ts"] for msg in new_threads]
    return {"new_thread_ids": new_thread_ids, "updated_thread_ids": updated_thread_ids}


def _count_older_than(thread_ids: Sequence[str], threads: Dict[str, Any], now: float, age_days: float) -> int:
    window_seconds = age_days * SECONDS_PER_DAY
    older = 0
    for thread_ts in thread_ids:
        activity = parent_activity_ts(thread_ts, threads.get(thread_ts) or {})
        if now - activity > window_seconds:
            older += 1
    return older


def main() -> None:
    result = sync()
    print(f"Brand new threads: {len(result['new_thread_ids'])}")
    for thread_ts in result["new_thread_ids"]:
        print(f"  {thread_ts}")
    print(f"Existing threads with new activity: {len(result['updated_thread_ids'])}")
    for thread_ts in result["updated_thread_ids"]:
        print(f"  {thread_ts}")


if __name__ == "__main__":
    main()
