"""
slack_channel_check.py
====================================
Logs, once at startup, which Slack channel each configured channel ID points
at, e.g. `Slack channel TECHSUPPORT_CHANNEL_ID = C0123ABCD (#techsupport)`.
A wrong or misplaced ID is then obvious from the startup log instead of
surfacing later as a pipeline that silently reads the wrong channel.

One Slack `conversations.info` call per configured ID (no paging). Public
channels need `channels:read`; private ones also need `groups:read`, and the
bot only sees private channels it is a member of. Best-effort by design: a
failed lookup is logged as a WARNING naming the Slack error and never raises,
because the IDs are validated for presence by techsupport_preflight.py and
this check must not turn a Slack hiccup into a failed rollout.

Channel IDs are read from cron/env/techsupport_sync.env overlaid by the
process environment, the same load_env pattern as nightly_techsupport_sync.py.
CHANNEL_ID_IDA / _IDDM / _IDO come from lil-lisa.env when this file does not
set them.

Usage (from src/techsupport_cron.py at server startup):
    from slack_channel_check import log_configured_channels
    log_configured_channels()
"""

import logging
import os
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import requests
from dotenv import dotenv_values
from paths import CHANNEL_ID_KEYS, PACKAGE_ROOT, read_lil_lisa_channel_ids

ENV_PATH = PACKAGE_ROOT / "env" / "techsupport_sync.env"
SLACK_API_BASE = "https://slack.com/api"
SLACK_TIMEOUT_SECONDS = 10

CHANNEL_KEYS: Tuple[str, ...] = (
    "TECHSUPPORT_CHANNEL_ID",
    "ADMIN_CHANNEL_ID",
    *CHANNEL_ID_KEYS,
)

logger = logging.getLogger(__name__)

# The injected Slack call: channel id -> conversations.info JSON payload.
FetchChannelInfo = Callable[[str], Mapping[str, Any]]


def load_env() -> Dict[str, str]:
    env: Dict[str, str] = read_lil_lisa_channel_ids()
    env.update({k: v for k, v in dotenv_values(str(ENV_PATH)).items() if v is not None})
    env.update(os.environ)
    return env


def make_slack_channel_info(token: str) -> FetchChannelInfo:
    """Default fetcher: Slack `conversations.info` over `requests`."""

    def fetch(channel_id: str) -> Mapping[str, Any]:
        resp = requests.get(
            f"{SLACK_API_BASE}/conversations.info",
            headers={"Authorization": f"Bearer {token}"},
            params={"channel": channel_id},
            timeout=SLACK_TIMEOUT_SECONDS,
        )
        return resp.json()

    return fetch


def describe_channel(payload: Mapping[str, Any]) -> Tuple[bool, str]:
    """(ok, text) for a conversations.info payload: "#name" plus any flags
    that would stop the pipeline reading the channel, or the Slack error."""
    if not payload.get("ok"):
        error = payload.get("error") or "unknown_error"
        hint = ""
        if error == "channel_not_found":
            hint = " (wrong ID, or a private channel the bot is not in)"
        elif error == "missing_scope":
            hint = " (the Slack app needs channels:read, plus groups:read for private channels)"
        return False, f"{error}{hint}"
    channel = payload.get("channel") or {}
    flags = []
    if channel.get("is_private"):
        flags.append("private")
    if channel.get("is_archived"):
        flags.append("ARCHIVED")
    if channel.get("is_member") is False:
        flags.append("bot is NOT a member")
    suffix = f" [{', '.join(flags)}]" if flags else ""
    return not channel.get("is_archived") and channel.get("is_member") is not False, f"#{channel.get('name')}{suffix}"


def log_configured_channels(
    env: Optional[Mapping[str, str]] = None,
    fetch_channel_info: Optional[FetchChannelInfo] = None,
) -> Dict[str, str]:
    """Log each configured channel ID with its Slack name. Never raises.

    Returns {key: description} for the keys that were set, for tests and
    callers that want the result."""
    env = env if env is not None else load_env()
    configured = {key: (env.get(key) or "").strip() for key in CHANNEL_KEYS}
    configured = {key: value for key, value in configured.items() if value}
    if not configured:
        return {}
    if fetch_channel_info is None:
        token = (env.get("SLACK_BOT_TOKEN") or "").strip()
        if not token:
            logger.warning("Skipping Slack channel check: SLACK_BOT_TOKEN is not set")
            return {}
        fetch_channel_info = make_slack_channel_info(token)

    results: Dict[str, str] = {}
    for key, channel_id in configured.items():
        try:
            ok, text = describe_channel(fetch_channel_info(channel_id))
        except Exception as exc:  # noqa: BLE001 -- best-effort startup log, never fatal
            ok, text = False, f"lookup failed: {type(exc).__name__}: {exc}"
        results[key] = text
        if ok:
            logger.info("Slack channel %s = %s (%s)", key, channel_id, text)
        else:
            logger.warning("Slack channel %s = %s could not be confirmed: %s", key, channel_id, text)
    return results
