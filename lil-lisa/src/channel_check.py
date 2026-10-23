"""
channel_check.py
====================================
Logs, once at startup, which Slack channel each configured channel ID points
at, e.g. `Slack channel CHANNEL_ID_IDA = C0123ABCD (#rocknbot-ida)`. A wrong
or misplaced ID is then obvious from the startup log instead of surfacing as
a bot that silently ignores a channel.

One Slack `conversations.info` call per configured ID (no paging). Public
channels need `channels:read`; private ones also need `groups:read`, and the
bot only sees private channels it is a member of. Best-effort by design: a
failed lookup is logged as a warning naming the Slack error and never raises.

LilLisa_Server/cron/slack_channel_check.py is the backend twin (different
keys, blocking `requests` fetcher); keep the two aligned.

This module imports nothing from slack_bolt/slack_sdk: `conversations.info`
is injected as `fetch_channel_info`, returning Slack's JSON payload (including
`{"ok": false, "error": ...}` payloads rather than raising for them).
"""

from typing import Any, Awaitable, Callable, Dict, Mapping, Tuple

from utils import logger

CHANNEL_KEYS: Tuple[str, ...] = (
    "CHANNEL_ID_IDA",
    "CHANNEL_ID_IDDM",
    "CHANNEL_ID_IDO",
    "ADMIN_CHANNEL_ID_IDA",
    "ADMIN_CHANNEL_ID_IDDM",
    "ADMIN_CHANNEL_ID_IDO",
    "TECHSUPPORT_CHANNEL_ID_IDA",
    "TECHSUPPORT_CHANNEL_ID_IDDM",
    "TECHSUPPORT_CHANNEL_ID_IDO",
)

# The injected Slack call: channel id -> conversations.info JSON payload.
FetchChannelInfo = Callable[[str], Awaitable[Mapping[str, Any]]]


def describe_channel(payload: Mapping[str, Any]) -> Tuple[bool, str]:
    """(ok, text) for a conversations.info payload: "#name" plus any flags
    that would stop the bot working in the channel, or the Slack error."""
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


async def log_configured_channels(env: Mapping[str, Any], fetch_channel_info: FetchChannelInfo) -> Dict[str, str]:
    """Log each configured channel ID with its Slack name. Never raises.

    Returns {key: description} for the keys that were set."""
    configured = {key: str(env.get(key) or "").strip() for key in CHANNEL_KEYS}
    results: Dict[str, str] = {}
    for key, channel_id in configured.items():
        if not channel_id:
            continue
        try:
            ok, text = describe_channel(await fetch_channel_info(channel_id))
        except Exception as exc:  # pylint:disable=broad-except
            ok, text = False, f"lookup failed: {type(exc).__name__}: {exc}"
        results[key] = text
        if ok:
            logger.info(f"Slack channel {key} = {channel_id} ({text})")
        else:
            logger.warning(f"Slack channel {key} = {channel_id} could not be confirmed: {text}")
    return results
