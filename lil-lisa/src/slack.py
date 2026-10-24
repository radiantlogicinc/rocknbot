"""
slack.py
====================================
The core module of rocknbot responsible
for catching and handling events from
Slack
"""

import asyncio
import os
import time
import jwt

from typing import Dict, List, Any
import json

import requests
from dotenv import dotenv_values
from slack_bolt.adapter.socket_mode.async_handler import (  # type: ignore
    AsyncSocketModeHandler,
)
from slack_bolt.async_app import AsyncApp  # type: ignore
from slack_sdk.errors import SlackApiError  # type: ignore

from channel_check import log_configured_channels
from escalation_tracker import (
    claim_escalation,
    clear_escalation,
    hydrate_endorsement_tracker,
    is_escalation_active,
)
from expert_group import (
    ExpertResolver,
    RateLimited,
    cache_seconds_from_env,
    group_id_for_handle_across_pages,
    group_ids_from_env,
)
from utils import (
    ESCALATE_VALUE_QUERY_MAX_LENGTH,
    build_escalation_button_value,
    logger,
    parse_get_ans_result,
    truncate_preserving_code_fences,
    warn_if_escalate_body_channel_mismatch,
)

lil_lisa_env = dotenv_values("./app_envfiles/lil-lisa.env")

lil_lisa_env = {
    **lil_lisa_env,
    **os.environ,  # override loaded values with environment variables
}

LIL_LISA_SLACK_USERID = lil_lisa_env["LIL_LISA_SLACK_USERID"]
SLACK_BOT_TOKEN = lil_lisa_env["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = lil_lisa_env["SLACK_APP_TOKEN"]
CHANNEL_ID_IDDM = lil_lisa_env["CHANNEL_ID_IDDM"]
ADMIN_CHANNEL_ID_IDDM = lil_lisa_env["ADMIN_CHANNEL_ID_IDDM"]
CHANNEL_ID_IDA = lil_lisa_env["CHANNEL_ID_IDA"]
ADMIN_CHANNEL_ID_IDA = lil_lisa_env["ADMIN_CHANNEL_ID_IDA"]
# IDO Slack channels are optional — set to None if not configured
CHANNEL_ID_IDO = lil_lisa_env.get("CHANNEL_ID_IDO")
ADMIN_CHANNEL_ID_IDO = lil_lisa_env.get("ADMIN_CHANNEL_ID_IDO")
if not CHANNEL_ID_IDO:
    logger.warning("CHANNEL_ID_IDO not found in lil-lisa.env — IDO Slack channel support will be disabled")

# One shared tech-support channel for every product (distinct from the
# ADMIN_CHANNEL_ID_* bot-management channels). Optional — if unset, process_msg
# omits the escalate button.
TECHSUPPORT_CHANNEL_ID = (lil_lisa_env.get("TECHSUPPORT_CHANNEL_ID") or "").strip() or None
if not TECHSUPPORT_CHANNEL_ID:
    logger.warning(
        "TECHSUPPORT_CHANNEL_ID not found in lil-lisa.env — "
        "escalate-to-techsupport button will be omitted"
    )
# Expert identity is a Slack user group per product. EXPERT_GROUP_ID_IDA and
# _IDDM are required: the resolver below refuses to build without them, so the
# bot fails to boot on a misconfiguration instead of silently having no experts.
EXPERT_GROUP_ID_IDA = lil_lisa_env.get("EXPERT_GROUP_ID_IDA")
EXPERT_GROUP_ID_IDDM = lil_lisa_env.get("EXPERT_GROUP_ID_IDDM")
EXPERT_GROUP_ID_IDO = lil_lisa_env.get("EXPERT_GROUP_ID_IDO")
AUTHENTICATION_KEY = lil_lisa_env["AUTHENTICATION_KEY"]
ENCRYPTED_AUTHENTICATION_KEY = jwt.encode({"some": "payload"}, AUTHENTICATION_KEY, algorithm="HS256")  # type: ignore
MAX_LENGTH = int(lil_lisa_env["MAX_LENGTH"])
# Slack's hard cap on a section block's text field is 3000 characters. MAX_LENGTH (env-configured)
# is not guaranteed to respect that, so this is enforced independently as a defensive backstop.
SLACK_BLOCK_TEXT_LIMIT = 2900

BASE_URL = os.getenv("LIL_LISA_SERVER_URL", lil_lisa_env["LIL_LISA_SERVER_URL"])
BASE_URL = BASE_URL.rstrip('/')     # this is IMPORTANT! Otherwise you will see {"detail": "Not Found"} in the response
logger.info(f"LIL_LISA_SERVER_URL: {BASE_URL}")

TIMEOUT = 10
app = AsyncApp(token=SLACK_BOT_TOKEN)


async def _fetch_usergroup_members(usergroup_id):
    """Slack `usergroups.users.list` for the expert resolver (needs `usergroups:read`).

    Raises expert_group.RateLimited on `ratelimited` so the resolver owns the
    retry policy; every other error propagates and the resolver either serves
    its last cached membership or raises ExpertLookupError.
    """
    try:
        resp = await app.client.usergroups_users_list(usergroup=usergroup_id)
    except SlackApiError as exc:
        response = getattr(exc, "response", None)
        error = (response.get("error") if response is not None else None) or ""
        if error == "ratelimited":
            headers = getattr(response, "headers", {}) or {}
            raise RateLimited(float(headers.get("Retry-After", 1))) from exc
        raise
    return list(resp.get("users") or [])


async def _fetch_usergroup_page(cursor):
    """One page of Slack `usergroups.list`. `cursor` is "" for the first page."""
    try:
        kwargs = {"cursor": cursor} if cursor else {}
        return await app.client.usergroups_list(**kwargs)
    except SlackApiError as exc:
        response = getattr(exc, "response", None)
        error = (response.get("error") if response is not None else None) or ""
        if error == "ratelimited":
            headers = getattr(response, "headers", {}) or {}
            raise RateLimited(float(headers.get("Retry-After", 1))) from exc
        raise


async def _find_usergroup_id(handle):
    """Slack `usergroups.list` lookup of an @handle's group id (needs `usergroups:read`).

    Follows `response_metadata.next_cursor` until the handle is found or Slack
    has no further page. Same error contract as _fetch_usergroup_members.
    """
    return await group_id_for_handle_across_pages(_fetch_usergroup_page, handle)


async def _channel_info(channel_id):
    """Slack `conversations.info` payload, including `ok: false` error payloads."""
    try:
        return (await app.client.conversations_info(channel=channel_id)).data
    except SlackApiError as exc:
        return exc.response.data


# Expert identity: one Slack user group per product. Membership is cached
# (EXPERT_GROUP_CACHE_SECONDS, default 300s) so the per-message and per-reaction
# hot paths do not call Slack. See src/expert_group.py.
# A missing EXPERT_GROUP_ID_IDA / _IDDM raises here, at import, exactly like a
# missing SLACK_BOT_TOKEN: there is no fallback expert any more.
expert_resolver = ExpertResolver(
    group_ids=group_ids_from_env(lil_lisa_env),
    fetch_members=_fetch_usergroup_members,
    cache_seconds=cache_seconds_from_env(lil_lisa_env),
    find_group_id=_find_usergroup_id,
)
if not expert_resolver.is_configured("IDO"):
    logger.warning("No EXPERT_GROUP_ID_IDO configured — IDO expert support disabled")


async def _requests_call(func, *args, **kwargs):
    """Run a blocking `requests` call in a thread so the Bolt event loop stays free."""
    return await asyncio.to_thread(func, *args, **kwargs)


BOT_USER_ID: str = None
ESCALATE_ACTION_ID = "escalate_to_techsupport"
# Dictionary to track which message threads have already been endorsed or escalated
# Format: {conv_id: {"message_endorsed": True/False, "reaction_endorsed": "up"/"down"/False, "escalated": True/False}}
# Thumbs stay in process memory. Successful escalations are also persisted
# (escalation_tracker.json, 90-day TTL) so silence-after-escalate survives restart.
ENDORSEMENT_TRACKER: Dict[str, Dict[str, Any]] = {}
hydrate_endorsement_tracker(ENDORSEMENT_TRACKER)

ESCALATE_BUTTON_TEXT = "Post question in tech support channel"
ESCALATE_NOTE_TEXT = (
    "If this didn't help and your question is complete, post it to tech support."
)
ESCALATE_ALREADY_POSTED_TEXT = "Already posted to tech support."
TS_CHANNEL_MENTION_REPLY = "Ask in the relevant product channel, not here."
ESCALATION_BLOCK_IDS = frozenset({"escalation_note", "escalation_actions"})


def build_escalation_blocks(button_value: str, prominent: bool) -> List[Dict[str, Any]]:
    """
    Note+button block pair for the tech-support escalation action.

    Prominent (no-answer / disagreement-detected cases): bold section text + "primary"
    (colored) button, so it stands out.
    Subtle (answer-found case): a context block (Block Kit's native small/de-emphasized
    text) + default-styled button, so it fades into the background.
    """
    if prominent:
        note_block = {
            "type": "section",
            "block_id": "escalation_note",
            "text": {"type": "mrkdwn", "text": f"*{ESCALATE_NOTE_TEXT}*"},
        }
    else:
        note_block = {
            "type": "context",
            "block_id": "escalation_note",
            "elements": [{"type": "mrkdwn", "text": ESCALATE_NOTE_TEXT}],
        }

    button: Dict[str, Any] = {
        "type": "button",
        "text": {"type": "plain_text", "text": ESCALATE_BUTTON_TEXT},
        "action_id": ESCALATE_ACTION_ID,
        "value": button_value,
    }
    if prominent:
        button["style"] = "primary"

    return [
        note_block,
        {"type": "actions", "block_id": "escalation_actions", "elements": [button]},
    ]


def _message_has_escalation_blocks(blocks: List[Dict[str, Any]]) -> bool:
    return any(block.get("block_id") in ESCALATION_BLOCK_IDS for block in blocks)


def _blocks_without_escalation(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop escalate note/actions and append a short already-posted marker."""
    stripped = [block for block in blocks if block.get("block_id") not in ESCALATION_BLOCK_IDS]
    stripped.append(
        {
            "type": "context",
            "block_id": "escalation_note",
            "elements": [{"type": "mrkdwn", "text": ESCALATE_ALREADY_POSTED_TEXT}],
        }
    )
    return stripped


async def strip_escalation_blocks_from_thread(client, channel_id: str, thread_ts: str) -> None:
    """Remove escalate buttons from every bot reply in the thread.

    After a successful escalate (or a duplicate click when the lock already holds),
    leftover buttons on earlier/later bot messages are confusing: clicks no-op or
    look broken. Replace them with an "Already posted to tech support." note.
    """
    await ensure_bot_id()
    try:
        replies = await conversations_replies_with_retry(channel=channel_id, ts=thread_ts)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            f"[ESCALATE] Failed to fetch thread replies to strip escalate buttons "
            f"channel={channel_id!r} thread_ts={thread_ts!r}: {exc}"
        )
        return

    for msg in replies.get("messages", []):
        if msg.get("user") != BOT_USER_ID:
            continue
        blocks = msg.get("blocks") or []
        if not _message_has_escalation_blocks(blocks):
            continue
        updated_blocks = _blocks_without_escalation(blocks)
        try:
            await client.chat_update(
                channel=channel_id,
                ts=msg["ts"],
                text=msg.get("text", "") or ESCALATE_ALREADY_POSTED_TEXT,
                blocks=updated_blocks,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(
                f"[ESCALATE] Failed to strip escalate blocks from message "
                f"channel={channel_id!r} ts={msg.get('ts')!r}: {exc}"
            )


def get_last_bot_message(messages, current_ts, skip_text="Processing..."):
    """
    Helper function to find the last bot message before the current timestamp.
    
    Args:
        messages: List of messages from conversations.replies
        current_ts: Current message timestamp to find preceding message
        skip_text: Text to skip when looking for bot messages
    
    Returns:
        The last bot message dict or None if not found
    """
    # Convert timestamps to floats for comparison
    current_ts_float = float(current_ts)
    
    # Filter and sort messages by timestamp
    bot_messages = []
    for msg in messages:
        msg_ts = float(msg.get("ts", "0"))
        if (msg_ts < current_ts_float and 
            msg.get("bot_id") and 
            msg.get("text", "").strip() != skip_text):
            bot_messages.append(msg)
    
    # Return the most recent bot message
    if bot_messages:
        return max(bot_messages, key=lambda x: float(x.get("ts", "0")))
    return None

def is_expert_upvote(reactor_is_expert, emoji, message_text):
    """
    Helper function to check if this is an expert upvote on a non-chunk message.

    Args:
        reactor_is_expert: Whether the reacting user is an expert for this
            channel's product (resolved via expert_resolver.is_expert()).
        emoji: The emoji (👍 or 👎)
        message_text: Text of the message being endorsed

    Returns:
        Boolean indicating if this is an expert upvote on a full answer
    """
    return (bool(reactor_is_expert) and
            emoji == "👍" and
            not message_text.startswith("*Chunk "))

def parse_chunk_message(text):
    """
    Helper function to parse chunk message and extract index, content, and URL.
    
    Args:
        text: The chunk message text starting with "*Chunk X*"
    
    Returns:
        Tuple of (chunk_index, chunk_content, webportal_url)
    """
    # Extract chunk index from header (e.g., "*Chunk 3*" -> 2 for 0-based)
    try:
        chunk_header_end = text.find("*", 7) + 1
        chunk_header = text[:chunk_header_end]
        chunk_index = int(chunk_header.split()[1].rstrip("*")) - 1  # Convert to 0-based index
    except (IndexError, ValueError):
        chunk_index = 0
    
    # Extract chunk content (everything after the header)
    chunk_content = text[chunk_header_end:].strip()
    
    # Split by newlines and find GitHub URL
    lines = chunk_content.split('\n')
    chunk_url = ""
    webportal_line_index = -1

    # Look for webportal URL and extract it (handle both formats)
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if line_stripped.startswith("https://developer.radiantlogic.com") or line_stripped.startswith("<https://developer.radiantlogic.com"):
            # Extract URL from markdown format <https://...> or plain format
            if line_stripped.startswith("<") and line_stripped.endswith(">"):
                chunk_url = line_stripped[1:-1]  # Remove < and >
            else:
                chunk_url = line_stripped
            webportal_line_index = i
            break

    # Extract chunk text (everything except the webportal URL line)
    if webportal_line_index >= 0:
        # Remove the webportal URL line from the chunk text
        chunk_text_lines = lines[:webportal_line_index] + lines[webportal_line_index + 1:]
        chunk_text = '\n'.join(chunk_text_lines).strip()
    else:
        chunk_text = chunk_content
    
    return chunk_index, chunk_text, chunk_url

async def ensure_bot_id():
    """
    If BOT_USER_ID is not yet known, call auth_test() once and cache it.
    """
    global BOT_USER_ID
    if BOT_USER_ID is None:
        auth = await app.client.auth_test()
        BOT_USER_ID = auth.get("user_id")
        logger.info(f"[BOT_ID SET] BOT_USER_ID = {BOT_USER_ID}")

def truncate_message_with_url(text: str, webportal_url: str = "", header: str = "") -> str:
    """
    Truncate message text to fit within MAX_LENGTH while preserving WebPortal URL.
    
    Args:
        text: The main text content to potentially truncate
        webportal_url: The WebPortal URL that must be preserved
        header: Any header text (like "Chunk X")
    
    Returns:
        Formatted message that fits within the length limit
    """
    # Calculate the components that must be preserved
    url_part = f"\n{webportal_url}" if webportal_url else ""
    ellipsis = "..."
    
    # Calculate total fixed length (header + URL + ellipsis + newlines)
    fixed_length = len(header) + len(url_part) + len(ellipsis)
    
    # Calculate available space for the main text
    available_length = MAX_LENGTH - fixed_length
    
    # Ensure we have reasonable space for text (at least 100 characters)
    if available_length < 100:
        logger.warning(f"Very little space available for text content: {available_length} characters")
    
    # Truncate text if necessary
    if len(text) > available_length:
        truncated_text = text[:available_length] + ellipsis
        logger.info(f"Message truncated from {len(text)} to {len(truncated_text)} characters")
    else:
        truncated_text = text
    
    # Construct final message
    if webportal_url:
        message = f"{header}{truncated_text}{url_part}"
    else:
        message = f"{header}{truncated_text}"
    
    return message


def clamp_to_slack_block_limit(text: str) -> str:
    """
    Hard backstop against Slack's "must be less than 3001 characters" invalid_blocks error.

    truncate_message_with_url() already trims to MAX_LENGTH, but MAX_LENGTH is an env-configured
    value that isn't guaranteed to respect Slack's block limit. This clamps unconditionally so a
    misconfigured MAX_LENGTH (or an unusual edge case that slips past upstream caps) can't crash
    the send and leave the user staring at "Processing..." forever.
    """
    if len(text) <= SLACK_BLOCK_TEXT_LIMIT:
        return text
    return truncate_preserving_code_fences(text, SLACK_BLOCK_TEXT_LIMIT)


CONVERSATIONS_REPLIES_RETRY_DELAY_SECONDS = 5.0
CONVERSATIONS_REPLIES_MAX_RETRIES = 4


async def conversations_replies_with_retry(channel: str, ts: str):
    """conversations_replies() wrapped with a retry, specifically for
    "channel_not_found" -- a message posted an instant earlier can momentarily
    fail to resolve via this call before it's fully replicated on Slack's
    backend, even though the channel and message are both otherwise valid
    (confirmed via isolated reproduction with the same token/channel/ts
    succeeding when run a few seconds later). Only retries on this specific
    error code -- any other SlackApiError (real permission/channel issues)
    is raised immediately so it isn't masked.
    """
    last_exc = None
    for attempt in range(CONVERSATIONS_REPLIES_MAX_RETRIES + 1):
        try:
            return await app.client.conversations_replies(channel=channel, ts=ts)
        except SlackApiError as exc:
            if exc.response.get("error") != "channel_not_found":
                raise
            last_exc = exc
            if attempt < CONVERSATIONS_REPLIES_MAX_RETRIES:
                logger.info(
                    f"[RETRY conversations_replies] channel_not_found for channel={channel!r} ts={ts!r} "
                    f"(attempt {attempt + 1}/{CONVERSATIONS_REPLIES_MAX_RETRIES + 1}) -- "
                    f"retrying in {CONVERSATIONS_REPLIES_RETRY_DELAY_SECONDS}s"
                )
                await asyncio.sleep(CONVERSATIONS_REPLIES_RETRY_DELAY_SECONDS)

    logger.error(
        f"[RETRY conversations_replies] Exhausted retries -- channel_not_found persisted for "
        f"channel={channel!r} ts={ts!r}"
    )
    raise last_exc


@app.event("message")
async def handle_message_events(event, say):
    """
    Handles Slack message events.

    This asynchronous function is an event handler for Slack "message" events. It processes the event based on
    the amount of people in a specific thread or whether the bot was tagged with an '@'.

    Args:
        event (dict): The Slack message event object.
        say (function): A function used to send messages in Slack.
    """

    channel_id = event["channel"]
    thread_ts = event.get("thread_ts")
    message_ts = event.get("ts")
    conv_id = thread_ts or message_ts

    # Once a thread has been escalated to techsupport, the conversation has moved there --
    # stay completely silent on any further activity in the original thread (no LLM call,
    # no post). Checked here, before the conversations_replies fetch below (which exists only
    # to count participants for the "should we respond" decision), so an escalated thread
    # costs nothing beyond this dict lookup.
    if is_escalation_active(conv_id):
        logger.info(f"[ESCALATED SILENCE] Thread {conv_id} already escalated to techsupport, staying silent.")
        return

    logger.info(f"[DEBUG conversations_replies] channel_id={channel_id!r} conv_id={conv_id!r}")
    participants = set()
    try:
        replies = await conversations_replies_with_retry(channel=channel_id, ts=conv_id)
        for message in replies.data["messages"]:
            participants.add(message["user"])
            if len(participants) >= 3:
                break
    except Exception as exc:  # pylint: disable=broad-except
        # Participant count is only used for the "should we respond" heuristic below --
        # fail safe by assuming a low count (i.e. still respond) rather than crashing the
        # whole handler when Slack can't be reached for this.
        logger.error(f"[PARTICIPANT COUNT FALLBACK] conversations_replies failed for channel={channel_id!r} conv_id={conv_id!r}: {exc}")
        participants = set()

    # The techsupport channel is for human discussion: only respond when directly mentioned,
    # never based on participant count.
    if TECHSUPPORT_CHANNEL_ID and channel_id == TECHSUPPORT_CHANNEL_ID:
        if LIL_LISA_SLACK_USERID in event.get("text", ""):
            thread_ts = event.get("thread_ts") or event.get("ts")
            await say(text=TS_CHANNEL_MENTION_REPLY, thread_ts=thread_ts)
        return

    # Process message if bot was mentioned OR thread has < 3 participants
    if LIL_LISA_SLACK_USERID in event["text"] or len(participants) < 3:
        await process_msg(event, say)


async def get_ans(query, thread_id, msg_id, product, is_expert_answering):
    """Get the answer from the chain"""
    conv_id = None
    conv_id = thread_id or msg_id
    t0 = time.perf_counter()
    try:
        # Call the invoke API
        full_url = f"{BASE_URL}/invoke/"
        response = await _requests_call(requests.post, 
            full_url,
            params={
                "session_id": str(conv_id),  # pylint: disable=missing-timeout
                "locale": "en",
                "product": product,
                "nl_query": query,
                "is_expert_answering": is_expert_answering,
                "is_followup": bool(thread_id),

            },
            timeout=90,  # Increased timeout to 90
        )
    except (requests.exceptions.ReadTimeout, requests.exceptions.Timeout) as timeout_exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[GET_ANS] conv_id=%s http_status=timeout bytes=0 elapsed_ms=%.0f",
            conv_id, elapsed_ms,
        )
        logger.error(f"Request timed out: {timeout_exc}")
        return "The agent failed to generate an answer. Please try again in a new message thread. Frame clear queries using full sentence(s)"
    except Exception as exc:  # pylint: disable=broad-except
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[GET_ANS] conv_id=%s http_status=error bytes=0 elapsed_ms=%.0f",
            conv_id, elapsed_ms,
        )
        logger.error(f"An error occurred during the asynchronous call get_ans: {exc}")
        return f"Lil lisa Slack-An error occured: {exc}"

    elapsed_ms = (time.perf_counter() - t0) * 1000
    body = response.text
    logger.info(
        "[GET_ANS] conv_id=%s http_status=%s bytes=%s elapsed_ms=%.0f",
        conv_id, response.status_code, len(body), elapsed_ms,
    )
    logger.debug("[GET_ANS BODY] conv_id=%s post=%s poster=Lil-Lisa", conv_id, body)
    return body


async def add_expert_verified_qa_pair(item_ts, channel_id, expert_user_id):
    """
    Automatically add expert-verified question-answer pair to golden QA pairs and DM expert.
    Extracts the actual thumbs-upped message and finds the corresponding question from Slack.
    
    Args:
        item_ts: The timestamp of the message that received thumbs up
        channel_id: The channel where the endorsement happened
        expert_user_id: The expert who gave the thumbs up — recorded with the QA
            pair and DMed the confirmation (not a configured single expert ID)
    """
    try:
        # Get the actual message that was thumbs-upped
        resp = await conversations_replies_with_retry(ts=item_ts, channel=channel_id)
        thumbs_up_message = resp["messages"][0]

        # Get the full thread to find the original question
        thread_ts = thumbs_up_message.get("thread_ts") or thumbs_up_message.get("ts")
        thread_resp = await conversations_replies_with_retry(ts=thread_ts, channel=channel_id)
        thread_messages = thread_resp["messages"]
        
        # Find the original user question and the thumbs-upped answer
        user_question = None
        expert_answer = thumbs_up_message.get("text", "").strip()
        
        # Look for the first non-bot message in the thread (original question)
        # Skip bot messages and processing messages
        for message in thread_messages:
            msg_user = message.get("user")
            msg_text = message.get("text", "").strip()
            
            # Skip bot messages, processing messages, and empty messages
            if (msg_user != BOT_USER_ID and 
                msg_text and 
                msg_text.lower() != "processing..." and
                not msg_text.startswith("*Chunk ")):
                user_question = msg_text
                break
        
        # Clean up the question (remove bot mentions)
        if user_question:
            user_question = user_question.replace(f"<@{LIL_LISA_SLACK_USERID}>", "").strip()
            # Remove any leading "> " quoting
            text_items = user_question.split("> ")
            user_question = text_items[1] if len(text_items) == 2 else text_items[0]
        
        # Clean up the answer (remove any #answer prefix if present)
        if expert_answer.lower().startswith("#answer"):
            expert_answer = expert_answer[7:].lstrip()
        
        if not user_question or not expert_answer:
            logger.warning(f"Could not extract complete Q&A pair. Question: {bool(user_question)}, Answer: {bool(expert_answer)}")
            return
        
        # Determine product based on channel
        product = determine_product(channel_id)
        
        if not product:
            logger.warning(f"Could not determine product for channel {channel_id}")
            return
        
        # Call the new endpoint to add the QA pair
        full_url = f"{BASE_URL}/add_expert_qa_pair/"
        
        response = await _requests_call(requests.post, 
            full_url,
            json={
                "question": user_question,
                "answer": expert_answer,
                "product": product,
                "expert_user_id": expert_user_id,
                "channel_id": channel_id,
                "message_ts": item_ts
            },
            params={
                "encrypted_key": ENCRYPTED_AUTHENTICATION_KEY
            },
            timeout=90,
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                # DM the expert with the new QA pair
                direct_message_convo = await app.client.conversations_open(users=expert_user_id)
                dm_channel_id = direct_message_convo.data["channel"]["id"]

                # The server also pushes the pair to the golden QA pairs GitHub repo so it
                # survives the next rebuild of the index. Say so, or warn if that push failed.
                if result.get("pushed"):
                    persistence_note = (
                        "This QA pair has been automatically added to the golden QA database and pushed to the "
                        "QA pairs repo, so it will survive the next golden QA rebuild."
                    )
                else:
                    persistence_note = (
                        "⚠️ This QA pair was added to the live index, but it could NOT be pushed to the QA pairs "
                        "repo — it will be lost on the next golden QA rebuild. Please tell the admins."
                    )

                qa_pair_text = f"""🎉 **New Expert-Verified QA Pair Added!**

**Question:** 
{user_question}

**Answer:** 
{expert_answer}

**Product:** {product}

{persistence_note} Please consider updating the official documentation with this verified information.

*Format for official documentation:*
```
Question: {user_question}

Answer: {expert_answer}
```"""
                
                await app.client.chat_postMessage(
                    channel=dm_channel_id,
                    text=qa_pair_text
                )
                logger.info(f"Successfully added expert QA pair and notified expert {expert_user_id}")
            else:
                logger.warning(f"Failed to add QA pair: {result.get('message', 'Unknown error')}")
        else:
            logger.error(f"Failed to add expert QA pair: HTTP {response.status_code} - {response.text}")
            
    except Exception as exc:
        logger.error(f"Error in add_expert_verified_qa_pair: {exc}")


async def record_endorsement(conv_id, is_expert, thumbs_up, endorsement_type="response", query_id=None, chunk_index=None, chunk_text=None, chunk_url=None):
    """Record feedback given to a bot response"""
    try:
        # Call the record_endorsement API
        full_url = f"{BASE_URL}/record_endorsement/"
        
        # Common parameters
        params = {
            "session_id": str(conv_id),
            "is_expert": is_expert,
            "thumbs_up": thumbs_up,
            "endorsement_type": endorsement_type
        }
        if query_id is not None:
            params["query_id"] = query_id
            
        # For chunk-specific endorsements, send large text data in the body
        if endorsement_type == "chunks" and chunk_text is not None:
            # Add chunk-specific parameter
            if chunk_index is not None:
                params["chunk_index"] = chunk_index
                
            # Send chunk text and URL in the request body
            json_data = {
                "chunk_text": chunk_text,
                "chunk_url": chunk_url or ""
            }
            
            # Use json parameter for request body
            await _requests_call(requests.post, 
                full_url,
                params=params,
                json=json_data,
                timeout=60,
            )
        else:
            # Regular response endorsement - all parameters already in params
            await _requests_call(requests.post, 
                full_url,
                params=params,
                timeout=60,
            )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"An error occurred during the asynchronous call record_endorsement: {exc}")
        return "An error occured"

async def tag_techsupport_thread(thread_ts: str, related_entry_title: str) -> None:
    """Tell LilLisa_Server that a newly-created techsupport thread relates to an existing
    verified entry, so the nightly pipeline can merge new insight into it instead of adding
    a duplicate. Best-effort: failures are logged and swallowed, never block escalation."""
    try:
        full_url = f"{BASE_URL}/tag_techsupport_thread/"
        response = await _requests_call(requests.post, 
            full_url,
            params={
                "thread_ts": thread_ts,
                "related_entry_title": related_entry_title,
                "encrypted_key": ENCRYPTED_AUTHENTICATION_KEY,
            },
            timeout=60,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        body = (exc.response.text or "")[:500] if exc.response is not None else ""
        logger.error(
            "An error occurred during the asynchronous call tag_techsupport_thread: HTTP %s body=%r",
            status,
            body,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"An error occurred during the asynchronous call tag_techsupport_thread: {exc}")

async def fetch_conversation_history(session_id) -> str:
    """Fetch the full conversation history for a session from the LilLisa_Server."""
    full_url = f"{BASE_URL}/get_conversation_history/"
    response = await _requests_call(requests.get, 
        full_url,
        params={
            "session_id": str(session_id),
            "encrypted_key": ENCRYPTED_AUTHENTICATION_KEY,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.text


async def get_refined_escalation_query(conversation_history: str) -> str:
    """Ask the LilLisa_Server to combine a multi-message thread into one faithful question for escalation."""
    full_url = f"{BASE_URL}/refine_escalation_query/"
    response = await _requests_call(requests.post, 
        full_url,
        json={"conversation_history": conversation_history},
        params={"encrypted_key": ENCRYPTED_AUTHENTICATION_KEY},
        timeout=60,
    )
    response.raise_for_status()
    return response.text


async def _post_message_with_fallback(channel_id: str, thread_ts: str, text: str, blocks: List[Dict[str, Any]]):
    """
    Post a message with blocks, falling back to a plain hard-truncated text-only message if Slack
    rejects the blocks (e.g. invalid_blocks from an edge case that slipped past the upstream caps).

    Without this, a rejected send leaves the user's thread showing only "Processing..." forever.
    A partial/plain answer is far better than that silence.
    """
    try:
        return await app.client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=text,
            blocks=blocks,
            unfurl_links=False,
        )
    except SlackApiError:
        logger.exception(
            f"[BLOCKS SEND FAILED] channel={channel_id!r} thread_ts={thread_ts!r} "
            "falling back to plain truncated text message"
        )
        return await app.client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=clamp_to_slack_block_limit(text),
            unfurl_links=False,
        )


async def process_msg(event, say):
    """
    - Call get_ans(...) to get a JSON string like
      {"response": "...", "needs_escalation": ..., ...}.
    - Post only parsed["response"] into Slack (plus escalate UI when configured).
    """
    user_id = event["user"]
    channel_id = event["channel"]
    orig_thread_ts = event["ts"]

    # Let user know we are working
    _ = await say(channel=channel_id, text="Processing...", thread_ts=orig_thread_ts)

    text = event["text"]
    thread_ts = event.get("thread_ts")
    message_ts = event.get("ts")
    conv_id = thread_ts or message_ts

    product = determine_product(channel_id)
    if product is None:
        await say(
            channel=channel_id,
            text="I am unable to provide answers in this channel. Please refer to the appropriate channels.",
            thread_ts=orig_thread_ts,
        )
        return

    # Remove any leading "> " quoting
    text_items = text.split("> ")
    text = text_items[1] if len(text_items) == 2 else text_items[0]

    is_expert_answering = False
    if text.lower().startswith("#answer") and await expert_resolver.is_expert(user_id, product):
        text = text[7:].lstrip()
        is_expert_answering = True

    # 1) Call your FastAPI server and get raw JSON (or a plain error string on timeout).
    raw_result = await get_ans(text, thread_ts, message_ts, product, is_expert_answering)

    # 2) Parsing it as JSON. Timeout/exception paths from get_ans are plain strings.
    parsed = parse_get_ans_result(raw_result)

    # 3) Extract "response", "links_text", and whether this needs escalation
    bot_text = parsed.get("response", "").strip()
    links_text = parsed.get("links_text", "").strip()
    needs_escalation = parsed.get("needs_escalation", False)
    primary_techsupport_match_title = parsed.get("primary_techsupport_match_title")
    logger.debug(
        f"DEBUG_NO_ANSWER | parsed keys={list(parsed.keys())} "
        f"needs_escalation={needs_escalation!r} bot_text={bot_text!r}"
    )

    # 4) Truncate bot response if necessary and post it back into Slack
    truncated_bot_text = truncate_message_with_url(bot_text)
    # Independent hard backstop on the block's text field specifically (see clamp_to_slack_block_limit
    # docstring) -- MAX_LENGTH is not guaranteed to respect Slack's 3000-char block limit.
    block_text = clamp_to_slack_block_limit(truncated_bot_text)

    # Order: main answer -> doc links.
    base_blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": block_text}}]
    if links_text:
        base_blocks.append({"type": "divider"})
        base_blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": clamp_to_slack_block_limit(links_text)},
        })
    base_blocks.append({"type": "divider"})

    # Encode everything the button click handler will need to re-derive the original
    # question and where to reply — no techsupport thread is created yet. Always use conv_id
    # (the actual thread root) here, not orig_thread_ts (this specific message's own ts), so
    # every escalate button in a thread — no matter which bot reply it's attached to — tags
    # the same ENDORSEMENT_TRACKER key that the silence-after-escalation check looks up.
    button_value = build_escalation_button_value(
        text, channel_id, conv_id, conv_id, user_id,
        primary_techsupport_match_title=primary_techsupport_match_title,
    )
    logger.info(
        f"[ESCALATE BUTTON CREATE] conv_id={conv_id!r} thread_ts(event)={thread_ts!r} "
        f"message_ts={message_ts!r} orig_thread_ts={orig_thread_ts!r} query_chars={len(text)}"
    )
    logger.debug(
        f"[ESCALATE BUTTON CREATE] conv_id={conv_id!r} query={text[:ESCALATE_VALUE_QUERY_MAX_LENGTH]!r}"
    )

    techsupport_ready = bool(TECHSUPPORT_CHANNEL_ID)

    if needs_escalation:
        # Bot could not answer: prominent button+note when a TS channel is configured.
        blocks = [*base_blocks]
        if techsupport_ready:
            blocks.extend(build_escalation_blocks(button_value, prominent=True))
        await _post_message_with_fallback(
            channel_id=channel_id, thread_ts=orig_thread_ts, text=truncated_bot_text, blocks=blocks
        )
    elif is_expert_answering:
        # A human expert answered directly - no escalation prompt needed.
        await app.client.chat_postMessage(
            channel=channel_id,
            thread_ts=orig_thread_ts,
            text=truncated_bot_text,
        )
    else:
        # Normal successful AI answer: subtle button+note when a TS channel is configured.
        blocks = [*base_blocks]
        if techsupport_ready:
            blocks.extend(build_escalation_blocks(button_value, prominent=False))
        await _post_message_with_fallback(
            channel_id=channel_id, thread_ts=orig_thread_ts, text=truncated_bot_text, blocks=blocks
        )


def clear_escalation_claim(conv_id: str) -> None:
    """Undo check_and_update_endorsement(..., "escalated") after a failed tech-support post."""
    clear_escalation(conv_id)
    entry = ENDORSEMENT_TRACKER.get(conv_id)
    if not entry or not entry.get("escalated"):
        return
    entry["escalated"] = False
    logger.info(f"[ESCALATE ROLLBACK] Cleared escalated flag for thread {conv_id} after failed techsupport post")


async def _notify_escalate_click_failure(client, body, orig_channel_id, orig_thread_ts, message: str) -> None:
    """Tell the clicker the escalate action failed (ephemeral, then in-thread fallback)."""
    clicker = (body.get("user") or {}).get("id")
    if clicker and orig_channel_id:
        try:
            await client.chat_postEphemeral(
                channel=orig_channel_id,
                user=clicker,
                text=message,
                thread_ts=orig_thread_ts,
            )
            return
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"[ESCALATE] Failed to post ephemeral error: {exc}")
    if not orig_channel_id or not orig_thread_ts:
        return
    try:
        await client.chat_postMessage(
            channel=orig_channel_id,
            thread_ts=orig_thread_ts,
            text=message,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"[ESCALATE] Failed to post in-thread escalate error: {exc}")


@app.action(ESCALATE_ACTION_ID)
async def handle_escalate_to_techsupport(ack, body, client):
    """
    Handles clicks on the "Post question in tech support channel" button.

    Only at this point (never earlier, e.g. when the bot decides needs_escalation) do we
    actually create a techsupport thread. Parses the query/channel_id/thread_ts encoded in
    the button's value, posts the escalation into the appropriate techsupport channel,
    replaces the button with a short confirmation (so it can't be clicked again), and posts
    a link-preview-free confirmation reply with a hyperlink to the new thread.

    The escalated lock is claimed before history/refine I/O so a double-click cannot
    post twice. If the tech-support post fails, the lock is rolled back so the user
    can click again.
    """
    await ack()

    action = body["actions"][0]
    try:
        payload = json.loads(action["value"])
    except (KeyError, json.JSONDecodeError) as exc:
        logger.error(f"[ESCALATE] Failed to parse button value: {exc}")
        return

    query = payload.get("query", "")
    orig_channel_id = payload.get("channel_id")
    orig_thread_ts = payload.get("thread_ts")
    session_id = payload.get("session_id")
    asker_user_id = payload.get("user_id")
    related_entry_title = payload.get("primary_techsupport_match_title")
    clicker_user_id = (body.get("user") or {}).get("id")

    logger.info(
        f"[ESCALATE CLICK] session_id={session_id!r} query_chars={len(query)} "
        f"thread_ts={orig_thread_ts!r} channel_id={orig_channel_id!r}"
    )
    logger.debug(
        f"[ESCALATE CLICK] payload={payload!r} session_id={session_id!r} query={query!r}"
    )

    if not orig_channel_id or not orig_thread_ts:
        logger.error(f"[ESCALATE] Missing channel_id/thread_ts in button value: {payload}")
        return

    warn_if_escalate_body_channel_mismatch(body, orig_channel_id)

    techsupport_channel_id = TECHSUPPORT_CHANNEL_ID
    if not techsupport_channel_id:
        logger.warning(f"[ESCALATE] TECHSUPPORT_CHANNEL_ID is not set (channel {orig_channel_id})")
        await _notify_escalate_click_failure(
            client,
            body,
            orig_channel_id,
            orig_thread_ts,
            "Tech support escalation is not configured. Please contact an administrator.",
        )
        return

    # Claim the lock before slow I/O so two clicks cannot both refine and both post.
    if not check_and_update_endorsement(orig_thread_ts, "escalated"):
        logger.info(f"[ESCALATE DUPLICATE] Thread {orig_thread_ts} already escalated, skipping.")
        # Still strip leftover buttons on other bot replies so duplicate clicks aren't silent no-ops.
        await strip_escalation_blocks_from_thread(client, orig_channel_id, orig_thread_ts)
        return

    # Refine the query only when there was more than one user message in the thread; a single
    # message is posted as-is since there's nothing to combine.
    escalation_query = query
    escalation_query_source = "raw button query (no session_id)"
    if session_id:
        try:
            conversation_history = await fetch_conversation_history(session_id)
            logger.info(
                f"[ESCALATE HISTORY] session_id={session_id!r} "
                f"history_chars={len(conversation_history)}"
            )
            logger.debug(
                f"[ESCALATE HISTORY] session_id={session_id!r} conversation_history={conversation_history!r}"
            )
            user_message_count = sum(
                1 for line in conversation_history.splitlines() if line.startswith("User:")
            )
            logger.info(f"[ESCALATE HISTORY] session_id={session_id!r} user_message_count={user_message_count}")

            if user_message_count > 1:
                refined_query = await get_refined_escalation_query(conversation_history)
                logger.info(
                    f"[ESCALATE REFINE] session_id={session_id!r} called=True "
                    f"refined_chars={len(refined_query)}"
                )
                logger.debug(
                    f"[ESCALATE REFINE] session_id={session_id!r} called=True refined_query={refined_query!r}"
                )
                if refined_query.strip():
                    escalation_query = refined_query.strip()
                    escalation_query_source = "refined_query"
                else:
                    escalation_query_source = "raw button query (refined_query was empty)"
            else:
                logger.info(
                    f"[ESCALATE REFINE] session_id={session_id!r} called=False reason=user_message_count<=1"
                )
                escalation_query_source = "raw button query (single user message, refinement skipped)"
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"[ESCALATE] Failed to refine escalation query, falling back to original: {exc}")
            escalation_query_source = f"raw button query (refinement error: {exc})"

    logger.info(
        f"[ESCALATE FINAL] session_id={session_id!r} query_chars={len(escalation_query)} "
        f"source={escalation_query_source!r}"
    )
    logger.debug(
        f"[ESCALATE FINAL] session_id={session_id!r} escalation_query={escalation_query!r} "
        f"source={escalation_query_source!r}"
    )

    # Permalink to the original question, included for context in the techsupport channel
    orig_permalink = None
    try:
        orig_permalink_resp = await client.chat_getPermalink(channel=orig_channel_id, message_ts=orig_thread_ts)
        orig_permalink = orig_permalink_resp.get("permalink")
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"[ESCALATE] Failed to get permalink to original thread: {exc}")

    # Name of the originating channel (e.g. "lil-lisa"), used as the link text below.
    orig_channel_name = None
    try:
        orig_channel_info = await client.conversations_info(channel=orig_channel_id)
        orig_channel_name = orig_channel_info.get("channel", {}).get("name")
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"[ESCALATE] Failed to get channel info for {orig_channel_id}: {exc}")

    channel_link_text = orig_channel_name or "here"
    channel_ref = f"<{orig_permalink}|{channel_link_text}>" if orig_permalink else channel_link_text
    asker_ref = f"<@{asker_user_id}>" if asker_user_id else "Someone"
    clicker_ref = f"<@{clicker_user_id}>" if clicker_user_id else "someone"

    techsupport_note = (
        f"Question posted in {channel_ref}. {asker_ref} dissatisfied with answer and requests help. "
        f"Clicked by {clicker_ref}."
    )
    techsupport_fallback_text = (
        f"Question posted in {orig_channel_name or orig_channel_id}. {escalation_query}"
    )

    try:
        techsupport_msg = await client.chat_postMessage(
            channel=techsupport_channel_id,
            text=techsupport_fallback_text,
            unfurl_links=False,
            blocks=[
                {"type": "context", "elements": [{"type": "mrkdwn", "text": techsupport_note}]},
                {"type": "section", "text": {"type": "mrkdwn", "text": escalation_query}},
            ],
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"[ESCALATE] Failed to post to techsupport channel {techsupport_channel_id}: {exc}")
        clear_escalation_claim(orig_thread_ts)
        await _notify_escalate_click_failure(
            client,
            body,
            orig_channel_id,
            orig_thread_ts,
            "Could not post this question to tech support. Please try the button again, or contact an administrator.",
        )
        return

    if related_entry_title:
        try:
            await tag_techsupport_thread(techsupport_msg["ts"], related_entry_title)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"[ESCALATE] Failed to tag techsupport thread {techsupport_msg['ts']}: {exc}")

    # Permalink to the newly-posted techsupport message, included in the confirmation reply below.
    techsupport_permalink = None
    try:
        techsupport_permalink_resp = await client.chat_getPermalink(
            channel=techsupport_channel_id, message_ts=techsupport_msg["ts"]
        )
        techsupport_permalink = techsupport_permalink_resp.get("permalink")
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"[ESCALATE] Failed to get permalink to techsupport message: {exc}")

    # Short confirmation with the link embedded as a hyperlink (not a raw pasted URL), and
    # with previews disabled - chat.postMessage supports unfurl_links, chat.update does not,
    # which is why this is a new reply rather than baked into the block-strip update below.
    confirmation_text = (
        f"Refer to question posted in <{techsupport_permalink}|tech support>"
        if techsupport_permalink
        else "Refer to question posted in tech support."
    )
    try:
        await client.chat_postMessage(
            channel=orig_channel_id,
            thread_ts=orig_thread_ts,
            text=confirmation_text,
            unfurl_links=False,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"[ESCALATE] Failed to post confirmation reply: {exc}")

    # Strip escalate buttons from every bot reply in the thread (not just the clicked message).
    await strip_escalation_blocks_from_thread(client, orig_channel_id, orig_thread_ts)


@app.event("reaction_added")
async def reaction(event, say):
    """
    1) Ensure we know our BOT_USER_ID.
    2) Handle reactions from experts to ANY message in the thread.
    3) On 👍 from expert, add to golden QA pairs and record endorsement.
    4) On 👍/👎 to bot answers, record endorsement (do not dump retrieved chunks).
    5) Handle chunk-specific reactions on leftover historical "*Chunk N*" messages.
    """
    await ensure_bot_id()

    channel_id = event["item"]["channel"]
    item_ts = event["item"]["ts"]
    item_user = event["item_user"]
    reactor_user_id = event["user"]

    # Determine product for this channel
    product = determine_product(channel_id)
    if not product:
        logger.info(f"[IGNORE] Unknown channel {channel_id}, skipping reaction.")
        return

    # Check if the reactor is an expert (a member of the product's expert user group).
    # An unreadable expert group raises ExpertLookupError out of this handler on
    # purpose: Bolt logs it, and a silent "nobody is an expert" is worse.
    is_expert = await expert_resolver.is_expert(reactor_user_id, product)

    # 1) Determine if this is a 👍 or a 👎
    reaction_name = event["reaction"]
    thumbs_up = None
    if reaction_name.startswith("+1"):
        thumbs_up = True
    elif reaction_name.startswith("-1"):
        thumbs_up = False

    # Get conversation details
    resp = await conversations_replies_with_retry(ts=item_ts, channel=channel_id)
    first_msg = resp["messages"][0]
    conv_id = first_msg.get("thread_ts") or first_msg.get("ts")

    # Check if this thread has already been endorsed (reactions allow changing thumbs up/down)
    if not check_and_update_endorsement(conv_id, "endorsed", "reaction", thumbs_up):
        logger.info(f"[REACTION DUPLICATE] Thread {conv_id} already endorsed, skipping.")
        return

    # 2) If expert gives thumbs up to ANY message (except chunks), add to golden QA pairs
    if thumbs_up is True and is_expert:
        # Get the message text to check if this is a chunk message
        message_text = first_msg.get("text", "")
        
        # Use helper function to check if this is an expert upvote on a non-chunk
        if is_expert_upvote(is_expert, "👍", message_text):
            logger.info(f"[EXPERT THUMBS UP] Expert {reactor_user_id} gave thumbs up to message {item_ts}")
            # DM the expert who actually reacted, not a configured single ID.
            await add_expert_verified_qa_pair(item_ts, channel_id, reactor_user_id)
        else:
            logger.info(f"[EXPERT THUMBS UP CHUNK] Expert {reactor_user_id} gave thumbs up to chunk {item_ts} - skipping golden QA pair addition")

    # 3) Handle bot-specific message reactions (for endorsement recording and chunk display)
    if item_user == BOT_USER_ID:
        # Get the message text to determine if this is a chunk message
        message_text = first_msg.get("text", "")
        
        # 4) Check if this is a chunk message (starts with "*Chunk X*")
        if message_text.startswith("*Chunk "):
            logger.info(f"[REACTION CHUNK] This is a chunk message, allowing reaction without duplication check")
            # Reset the endorsement tracker for this thread since we want to allow chunk endorsements
            if conv_id in ENDORSEMENT_TRACKER:
                ENDORSEMENT_TRACKER[conv_id]["reaction_endorsed"] = False
            
            # Use helper function to parse chunk information
            chunk_index, chunk_text, chunk_url = parse_chunk_message(message_text)

            # Record chunk-specific endorsement
            await record_endorsement(
                conv_id=conv_id,
                is_expert=is_expert,
                thumbs_up=thumbs_up,
                endorsement_type="chunks",
                chunk_index=chunk_index,
                chunk_text=chunk_text,
                chunk_url=chunk_url
            )
            
            return

        # 5) This is a regular bot response message
        # On 👍/👎: record endorsement. Slack no longer dumps retrieved chunks;
        # users who need a human should use the tech-support escalate button.
        if thumbs_up is True or thumbs_up is False:
            await record_endorsement(conv_id, is_expert, thumbs_up, endorsement_type="response")
            await say(channel=channel_id, text="Thank you for your feedback!", thread_ts=conv_id)
            return


async def check_members(channel_id, user_id):
    """
    Check if a user is a member of a specific channel.

    This asynchronous function checks whether a user with the given user_id is a member of the specified channel.
    It uses the Slack API method 'conversations_members' to fetch the members of the channel.

    Args:
        channel_id (str): The ID of the channel to check for membership.
        user_id (str): The ID of the user whose membership status in the channel is to be checked.

    Returns:
        bool: True if the user is a member of the channel, False otherwise.

    Raises:
        Exception: If an unexpected error occurs while querying the Slack API.
                   Note: This function handles exceptions and returns False in case of errors to indicate
                   that the user's membership status could not be determined.
    """
    try:
        response = await app.client.conversations_members(channel=channel_id)
        members = response["members"]
        return user_id in members
    except Exception as exc:  # pylint:disable=broad-except
        print(f"Error occurred: {str(exc)}")


def determine_product(channel_id):
    """
    Determines the product string from a channel_id.

    Experts are a Slack user group per product, so there is no single "the
    expert" to return alongside it. To decide whether somebody is an expert,
    call `await expert_resolver.is_expert(user_id, product)`.

    Parameters:
    - channel_id (str): The ID of the channel to check.

    Returns:
    - str or None: the product for this channel, or None if it is not a
      product channel.
    """

    if channel_id == CHANNEL_ID_IDDM or channel_id == ADMIN_CHANNEL_ID_IDDM:
        product = "IDDM"
    elif channel_id == CHANNEL_ID_IDA or channel_id == ADMIN_CHANNEL_ID_IDA:
        product = "IDA"
    elif CHANNEL_ID_IDO and (channel_id == CHANNEL_ID_IDO or channel_id == ADMIN_CHANNEL_ID_IDO):
        product = "IDO"
    else:
        return None

    return product


def check_and_update_endorsement(conv_id, action_type="endorsed", endorsement_source="message", thumbs_up=None):
    """
    Check if a conversation has already been endorsed and update the tracker.

    Args:
        conv_id (str): The conversation ID (thread_ts) to check
        action_type (str): The action type, either "endorsed" or "escalated"
        endorsement_source (str): Either "message" or "reaction"
        thumbs_up (bool): For reactions, whether this is thumbs up (True) or thumbs down (False)

    Returns:
        bool: True if this endorsement should be processed, False if it should be skipped
    """
    # Initialize the tracking entry if it doesn't exist
    if conv_id not in ENDORSEMENT_TRACKER:
        ENDORSEMENT_TRACKER[conv_id] = {"message_endorsed": False, "reaction_endorsed": False, "escalated": False}
    elif "escalated" not in ENDORSEMENT_TRACKER[conv_id]:
        ENDORSEMENT_TRACKER[conv_id]["escalated"] = False

    # Handle techsupport escalation action. Persist before flipping RAM so a
    # crash after disk write still looks escalated after restart (duplicate
    # click is safer than a double post).
    if action_type == "escalated":
        if not claim_escalation(conv_id):
            logger.info(f"[DUPLICATE ESCALATED] Conv {conv_id} already has escalated=True")
            ENDORSEMENT_TRACKER[conv_id]["escalated"] = True
            return False
        return update_tracker_entry(conv_id, "escalated", True)

    # Handle endorsement actions
    elif action_type == "endorsed":
        if endorsement_source == "message":
            # For messages: block if ANY endorsement exists (message or reaction)
            if ENDORSEMENT_TRACKER[conv_id]["message_endorsed"] or ENDORSEMENT_TRACKER[conv_id]["reaction_endorsed"]:
                logger.info(f"[DUPLICATE MESSAGE] Conv {conv_id} already endorsed via message or reaction")
                return False
            return update_tracker_entry(conv_id, "message_endorsed", True)
            
        elif endorsement_source == "reaction":
            # For reactions: block if message endorsement exists, but allow changing reaction type
            if ENDORSEMENT_TRACKER[conv_id]["message_endorsed"]:
                logger.info(f"[DUPLICATE REACTION] Conv {conv_id} already endorsed via message")
                return False
            
            # Get current reaction state
            current_reaction = ENDORSEMENT_TRACKER[conv_id]["reaction_endorsed"]
            new_reaction = "up" if thumbs_up else "down"
            
            # If no reaction yet, or changing reaction type, allow it
            if not current_reaction or current_reaction != new_reaction:
                return update_tracker_entry(conv_id, "reaction_endorsed", new_reaction)
            else:
                # Same reaction type as before, skip
                logger.info(f"[DUPLICATE REACTION] Conv {conv_id} already has same reaction: {new_reaction}")
                return False
    
    return False

def update_tracker_entry(conv_id, field, value):
    """
    Update a specific field in the endorsement tracker and log the change.
    
    Args:
        conv_id (str): The conversation ID to update
        field (str): The field to update ('escalated', 'message_endorsed', or 'reaction_endorsed')
        value: The value to set (True/False or 'up'/'down' for reactions)

    Returns:
        bool: True if the field was updated, False if it was already set to the specified value
    """
    # Check if value already matches (no change needed)
    if ENDORSEMENT_TRACKER[conv_id][field] == value:
        action_type = field.replace("_endorsed", "").upper()
        logger.info(f"[DUPLICATE {action_type}] Conv {conv_id} already has {field}={value}")
        return False

    # Update the field with the new value
    ENDORSEMENT_TRACKER[conv_id][field] = value

    # Log the update based on field type
    if field == "escalated":
        logger.info(f"[NEW ESCALATION] Marking conv {conv_id} as escalated to techsupport")
    elif field == "message_endorsed":
        logger.info(f"[NEW MESSAGE ENDORSEMENT] Marking conv {conv_id} as message endorsed")
    elif field == "reaction_endorsed":
        logger.info(f"[NEW/CHANGED REACTION] Marking conv {conv_id} as reaction endorsed: {value}")

    return True


@app.command("/get_golden_qa_pairs")
async def get_golden_qa_pairs(ack, body, say):
    """
    Slack command to retrieve the golden qa pairs.

    This asynchronous function is a Slack slash command handler for "/get_golden_qa_pairs". It sends progress and success messages in the Slack channel.

    Args:
        ack (function): A function used to acknowledge the Slack command.
        body (dict): A dictionary containing the payload of the event, command, or action.
        say (function): A function used to send messages in Slack.

    """
    await ack()
    user_id = body.get("user_id")
    channel_id = body.get("channel_id")

    direct_message_convo = await app.client.conversations_open(users=user_id)
    dm_channel_id = direct_message_convo.data["channel"]["id"]
    contains_user = await check_members(ADMIN_CHANNEL_ID_IDDM, user_id) or await check_members(ADMIN_CHANNEL_ID_IDA, user_id) or (ADMIN_CHANNEL_ID_IDO and await check_members(ADMIN_CHANNEL_ID_IDO, user_id)) 
    if not contains_user:
        # Return an error message or handle unauthorized users
        await say(
            text="""Unauthorized! Please contact one of the admins (@nico/@Dhar Rawal) and ask for authorization. Once you are added to the appropriate admin Slack channel, you will be able to use '/' commands to manage rocknbot.""",
        )
        return

    product = determine_product(channel_id)

    if product is None:
        _ = await say(
            channel=dm_channel_id,
            text="I am unable to retrieve the golden QA pairs from this channel. Please go to the approriate channel and try the command again.",
        )
        return
    try:
        # Call the get_golden_qa_pairs API
        full_url = f"{BASE_URL}/get_golden_qa_pairs/"
        if response := await _requests_call(requests.post, 
            full_url,
            params={
                "product": product,  # pylint: disable=missing-timeout
                "encrypted_key": ENCRYPTED_AUTHENTICATION_KEY,
            },
            timeout=60,
        ):
            await app.client.files_upload_v2(
                file=response.content,
                filename="qa_pairs.md",
                channel=dm_channel_id,
                initial_comment="Here are the QA pairs you requested!",
            )
        else:
            error_msg = f"Call to lil-lisa server {full_url} has failed."
            logger.error(error_msg)
            return error_msg

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"An error occurred during the asynchronous call get_golden_qa_pairs: {exc}")
        return "An error occured"


@app.command("/update_golden_qa_pairs")
async def update_golden_qa_pairs(ack, body, say):    # pylint: disable=too-many-locals
    """
    Slack command to replace the existing golden qa pairs in the database.

    This asynchronous function is a Slack slash command handler for "/update_golden_qa_pairs". It sends progress and success messages in the Slack channel.

    Args:
        ack (function): A function used to acknowledge the Slack command.
        body (dict): A dictionary containing the payload of the event, command, or action.
        say (function): A function used to send messages in Slack.

    """
    await ack()
    user_id = body.get("user_id")
    channel_id = body.get("channel_id")
    direct_message_convo = await app.client.conversations_open(users=user_id)
    dm_channel_id = direct_message_convo.data["channel"]["id"]
    contains_user = await check_members(ADMIN_CHANNEL_ID_IDDM, user_id) or await check_members(ADMIN_CHANNEL_ID_IDA, user_id) or (ADMIN_CHANNEL_ID_IDO and await check_members(ADMIN_CHANNEL_ID_IDO, user_id))
    if not contains_user:
        # Return an error message or handle unauthorized users
        await say(
            text="""Unauthorized! Please contact one of the admins (@nico/@Dhar Rawal) and ask for authorization. Once you are added to the appropriate admin Slack channel, you will be able to use '/' commands to manage rocknbot.""",
        )
        return

    product = determine_product(channel_id)

    if not product:
        _ = await say(
            channel=dm_channel_id,
            text="I am unable to update the golden QA pairs from this channel. Please go to the approriate channel and try the command again.",
        )
        return
    try:
        # Call the update_golden_qa_pairs API
        full_url = f"{BASE_URL}/update_golden_qa_pairs/"
        if response := await _requests_call(requests.post, 
            full_url,
            params={
                "product": product,  # pylint: disable=missing-timeout
                "encrypted_key": ENCRYPTED_AUTHENTICATION_KEY,
            },
            timeout=60,
        ):
            _ = await say(channel=dm_channel_id, text=response.text)
        else:
            error_msg = f"Call to lil-lisa server {full_url} has failed."
            logger.error(error_msg)
            return error_msg

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"An error occurred during the asynchronous call update_golden_qa_pairs: {exc}")
        return "An error occured"


@app.command("/run_nightly_pipeline")
async def run_nightly_pipeline(ack, body, say):
    """Start the server's techsupport pipeline for an authorized administrator.

    The server accepts the request immediately and runs the work in a
    background task, so the bot can confirm promptly without waiting for Slack
    collection, classification, embedding, or the GitHub push to finish.
    """
    await ack()
    user_id = body.get("user_id")
    direct_message_convo = await app.client.conversations_open(users=user_id)
    dm_channel_id = direct_message_convo.data["channel"]["id"]
    contains_user = (
        await check_members(ADMIN_CHANNEL_ID_IDDM, user_id)
        or await check_members(ADMIN_CHANNEL_ID_IDA, user_id)
        or (ADMIN_CHANNEL_ID_IDO and await check_members(ADMIN_CHANNEL_ID_IDO, user_id))
    )
    if not contains_user:
        await say(
            text=(
                "Unauthorized! Please contact one of the admins (@nico/@Dhar Rawal) and ask for "
                "authorization. Once you are added to the appropriate admin Slack channel, you "
                "will be able to use '/' commands to manage rocknbot."
            ),
        )
        return

    full_url = f"{BASE_URL}/run_nightly_pipeline/"
    try:
        response = await _requests_call(
            requests.post,
            full_url,
            params={"encrypted_key": ENCRYPTED_AUTHENTICATION_KEY},
            timeout=60,
        )
        response.raise_for_status()
        await say(channel=dm_channel_id, text=response.text)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Unable to start nightly techsupport pipeline: %s", exc)
        await say(
            channel=dm_channel_id,
            text="Unable to start the nightly techsupport pipeline. Check the server logs and configuration.",
        )


# @app.command("/rebuild_docs")
# async def rebuild_docs(ack, body, say):
#     """
#     Slack command to rebuild the documentation database.

#     This asynchronous function is a Slack slash command handler for "/rebuild_docs". It sends progress and success messages in the Slack channel.

#     Args:
#         ack (function): A function used to acknowledge the Slack command.
#         body (dict): A dictionary containing the payload of the event, command, or action.
#         say (function): A function used to send messages in Slack.

#     """
#     await ack()
#     user_id = body.get("user_id")
#     channel_id = body.get("channel_id")

#     direct_message_convo = await app.client.conversations_open(users=user_id)
#     dm_channel_id = direct_message_convo.data["channel"]["id"]
#     contains_user = await check_members(ADMIN_CHANNEL_ID_IDDM, user_id) or await check_members(
#         ADMIN_CHANNEL_ID_IDA, user_id
#     )
#     if not contains_user:
#         # Return an error message or handle unauthorized users
#         await say(
#             text="""Unauthorized! Please contact one of the admins (@nico/@Dhar Rawal) and ask for authorization. Once you are added to the appropriate admin Slack channel, you will be able to use '/' commands to manage rocknbot.""",
#         )
#         return

#     product = determine_product(channel_id)

#     if product is None:
#         _ = await say(
#             channel=dm_channel_id,
#             text="I am unable to retrieve the golden QA pairs from this channel. Please go to the approriate channel and try the command again.",
#         )
#         return
#     _ = await say(
#         channel=dm_channel_id,
#         text="You are rebuilding the entire documentation database. This process will take approximately 15-20 minutes.",
#     )
#     try:
#         # Call the rebuild_docs API
#         full_url = f"{BASE_URL}/rebuild_docs/"
#         if response := await _requests_call(requests.post, 
#             full_url,
#             params={"encrypted_key": ENCRYPTED_AUTHENTICATION_KEY},
#             timeout=2700,
#         ):
#             _ = await say(channel=dm_channel_id, text=response.text)
#         else:
#             error_msg = f"Call to lil-lisa server {full_url} has failed."
#             logger.error(error_msg)
#             return error_msg

#     except Exception as exc:  # pylint: disable=broad-except
#         logger.error(f"An error occurred during the asynchronous call rebuild_docs(): {exc}")
#         return "An error occured"

@app.command("/rebuild_docs_traditional")
async def rebuild_docs_traditional(ack, body, say):
    """
    Slack command to rebuild the documentation database.

    This asynchronous function is a Slack slash command handler for "/rebuild_docs_traditional". It sends progress and success messages in the Slack channel.
    This functions rebuild the documnt with traditional chunking using openai text-embedding-large-3.

    Args:
        ack (function): A function used to acknowledge the Slack command.
        body (dict): A dictionary containing the payload of the event, command, or action.
        say (function): A function used to send messages in Slack.

    """
    await ack()
    user_id = body.get("user_id")
    channel_id = body.get("channel_id")

    direct_message_convo = await app.client.conversations_open(users=user_id)
    dm_channel_id = direct_message_convo.data["channel"]["id"]
    contains_user = await check_members(ADMIN_CHANNEL_ID_IDDM, user_id) or await check_members(ADMIN_CHANNEL_ID_IDA, user_id) or (ADMIN_CHANNEL_ID_IDO and await check_members(ADMIN_CHANNEL_ID_IDO, user_id))
    if not contains_user:
        # Return an error message or handle unauthorized users
        await say(
            text="""Unauthorized! Please contact one of the admins (@nico/@Dhar Rawal) and ask for authorization. Once you are added to the appropriate admin Slack channel, you will be able to use '/' commands to manage rocknbot.""",
        )
        return

    product = determine_product(channel_id)

    if product is None:
        _ = await say(
            channel=dm_channel_id,
            text="I am unable to retrieve the golden QA pairs from this channel. Please go to the approriate channel and try the command again.",
        )
        return
    _ = await say(
        channel=dm_channel_id,
        text="You are rebuilding the entire documentation database with traditional chunking. This process will take approximately 15-20 minutes.",
    )
    try:
        # Call the rebuild_docs_traditional API
        full_url = f"{BASE_URL}/rebuild_docs_traditional/"
        if response := await _requests_call(requests.post, 
            full_url,
            params={"encrypted_key": ENCRYPTED_AUTHENTICATION_KEY},
            timeout=2700,
        ):
            _ = await say(channel=dm_channel_id, text=response.text)
        else:
            error_msg = f"Call to lil-lisa server {full_url} has failed."
            logger.error(error_msg)
            return error_msg

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"An error occurred during the asynchronous call rebuild_docs(): {exc}")
        return "An error occured"

@app.command("/rebuild_docs_contextual")
async def rebuild_docs_contextual(ack, body, say):
    """
    Slack command to rebuild the documentation database.

    This asynchronous function is a Slack slash command handler for "/rebuild_docs_contextual". It sends progress and success messages in the Slack channel.
    This functions rebuild the documnt with contextual chunking using voyage voyage-context-3.

    Args:
        ack (function): A function used to acknowledge the Slack command.
        body (dict): A dictionary containing the payload of the event, command, or action.
        say (function): A function used to send messages in Slack.

    """
    await ack()
    user_id = body.get("user_id")
    channel_id = body.get("channel_id")

    direct_message_convo = await app.client.conversations_open(users=user_id)
    dm_channel_id = direct_message_convo.data["channel"]["id"]
    contains_user = await check_members(ADMIN_CHANNEL_ID_IDDM, user_id) or await check_members(ADMIN_CHANNEL_ID_IDA, user_id) or (ADMIN_CHANNEL_ID_IDO and await check_members(ADMIN_CHANNEL_ID_IDO, user_id))
    if not contains_user:
        # Return an error message or handle unauthorized users
        await say(
            text="""Unauthorized! Please contact one of the admins (@nico/@Dhar Rawal) and ask for authorization. Once you are added to the appropriate admin Slack channel, you will be able to use '/' commands to manage rocknbot.""",
        )
        return

    product = determine_product(channel_id)

    if product is None:
        _ = await say(
            channel=dm_channel_id,
            text="I am unable to retrieve the golden QA pairs from this channel. Please go to the approriate channel and try the command again.",
        )
        return
    _ = await say(
        channel=dm_channel_id,
        text="You are rebuilding the entire documentation database with contextual chunking. This process will take approximately 15-20 minutes.",
    )
    try:
        # Call the rebuild_docs_contextual API
        full_url = f"{BASE_URL}/rebuild_docs_contextual/"
        if response := await _requests_call(requests.post, 
            full_url,
            params={"encrypted_key": ENCRYPTED_AUTHENTICATION_KEY},
            timeout=2700,
        ):
            _ = await say(channel=dm_channel_id, text=response.text)
        else:
            error_msg = f"Call to lil-lisa server {full_url} has failed."
            logger.error(error_msg)
            return error_msg

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"An error occurred during the asynchronous call rebuild_docs(): {exc}")
        return "An error occured"

@app.command("/cleanup_sessions")
async def cleanup_sessions(ack, body, say):
    """
    Slack command to cleanup old sessions.
    
    This asynchronous function is a Slack slash command handler for "/cleanup_sessions".
    It deletes session folders older than the configured SESSION_LIFETIME_DAYS.
    
    Args:
        ack (function): A function used to acknowledge the Slack command.
        body (dict): A dictionary containing the payload of the event, command, or action.
        say (function): A function used to send messages in Slack.
    """
    await ack()
    user_id = body.get("user_id")
    
    # Open a direct message conversation with the user
    direct_message_convo = await app.client.conversations_open(users=user_id)
    dm_channel_id = direct_message_convo.data["channel"]["id"]
    
    # Check if user is authorized (in admin channels)
    contains_user = await check_members(ADMIN_CHANNEL_ID_IDDM, user_id) or await check_members(ADMIN_CHANNEL_ID_IDA, user_id) or (ADMIN_CHANNEL_ID_IDO and await check_members(ADMIN_CHANNEL_ID_IDO, user_id)) 
    if not contains_user:
        # Return an error message for unauthorized users
        await say(
            text="""Unauthorized! Please contact one of the admins (@nico/@Dhar Rawal) and ask for authorization. Once you are added to the appropriate admin Slack channel, you will be able to use '/' commands to manage rocknbot.""",
        )
        return
    
    # Send initial message
    initial_message = await app.client.chat_postMessage(
        channel=dm_channel_id,
        text="Session cleanup has started. This will remove old session data based on the configured retention period."
    )
    thread_ts = initial_message["ts"]
    
    try:
        # Call the cleanup_sessions API
        full_url = f"{BASE_URL}/cleanup_sessions/"
        response = await _requests_call(requests.post, 
            full_url,
            params={"encrypted_key": ENCRYPTED_AUTHENTICATION_KEY},
            timeout=60,
        )
        
        if response.status_code == 200:
            # Send success message in the same thread
            await app.client.chat_postMessage(
                channel=dm_channel_id,
                thread_ts=thread_ts,
                text=response.text
            )
        else:
            error_msg = f"Cleanup sessions failed with status code {response.status_code}: {response.text}"
            logger.error(error_msg)
            await app.client.chat_postMessage(
                channel=dm_channel_id,
                thread_ts=thread_ts,
                text=f"Error: {error_msg}"
            )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"An error occurred during cleanup_sessions: {exc}")
        await app.client.chat_postMessage(
            channel=dm_channel_id,
            thread_ts=thread_ts,
            text=f"An error occurred: {str(exc)}"
        )

async def start_slackapp():
    """
    Main asynchronous function to handle the Slack app's interaction with the Socket Mode.

    This function is the entry point for running the Slack app with Socket Mode.
    It creates an instance of `AsyncSocketModeHandler` with the provided app instance and the Slack app token.
    The app will listen for incoming events and handle them asynchronously.

    Note: Make sure the 'app' variable is already initialized with the appropriate Slack app configuration.

    Raises:
        Exception: If an unexpected error occurs while running the Slack app.
                   Note: This function handles exceptions but does not raise them further.
                   Any error encountered will be logged using the logger, allowing the app to continue running.

    Returns:
        None
    """
    try:
        await log_configured_channels(lil_lisa_env, _channel_info)
        handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
        await handler.start_async()
    except Exception as exc:  # pylint:disable=broad-except
        logger.error(f"Error: {exc}")


def main():
    """main function"""
    asyncio.run(start_slackapp())


if __name__ == "__main__":
    main()