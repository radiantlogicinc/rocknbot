"""
slack.py
====================================
The core module of rocknbot responsible
for catching and handling events from
Slack
"""

import asyncio
import os
from typing import Optional
import jwt


from typing import Dict, List
import json

import requests
from dotenv import dotenv_values
from slack_bolt.adapter.socket_mode.async_handler import (  # type: ignore
    AsyncSocketModeHandler,
)
from slack_bolt.async_app import AsyncApp  # type: ignore

from utils import logger

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
EXPERT_USER_ID_IDA = lil_lisa_env["EXPERT_USER_ID_IDA"]
EXPERT_USER_ID_IDDM = lil_lisa_env["EXPERT_USER_ID_IDDM"]
AUTHENTICATION_KEY = lil_lisa_env["AUTHENTICATION_KEY"]
ENCRYPTED_AUTHENTICATION_KEY = jwt.encode({"some": "payload"}, AUTHENTICATION_KEY, algorithm="HS256")  # type: ignore
MAX_LENGTH = int(lil_lisa_env["MAX_LENGTH"])

BASE_URL = os.getenv("LIL_LISA_SERVER_URL", lil_lisa_env["LIL_LISA_SERVER_URL"])
BASE_URL = BASE_URL.rstrip('/')     # this is IMPORTANT! Otherwise you will see {"detail": "Not Found"} in the response
logger.info(f"LIL_LISA_SERVER_URL: {BASE_URL}")

TIMEOUT = 10
app = AsyncApp(token=SLACK_BOT_TOKEN)


BOT_USER_ID: str = None
RERANK_CACHE: Dict[str, List[Dict[str, str]]] = {}

async def ensure_bot_id():
    """
    If BOT_USER_ID is not yet known, call auth_test() once and cache it.
    """
    global BOT_USER_ID
    if BOT_USER_ID is None:
        auth = await app.client.auth_test()
        BOT_USER_ID = auth.get("user_id")
        logger.info(f"[BOT_ID SET] BOT_USER_ID = {BOT_USER_ID}")

def truncate_message_with_url(text: str, github_url: str = "", header: str = "") -> str:
    """
    Truncate message text to fit within MAX_LENGTH while preserving GitHub URL.
    
    Args:
        text: The main text content to potentially truncate
        github_url: The GitHub URL that must be preserved
        header: Any header text (like "Chunk X")
    
    Returns:
        Formatted message that fits within the length limit
    """
    # Calculate the components that must be preserved
    url_part = f"\n{github_url}" if github_url else ""
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
    if github_url:
        message = f"{header}{truncated_text}{url_part}"
    else:
        message = f"{header}{truncated_text}"
    
    return message

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
    replies = await app.client.conversations_replies(channel=channel_id, ts=conv_id)
    participants = set()
    for message in replies.data["messages"]:
        participants.add(message["user"])
        if len(participants) >= 3:
            break

    # ADD COMMENT HERE
    if LIL_LISA_SLACK_USERID in event["text"] or len(participants) < 3:
        await process_msg(event, say)


async def get_ans(query, thread_id, msg_id, product, is_expert_answering):
    """Get the answer from the chain"""
    conv_id = None
    conv_id = thread_id or msg_id
    try:
        # Call the invoke API
        full_url = f"{BASE_URL}/invoke/"
        response = requests.post(
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
        logger.error(f"Request timed out: {timeout_exc}")
        return "The agent failed to generate an answer. Please try again in a new message thread. Frame clear queries using full sentence(s)"
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"An error occurred during the asynchronous call get_ans: {exc}")
        return f"Lil lisa Slack-An error occured: {exc}"

    conv_dict = {"conv_id": conv_id, "post": response.text, "poster": "Lil-Lisa"}
    logger.info(str(conv_dict))
    return response.text


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
            requests.post(
                full_url,
                params=params,
                json=json_data,
                timeout=60,
            )
        else:
            # Regular response endorsement - all parameters already in params
            requests.post(
                full_url,
                params=params,
                timeout=60,
            )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"An error occurred during the asynchronous call record_endorsement: {exc}")
        return "An error occured"

async def process_msg(event, say):
    """
    - Call get_ans(...) to get a JSON string like
      {"response": "...", "reranked_nodes":[{"text":...}, … ]}.
    - Post only parsed["response"] into Slack.
    - Cache parsed["reranked_nodes"] under the new message's ts.
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

    product, expert_user_id = determine_product_and_expert(channel_id)
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
    if user_id == expert_user_id and text.lower().startswith("#answer"):
        text = text[7:].lstrip()
        is_expert_answering = True

    # 1) Call your FastAPI server and get raw JSON
    raw_result = await get_ans(text, thread_ts, message_ts, product, is_expert_answering)

    # 2) Parsing it as JSON.
    parsed = json.loads(raw_result)

    # 3) Extract "response" and "reranked_nodes"
    bot_text = parsed.get("response", "").strip()
    reranked_nodes = parsed.get("reranked_nodes", [])

    # 4) Truncate bot response if necessary and post it back into Slack
    truncated_bot_text = truncate_message_with_url(bot_text)
    post = await app.client.chat_postMessage(
        channel=channel_id,
        thread_ts=orig_thread_ts,
        text=truncated_bot_text,
    )

    # 5) Cache reranked_nodes under the bot‐message's ts
    bot_ts = post["ts"]  # Slack returns a string here
    if isinstance(reranked_nodes, list) and reranked_nodes:
        RERANK_CACHE[bot_ts] = reranked_nodes
        logger.info(f"[CACHE SET] {len(reranked_nodes)} nodes under ts={bot_ts}")
    else:
        logger.info(f"[CACHE] No reranked_nodes for ts={bot_ts}")


@app.event("reaction_added")
async def reaction(event, say):
    """
    1) Ensure we know our BOT_USER_ID.
    2) Compare event['item_user'] to BOT_USER_ID (not the old env var).
    3) On 👍, record endorsement. On 👎, unpack reranked_nodes from cache.
    4) Handle chunk-specific reactions on individual chunk messages.
    """
    await ensure_bot_id()

    channel_id = event["item"]["channel"]
    item_ts = event["item"]["ts"]
    item_user = event["item_user"]

    # 1) Only proceed if the message was posted by bot
    if item_user != BOT_USER_ID:
        logger.info(f"[IGNORE] Message ts={item_ts} was not posted by our bot, skipping.")
        return

    # 2) Determine if this is a 👍 or a 👎
    reaction_name = event["reaction"]
    thumbs_up = None
    if reaction_name.startswith("+1"):
        thumbs_up = True
    elif reaction_name.startswith("-1"):
        thumbs_up = False

    # Get conversation details
    resp = await app.client.conversations_replies(ts=item_ts, channel=channel_id)
    first_msg = resp["messages"][0]
    conv_id = first_msg.get("thread_ts") or first_msg.get("ts")
    is_expert = (event["user"] == determine_product_and_expert(channel_id)[1])

    _, expert_user_id = determine_product_and_expert(channel_id)

    if event["reaction"].startswith("sos"):
        _ = await say(channel=channel_id, text=f"<@{expert_user_id}> Can you help?", thread_ts=conv_id)

    # Get the message text to determine if this is a chunk message
    message_text = first_msg.get("text", "")
    
    # 3) Check if this is a chunk message (starts with "*Chunk X*")
    if message_text.startswith("*Chunk ") and "*" in message_text[7:]:
        # This is a chunk message - extract chunk information
        chunk_header_end = message_text.find("*", 7) + 1
        chunk_header = message_text[:chunk_header_end]
        
        # Extract chunk index from header (e.g., "*Chunk 3*" -> 3)
        try:
            chunk_index = int(chunk_header.split()[1].rstrip("*")) - 1  # Convert to 0-based index
        except (IndexError, ValueError):
            chunk_index = 0
            
        # Extract chunk text (everything after the header)
        chunk_content = message_text[chunk_header_end:].strip()
        
        # Split by newlines and find GitHub URL
        lines = chunk_content.split('\n')
        chunk_url = ""
        github_line_index = -1
        
        # Look for GitHub URL and extract it (handle both formats)
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith("https://github.com") or line_stripped.startswith("<https://github.com"):
                # Extract URL from markdown format <https://...> or plain format
                if line_stripped.startswith("<") and line_stripped.endswith(">"):
                    chunk_url = line_stripped[1:-1]  # Remove < and >
                else:
                    chunk_url = line_stripped
                github_line_index = i
                break
        
        # Extract chunk text (everything except the GitHub URL line)
        if github_line_index >= 0:
            # Remove the GitHub URL line from the chunk text
            chunk_text_lines = lines[:github_line_index] + lines[github_line_index + 1:]
            chunk_text = '\n'.join(chunk_text_lines).strip()

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

    # 4) This is a regular bot response message
    # On 👍: record endorsement for the response
    if thumbs_up is True:
        await record_endorsement(conv_id, is_expert, thumbs_up, endorsement_type="response")
        await say(channel=channel_id, text="Thank you for your feedback!", thread_ts=conv_id)
        return

    # 5) On 👎: look up RERANK_CACHE[item_ts] and post each node
    if thumbs_up is False:
        # First record the negative endorsement for the response
        await record_endorsement(conv_id, is_expert, thumbs_up, endorsement_type="response")
        
        logger.info(f"[THUMBS-DOWN] item_ts={item_ts} → cache keys: {list(RERANK_CACHE.keys())}")
        reranked = RERANK_CACHE.get(item_ts)

        # Find the parent thread_ts so new messages go into that same thread
        parent_thread = first_msg.get("thread_ts") or first_msg.get("ts")

        # Post each node as a separate message under parent_thread
        for idx, node in enumerate(reranked, start=1):
            # 1) Extract the core text
            chunk_text = node.get("text", "").strip()
            if not chunk_text:
                continue

            # 2) Pull out only the GitHub URL from metadata (if any)
            metadata = node.get("metadata", {})
            github_url = metadata.get("github_url", "").strip()

            # 3) Create header and use the new truncation function
            header = f"*Chunk {idx}*\n"
            message = truncate_message_with_url(chunk_text, github_url, header)

            # 4) Post that single, concise message to the same thread
            logger.info(f"[POST NODE {idx}] thread={parent_thread}, message_length={len(message)}")
            await app.client.chat_postMessage(
                channel=channel_id,
                thread_ts=parent_thread,
                text=message
            )

        # Clear that cache entry so no repost it on another 👎
        del RERANK_CACHE[item_ts]

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


def determine_product_and_expert(channel_id):
    """
    Determines the product string and expert user ID based on the given channel_id.

    Parameters:
    - channel_id (str): The ID of the channel to check.

    Returns:
    - tuple: A tuple containing the product string and the expert user ID.
    """

    if channel_id in CHANNEL_ID_IDDM or channel_id in ADMIN_CHANNEL_ID_IDDM:
        product = "IDDM"
        expert_user_id = EXPERT_USER_ID_IDDM
    elif channel_id in CHANNEL_ID_IDA or channel_id in ADMIN_CHANNEL_ID_IDA:
        product = "IDA"
        expert_user_id = EXPERT_USER_ID_IDA
    else:
        product = None
        expert_user_id = None

    return product, expert_user_id


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
    contains_user = await check_members(ADMIN_CHANNEL_ID_IDDM, user_id) or await check_members(
        ADMIN_CHANNEL_ID_IDA, user_id
    )
    if not contains_user:
        # Return an error message or handle unauthorized users
        await say(
            text="""Unauthorized! Please contact one of the admins (@nico/@Dhar Rawal) and ask for authorization. Once you are added to the appropriate admin Slack channel, you will be able to use '/' commands to manage rocknbot.""",
        )
        return

    product, _ = determine_product_and_expert(channel_id)

    if product is None:
        _ = await say(
            channel=dm_channel_id,
            text="I am unable to retrieve the golden QA pairs from this channel. Please go to the approriate channel and try the command again.",
        )
        return
    try:
        # Call the get_golden_qa_pairs API
        full_url = f"{BASE_URL}/get_golden_qa_pairs/"
        if response := requests.post(
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
    contains_user = await check_members(ADMIN_CHANNEL_ID_IDDM, user_id) or await check_members(ADMIN_CHANNEL_ID_IDA, user_id)
    if not contains_user:
        # Return an error message or handle unauthorized users
        await say(
            text="""Unauthorized! Please contact one of the admins (@nico/@Dhar Rawal) and ask for authorization. Once you are added to the appropriate admin Slack channel, you will be able to use '/' commands to manage rocknbot.""",
        )
        return

    product, _ = determine_product_and_expert(channel_id)

    if not product:
        _ = await say(
            channel=dm_channel_id,
            text="I am unable to update the golden QA pairs from this channel. Please go to the approriate channel and try the command again.",
        )
        return
    try:
        # Call the update_golden_qa_pairs API
        full_url = f"{BASE_URL}/update_golden_qa_pairs/"
        if response := requests.post(
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

#     product, _ = determine_product_and_expert(channel_id)

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
#         if response := requests.post(
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
    contains_user = await check_members(ADMIN_CHANNEL_ID_IDDM, user_id) or await check_members(
        ADMIN_CHANNEL_ID_IDA, user_id
    )
    if not contains_user:
        # Return an error message or handle unauthorized users
        await say(
            text="""Unauthorized! Please contact one of the admins (@nico/@Dhar Rawal) and ask for authorization. Once you are added to the appropriate admin Slack channel, you will be able to use '/' commands to manage rocknbot.""",
        )
        return

    product, _ = determine_product_and_expert(channel_id)

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
        if response := requests.post(
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
    contains_user = await check_members(ADMIN_CHANNEL_ID_IDDM, user_id) or await check_members(
        ADMIN_CHANNEL_ID_IDA, user_id
    )
    if not contains_user:
        # Return an error message or handle unauthorized users
        await say(
            text="""Unauthorized! Please contact one of the admins (@nico/@Dhar Rawal) and ask for authorization. Once you are added to the appropriate admin Slack channel, you will be able to use '/' commands to manage rocknbot.""",
        )
        return

    product, _ = determine_product_and_expert(channel_id)

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
        if response := requests.post(
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
    contains_user = await check_members(ADMIN_CHANNEL_ID_IDDM, user_id) or await check_members(
        ADMIN_CHANNEL_ID_IDA, user_id
    )
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
        response = requests.post(
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
        handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
        await handler.start_async()
    except Exception as exc:  # pylint:disable=broad-except
        logger.error(f"Error: {exc}")


def main():
    """main function"""
    asyncio.run(start_slackapp())


if __name__ == "__main__":
    main()