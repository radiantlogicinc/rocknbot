"""
slack.py
====================================
The core module of rocknbot responsible
for catching and handling events from
Slack
"""

import asyncio
import os
import jwt

import requests
from dotenv import dotenv_values
from slack_bolt.adapter.socket_mode.async_handler import (  # type: ignore
    AsyncSocketModeHandler,
)
from slack_bolt.async_app import AsyncApp  # type: ignore

from utils import logger

lil_lisa_env = dotenv_values("./app_envfiles/lil-lisa.env")

LIL_LISA_SLACK_USERID = lil_lisa_env["LIL_LISA_SLACK_USERID"]
SLACK_BOT_TOKEN = lil_lisa_env["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = lil_lisa_env["SLACK_APP_TOKEN"]
CHANNEL_ID_IDDM = lil_lisa_env["CHANNEL_ID_IDDM"]
ADMIN_CHANNEL_ID_IDDM = lil_lisa_env["ADMIN_CHANNEL_ID_IDDM"]
CHANNEL_ID_IDA = lil_lisa_env["CHANNEL_ID_IDA"]
ADMIN_CHANNEL_ID_IDA = lil_lisa_env["ADMIN_CHANNEL_ID_IDA"]
EXPERT_USER_ID_IDA = lil_lisa_env["EXPERT_USER_ID_IDA"]
EXPERT_USER_ID_IDDM = lil_lisa_env["EXPERT_USER_ID_IDDM"]
SECRET_AUTHENTICATION_KEY = lil_lisa_env["AUTHENTICATION_KEY"]
ENCRYPTED_AUTHENTICATION_KEY = jwt.encode({"some": "payload"}, SECRET_AUTHENTICATION_KEY, algorithm="HS256")  # type: ignore

REMOTE_NAME = lil_lisa_env["REMOTE_NAME"]
BASE_URL = os.getenv("LIL_LISA_SERVER_URL", lil_lisa_env["LIL_LISA_SERVER_URL"])

TIMEOUT = 10
app = AsyncApp(token=SLACK_BOT_TOKEN)


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
            },
            timeout=60,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"An error occurred during the asynchronous call get_ans: {exc}")
        return "An error occured"

    conv_dict = {"conv_id": conv_id, "post": response.text, "poster": "Lil-Lisa"}
    logger.info(str(conv_dict))
    return response.text


async def record_endorsement(conv_id, is_expert):
    """Record feedback given to a bot response"""
    try:
        # Call the record_endorsement API
        full_url = f"{BASE_URL}/record_endorsement/"
        requests.post(
            full_url,
            params={"session_id": str(conv_id), "is_expert": is_expert},  # pylint: disable=missing-timeout
            timeout=60,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"An error occurred during the asynchronous call record_endorsement: {exc}")
        return "An error occured"


async def process_msg(event, say):
    """
    Processes a user's message in Slack, retrieves and generates a response, and posts it in the channel.

    This asynchronous function handles a user's message event in Slack. It retrieves the message text, thread timestamp,
    and message timestamp from the event. It calls the `replies` function to retrieve the conversation context, and then
    calls the `getAns` function to generate a response based on the message and context. Finally, it posts the response in
    the channel as a reply to the original message. It also logs the conversation details.

    Args:
        event (dict): The Slack message event object.
        say (function): A function used to send messages in Slack.

    """
    user_id = event["user"]
    channel_id = event["channel"]
    _ = await say(channel=channel_id, text="Processing...", thread_ts=event["ts"])
    text = event["text"]
    thread_ts = event.get("thread_ts")
    message_ts = event.get("ts")
    conv_id = thread_ts or message_ts

    product, expert_user_id = determine_product_and_expert(channel_id)

    if product is None:
        _ = await say(
            channel=channel_id,
            text="I am unable to provide answers in this channel. Please refer to the appropriate channels.",
            thread_ts=event["ts"],
        )
        return

    text_items = text.split("> ")
    text = text_items[1] if len(text_items) == 2 else text_items[0]
    is_expert_answering = False
    if user_id == expert_user_id and text.lower().startswith("#answer"):
        text = text[7:].lstrip()
        is_expert_answering = True

    result = await get_ans(text, thread_ts, message_ts, product, is_expert_answering)
    await app.client.chat_postMessage(
        channel=channel_id,
        thread_ts=event["ts"],
        text=result,
    )
    response = await app.client.users_info(user=user_id)
    username = response["user"]["name"]
    conv_dict = {"conv_id": conv_id, "post": text, "poster": username}
    logger.info(str(conv_dict))


@app.event("reaction_added")
async def reaction(event, say):
    """
    Handles Slack reaction_added events, specifically for the +1 and -1 reactions.

    This asynchronous function is an event handler for Slack "reaction_added" events. It processes the event based on
    the reaction and user who triggered it.

    Args:
        event (dict): The Slack reaction_added event object.
        say (function): A function used to send messages in Slack.

    """
    # pylint:disable=too-many-locals

    logger.info(event)

    channel_id = event["item"]["channel"]
    time_stamp = event["item"]["ts"]
    response = await app.client.conversations_replies(ts=time_stamp, channel=channel_id)
    answer = response["messages"][0]
    conv_id = answer["thread_ts"]
    user = event["user"]

    _, expert_user_id = determine_product_and_expert(channel_id)

    if event["reaction"].startswith("+1") and event["item_user"] == LIL_LISA_SLACK_USERID:
        user = event["user"]
        is_expert = user == expert_user_id

        await record_endorsement(conv_id, is_expert)
        _ = await say(channel=channel_id, text="Thank you for your feedback!", thread_ts=conv_id)

    elif event["reaction"].startswith("sos"):
        _ = await say(channel=channel_id, text=f"<@{expert_user_id}> Can you help?", thread_ts=conv_id)


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
        response = requests.post(
            full_url,
            params={
                "product": product,  # pylint: disable=missing-timeout
                "encrypted_key": ENCRYPTED_AUTHENTICATION_KEY,
            },
            timeout=60,
        )
        if response:
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
async def update_golden_qa_pairs(ack, body, say):  # pylint: disable=too-many-locals
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
        response = requests.post(
            full_url,
            params={
                "product": product,  # pylint: disable=missing-timeout
                "encrypted_key": ENCRYPTED_AUTHENTICATION_KEY,
            },
            timeout=60,
        )
        if response:
            _ = await say(channel=dm_channel_id, text=response.text)
        else:
            error_msg = f"Call to lil-lisa server {full_url} has failed."
            logger.error(error_msg)
            return error_msg

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"An error occurred during the asynchronous call update_golden_qa_pairs: {exc}")
        return "An error occured"


@app.command("/get_conversations")
async def get_conversations(ack, body, say):
    """
    Slack command to retrieve the conversations endorsed by a specific entity.

    This asynchronous function is a Slack slash command handler for "/get_conversations". It sends progress and success messages in the Slack channel.

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
    endorsed_by = body.get("text").strip().lower()
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
        # Call the get_conversations API
        full_url = f"{BASE_URL}/get_conversations/"
        response = requests.post(
            full_url,
            params={
                "product": product,  # pylint: disable=missing-timeout
                "endorsed_by": endorsed_by,
                "encrypted_key": ENCRYPTED_AUTHENTICATION_KEY
            },
            timeout=60,
        )
        if response:
            await app.client.files_upload_v2(
                file=response.content,
                filename="conversations.zip",
                channel=dm_channel_id,
                initial_comment=f"Here are the conversations you requested! (endoresed by {endorsed_by})",
            )
        else:
            error_msg = f"Call to lil-lisa server {full_url} has failed."
            logger.error(error_msg)
            return error_msg

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"An error occurred during the asynchronous call get_conversations(): {exc}")
        return "An error occured"


@app.command("/rebuild_docs")
async def rebuild_docs(ack, body, say):
    """
    Slack command to rebuild the documentation database.

    This asynchronous function is a Slack slash command handler for "/rebuild_docs". It sends progress and success messages in the Slack channel.

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
        text="You are rebuilding the entire documentation database. This process will take approximately 15-20 minutes.",
    )
    try:
        # Call the rebuild_docs API
        full_url = f"{BASE_URL}/rebuild_docs/"
        response = requests.post(
            full_url,
            params={"encrypted_key": ENCRYPTED_AUTHENTICATION_KEY},
            timeout=2700,
        )
        if response:
            _ = await say(channel=dm_channel_id, text=response.text)
        else:
            error_msg = f"Call to lil-lisa server {full_url} has failed."
            logger.error(error_msg)
            return error_msg

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(f"An error occurred during the asynchronous call rebuild_docs(): {exc}")
        return "An error occured"


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
