""" utility functions """

import json
import logging
import os
from datetime import datetime, timezone
from time import time_ns
from typing import Any, Dict, Optional


def parse_get_ans_result(raw_result: str) -> Dict[str, Any]:
    """Parse LilLisa_Server /invoke/ JSON.

    get_ans() returns a JSON object on success, but a plain English string on
    timeout/exception. json.loads() of that string must not crash process_msg
    and leave Slack showing only "Processing...".
    """
    try:
        parsed = json.loads(raw_result)
    except (json.JSONDecodeError, TypeError):
        return {
            "response": str(raw_result),
            "links_text": "",
            "reranked_nodes": [],
            "needs_escalation": False,
        }
    if not isinstance(parsed, dict):
        return {
            "response": str(raw_result),
            "links_text": "",
            "reranked_nodes": [],
            "needs_escalation": False,
        }
    return parsed


def truncate_preserving_code_fences(text: str, max_length: int) -> str:
    """Truncate text without leaving an unclosed ``` fence (Slack mrkdwn).

    Same approach as LilLisa_Server `_truncate_match_answer`: if a hard slice
    would land inside a fence, cut before that fence; if the window is only the
    first fence, close it instead of dropping all text.
    """
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    if truncated.count("```") % 2 != 0:
        fence_index = truncated.rfind("```")
        before_fence = truncated[:fence_index].rstrip()
        if before_fence:
            return before_fence + "..."
        return truncated + "\n```..."
    return truncated + "..."


SLACK_ACTION_VALUE_MAX = 2000
ESCALATE_VALUE_QUERY_MAX_LENGTH = 1500


def build_escalation_button_value(
    query: str,
    channel_id: str,
    thread_ts: str,
    session_id,
    user_id: str,
    primary_techsupport_match_title: str = None,
) -> str:
    """Encode escalate-button state. Slack action `value` is capped at 2000 chars."""
    value: Dict[str, Any] = {
        "query": query[:ESCALATE_VALUE_QUERY_MAX_LENGTH],
        "channel_id": channel_id,
        "thread_ts": thread_ts,
        "session_id": str(session_id),
        "user_id": user_id,
    }
    encoded = json.dumps(value)
    if not primary_techsupport_match_title:
        return encoded
    title = primary_techsupport_match_title
    while True:
        candidate = dict(value)
        candidate["primary_techsupport_match_title"] = title
        blob = json.dumps(candidate)
        if len(blob) <= SLACK_ACTION_VALUE_MAX:
            return blob
        if len(title) <= 1:
            return encoded
        title = title[: max(0, len(title) - 32)]


def warn_if_escalate_body_channel_mismatch(body: Optional[Dict[str, Any]], orig_channel_id: Optional[str]) -> bool:
    """Warn when Slack interaction body.channel.id differs from button payload channel_id.

    Escalation chat_update/posts use orig_channel_id from the button payload only.
    body.channel can differ (e.g. forwarded messages) and must not be used for those calls.
    Returns True when a warning was logged.
    """
    if not body or not orig_channel_id:
        return False
    body_channel_id = (body.get("channel") or {}).get("id")
    if body_channel_id and body_channel_id != orig_channel_id:
        logger.warning(
            f"[ESCALATE] Slack body channel id {body_channel_id!r} differs from "
            f"button payload channel_id {orig_channel_id!r}; using payload channel_id"
        )
        return True
    return False


def get_env_variable(var_name: str, default: Optional[str] = None) -> str:
    """
    Helper function to get the environment variable or raise exception.
    DO NOT REMOVE!
    """
    try:
        return os.environ[var_name]
    except KeyError as exc:
        if default is not None:
            return default
        error_msg = f"The environment variable {var_name} was missing, abort..."
        raise EnvironmentError(error_msg) from exc


def format_ns(time_in_ns):
    """convert nanoseconds to text format"""
    formatted_time_upto_seconds = datetime.fromtimestamp(time_in_ns / 1e9, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    fractional_sec = (time_in_ns // 10**9) * 10**9
    nanoseconds_string = f"{fractional_sec}"[:9]

    return f"{formatted_time_upto_seconds}.{nanoseconds_string}Z"


# For time in nanoseconds
# https://stackoverflow.com/questions/31328300/python-logging-module-logging-timestamp-to-include-microsecond
class LogRecordNs(logging.LogRecord):  # pylint: disable=too-few-public-methods
    """class that returns nanoseconds"""

    def __init__(self, *args, **kwargs):
        self.created_ns = time_ns()  # Fetch precise timestamp
        super().__init__(*args, **kwargs)


LOG_LEVEL = logging.DEBUG if __debug__ else logging.INFO
logging.basicConfig(level=LOG_LEVEL)


class FormatterNs(logging.Formatter):
    """nanosecond log formatter"""

    default_nsec_format = "%Y-%m-%dT%H:%M:%S.%09dZ"

    def formatTime(self, record, datefmt=None):
        if datefmt is not None:  # Do not handle custom formats here ...
            return super().formatTime(record, datefmt)  # ... leave to original implementation
        return format_ns(record.created_ns)


logging.setLogRecordFactory(LogRecordNs)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s-%(funcName)s - %(message)s"
log_formatter = FormatterNs(LOG_FORMAT)

# create logger
logger = logging.getLogger("RL_Logger")
logger.setLevel(LOG_LEVEL)
logger.propagate = False  # otherwise you will see duplicate log entries

# # clear any existing handlers for our logger
logger.handlers.clear()

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(LOG_LEVEL)

# create and add formatter to ch
ch.setFormatter(log_formatter)

# add ch to logger
logger.addHandler(ch)


# create a separate logger for pytest_logger to log assertions
# with a slightly different format
pytest_assertion_logger = logging.getLogger("PyTest_Logger")
pytest_assertion_logger.setLevel(logging.DEBUG)
pytest_assertion_logger.propagate = False  # otherwise you will see duplicate log entries
pytest_assertion_logger.handlers.clear()
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(FormatterNs("%(asctime)s - %(levelname)s - %(message)s"))
pytest_assertion_logger.addHandler(ch)


# some testing code
if __name__ == "__main__":
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warn message")
    logger.error("error message")
    logger.critical("critical message")

# To disable __debug__ and set the log level to INFO, use the -O option as shown below
# python3 -O utils.py
