""" utility functions """

import logging
import os
from datetime import datetime, timezone
from time import time_ns
from typing import Optional

from dotenv import dotenv_values

LILLISA_SERVER_ENV_DICT = {**dotenv_values("./env/lillisa_server.env")}
# load config params from override folder
if areofp := LILLISA_SERVER_ENV_DICT.get("LILLISA_SERVER_ENV_OVERRIDE_FILEPATH", None):
    # if the folder exists and contains a file called lillisa_server.env, load it using dotenv
    if os.path.isfile(areofp):
        LILLISA_SERVER_ENV_DICT = {**LILLISA_SERVER_ENV_DICT, **dotenv_values(areofp)}
    else:
        print("The env file at LILLISA_SERVER_ENV_OVERRIDE_FILEPATH is missing")
else:
    print("LILLISA_SERVER_ENV_OVERRIDE_FILEPATH env variable not found in lillisa_server.env")
LILLISA_SERVER_ENV_DICT = {
    **LILLISA_SERVER_ENV_DICT,
    **os.environ,  # override loaded values with environment variables
}


def get_env_variable(var_name: str, default: Optional[str] = None) -> str:
    """
    Helper function to get the environment variable or raise exception.
    Used inside the container applications. DO NOT REMOVE!
    """
    try:
        return os.environ[var_name]
    except KeyError as exc:
        if default is not None:
            return default
        error_msg = f"The environment variable {var_name} was missing, abort..."
        logger.critical("%s", error_msg)
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
if log_level := LILLISA_SERVER_ENV_DICT["LOG_LEVEL"]:
    if log_level == "DEBUG":
        LOG_LEVEL = logging.DEBUG
    elif log_level == "INFO":
        LOG_LEVEL = logging.INFO
    elif log_level == "WARNING":
        LOG_LEVEL = logging.WARNING
    elif log_level == "ERROR":
        LOG_LEVEL = logging.ERROR
    elif log_level == "CRITICAL":
        LOG_LEVEL = logging.CRITICAL
    else:
        raise ValueError("LOG_LEVEL is not one of DEBUG, INFO, WARNING, ERROR, CRITICAL")
else:
    print("LOG_LEVEL env variable is not specified in env file or environment variable")
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

logging.getLogger("boto").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("speedict").setLevel(logging.WARNING)
logging.getLogger("llamaindex").setLevel(logging.WARNING)
logging.getLogger("llama_index.core").setLevel(logging.WARNING)
logging.getLogger("llama_index.core.indices").setLevel(logging.WARNING)
logging.getLogger("llama_index.core.indices.utils").setLevel(logging.WARNING)
logging.getLogger("src.llama_index_lancedb_vector_store").setLevel(logging.WARNING)

# some testing code
if __name__ == "__main__":
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warn message")
    logger.error("error message")
    logger.critical("critical message")

# To disable __debug__ and set the log level to INFO, use the -O option as shown below
# python3 -O utils.py
