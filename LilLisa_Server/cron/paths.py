"""
Filesystem layout for the nightly techsupport cron jobs.

These jobs live in LilLisa_Server/cron/ and run two ways: standalone under
host cron (`make run-nightly`), or in the API process via
LilLisa_Server/src/techsupport_cron.py. Either way the server tree is simply
this package's parent directory, which also makes the layout inside the
container (/app/cron -> /app) fall out for free.

LilLisa_Server remains the source of:
  - env/lillisa_server.env
  - data/ (historical import dumps)
  - lancedb / passwords (paths inside that env file, relative to the server root)
  - src/ (VoyageEmbedding, LanceDBVectorStore)

Everything the pipeline writes at runtime (state JSON, thread/answer tags,
verified markdown) lives in the state folder next to the LanceDB folder, not
on this tree -- see src/techsupport_state_paths.py:
  - techsupport_thread_tags.json (written by POST /tag_techsupport_thread/)
  - techsupport_answer_tags.json (written by invoke() at answer time)

Override LILLISA_SERVER_ROOT only for unusual layouts; the default is correct
for both the repo checkout and the image.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

from dotenv import dotenv_values

PACKAGE_ROOT = Path(__file__).resolve().parent
LILLISA_SERVER_ROOT = Path(os.environ.get("LILLISA_SERVER_ROOT", str(PACKAGE_ROOT.parent))).resolve()
LILLISA_SERVER_ENV_PATH = LILLISA_SERVER_ROOT / "env" / "lillisa_server.env"

# Product question channels. Defined once, in lil-lisa.env (CHANNEL_ID_IDA /
# _IDDM / _IDO). The nightly scan reads those same names. The API image does
# not contain lil-lisa.env, so a container supplies them through the process
# environment instead of a second copy in techsupport_sync.env.
CHANNEL_ID_KEYS = ("CHANNEL_ID_IDA", "CHANNEL_ID_IDDM", "CHANNEL_ID_IDO")


def bot_env_path() -> Path:
    """lil-lisa.env. Override with LIL_LISA_ENV_PATH for an unusual layout.

    Default in a checkout is <repo>/lil-lisa/app_envfiles/lil-lisa.env
    (LilLisa_Server/cron -> repo root). In the API image (/app/cron) that
    path does not exist.
    """
    override = os.environ.get("LIL_LISA_ENV_PATH")
    if override:
        return Path(override)
    return PACKAGE_ROOT.parent.parent / "lil-lisa" / "app_envfiles" / "lil-lisa.env"


def read_lil_lisa_channel_ids() -> Dict[str, str]:
    """CHANNEL_ID_* from lil-lisa.env. Empty when that file is absent."""
    path = bot_env_path()
    if not path.is_file():
        return {}
    values = dotenv_values(str(path))
    found: Dict[str, str] = {}
    for key in CHANNEL_ID_KEYS:
        raw = values.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            found[key] = text
    return found


def ensure_import_paths() -> None:
    """Allow `import nightly_pipeline` and `from src.embedding_config import ...`."""
    for path in (PACKAGE_ROOT, LILLISA_SERVER_ROOT):
        rendered = str(path)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)


ensure_import_paths()

# The FastAPI process writes these files; cron only reads them. Both sides
# import them from the same module so they cannot drift apart.
# THREAD_TAGS_PATH: {escalation thread_ts: related entry title}.
# ANSWER_TAGS_PATH: {session_id (Slack thread ts): title of the verified
# techsupport entry the bot's answer cited}, written by src/main.py's invoke().
from src.techsupport_state_paths import (  # noqa: E402
    ANSWER_TAGS_PATH,
    REEMBED_STATE_PATH,
    REVIEW_STATE_PATH,
    STATE_DIR,
    SYNC_STATE_PATH,
    THREAD_TAGS_PATH,
    VERIFIED_TECHSUPPORT_QA_FOLDERPATH,
)

__all__ = [
    "ANSWER_TAGS_PATH",
    "CHANNEL_ID_KEYS",
    "LILLISA_SERVER_ENV_PATH",
    "LILLISA_SERVER_ROOT",
    "PACKAGE_ROOT",
    "bot_env_path",
    "read_lil_lisa_channel_ids",
    "REEMBED_STATE_PATH",
    "REVIEW_STATE_PATH",
    "STATE_DIR",
    "SYNC_STATE_PATH",
    "THREAD_TAGS_PATH",
    "VERIFIED_TECHSUPPORT_QA_FOLDERPATH",
    "ensure_import_paths",
]
