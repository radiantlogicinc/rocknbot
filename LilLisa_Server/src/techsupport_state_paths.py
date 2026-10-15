"""Where the techsupport pipeline keeps the files it writes at runtime.

The API process, the cron/ jobs and the startup preflight all resolve these
paths here, so they always agree on one location.

These files must survive a container restart, and in the image only the
volume behind LANCEDB_FOLDERPATH does (/app is rebuilt from the image every
rollout). They also describe rows in the TECHSUPPORT_QA_PAIRS table and must
be restored together with it. So by default they live in a techsupport_state/
folder next to the LanceDB folder:

    LANCEDB_FOLDERPATH=/data/lancedb  ->  /data/techsupport_state/
    LANCEDB_FOLDERPATH=./lancedb      ->  LilLisa_Server/techsupport_state/

Both overrides are optional:
    TECHSUPPORT_STATE_DIR              - folder for the JSON state files
    VERIFIED_TECHSUPPORT_QA_FOLDERPATH - folder for techsupport_qa_pairs.md
                                         (default: <state dir>/verified_techsupport)

Read from env/lillisa_server.env (plus its override file) and the process
environment, the same precedence the server uses. Relative values are
resolved against the LilLisa_Server root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src import utils

SERVER_ROOT = Path(__file__).resolve().parent.parent
STATE_DIRNAME = "techsupport_state"
VERIFIED_QA_DIRNAME = "verified_techsupport"

SYNC_STATE_FILENAME = "techsupport_sync_state.json"
REEMBED_STATE_FILENAME = "techsupport_reembed_state.json"
REVIEW_STATE_FILENAME = "techsupport_review_state.json"
THREAD_TAGS_FILENAME = "techsupport_thread_tags.json"
ANSWER_TAGS_FILENAME = "techsupport_answer_tags.json"


def _clean(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _resolve(raw: str, server_root: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else server_root / path


def resolve_state_dir(env: Mapping[str, object], server_root: Path = SERVER_ROOT) -> Path:
    raw = _clean(env.get("TECHSUPPORT_STATE_DIR"))
    if raw:
        return _resolve(raw, server_root)
    # Same "./lancedb" default as env/lillisa_server.env.example.
    lancedb = _resolve(_clean(env.get("LANCEDB_FOLDERPATH")) or "lancedb", server_root)
    return lancedb.parent / STATE_DIRNAME


def resolve_verified_qa_dir(env: Mapping[str, object], server_root: Path = SERVER_ROOT) -> Path:
    raw = _clean(env.get("VERIFIED_TECHSUPPORT_QA_FOLDERPATH"))
    if raw:
        return _resolve(raw, server_root)
    return resolve_state_dir(env, server_root) / VERIFIED_QA_DIRNAME


STATE_DIR = resolve_state_dir(utils.LILLISA_SERVER_ENV_DICT)
VERIFIED_TECHSUPPORT_QA_FOLDERPATH = resolve_verified_qa_dir(utils.LILLISA_SERVER_ENV_DICT)

SYNC_STATE_PATH = STATE_DIR / SYNC_STATE_FILENAME
REEMBED_STATE_PATH = STATE_DIR / REEMBED_STATE_FILENAME
REVIEW_STATE_PATH = STATE_DIR / REVIEW_STATE_FILENAME
# {escalation thread_ts: related entry title}, written by POST /tag_techsupport_thread/.
THREAD_TAGS_PATH = STATE_DIR / THREAD_TAGS_FILENAME
# {session_id (Slack thread ts): title of the verified entry the answer cited}, written by invoke().
ANSWER_TAGS_PATH = STATE_DIR / ANSWER_TAGS_FILENAME
