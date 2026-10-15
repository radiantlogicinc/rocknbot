"""Read-modify-write helpers for the two techsupport tag stores in the
techsupport state folder (src/techsupport_state_paths.py).

* techsupport_thread_tags.json  {escalation thread_ts: related entry title},
  written by POST /tag_techsupport_thread/ (an async route), so callers must
  hold THREAD_TAGS_LOCK around upsert_thread_tag and two escalations cannot
  drop each other's keys.
* techsupport_answer_tags.json  {session_id (Slack thread ts): title of the
  verified techsupport entry Lil Lisa's answer cited}, written by invoke().
  invoke() is a plain `def` route running in FastAPI's threadpool, so that
  store is guarded by a threading.Lock, taken inside upsert_answer_tag()
  itself -- the caller only has to call it.

The nightly product-channel pass reads the answer store (cron/paths.py's
ANSWER_TAGS_PATH -> techsupport_qa_ingest.get_cited_entry_title) to tell an
expert correction of a cited entry apart from a brand-new Q&A.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Dict

from src.atomic_io import atomic_write_json
from src.techsupport_state_paths import ANSWER_TAGS_PATH, THREAD_TAGS_PATH

THREAD_TAGS_LOCK = asyncio.Lock()
ANSWER_TAGS_LOCK = threading.Lock()

DEFAULT_THREAD_TAGS_PATH = THREAD_TAGS_PATH
DEFAULT_ANSWER_TAGS_PATH = ANSWER_TAGS_PATH


def _load_tags(tags_path: Path) -> Dict[str, str]:
    if not tags_path.exists():
        return {}
    loaded = json.loads(tags_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {}
    return {str(key): str(value) for key, value in loaded.items()}


def load_thread_tags(path: Path | None = None) -> Dict[str, str]:
    return _load_tags(path or DEFAULT_THREAD_TAGS_PATH)


def save_thread_tags(tags: Dict[str, str], path: Path | None = None) -> None:
    tags_path = path or DEFAULT_THREAD_TAGS_PATH
    atomic_write_json(tags_path, tags)


def upsert_thread_tag(thread_ts: str, related_entry_title: str, path: Path | None = None) -> Dict[str, str]:
    """Load tags, set thread_ts -> title, atomic-write. Not itself locked."""
    tags = load_thread_tags(path)
    tags[thread_ts] = related_entry_title
    save_thread_tags(tags, path)
    return tags


def load_answer_tags(path: Path | None = None) -> Dict[str, str]:
    """{session_id: cited verified-entry title} written at answer time."""
    with ANSWER_TAGS_LOCK:
        return _load_tags(path or DEFAULT_ANSWER_TAGS_PATH)


def save_answer_tags(tags: Dict[str, str], path: Path | None = None) -> None:
    atomic_write_json(path or DEFAULT_ANSWER_TAGS_PATH, tags)


def upsert_answer_tag(session_id: str, title: str, path: Path | None = None) -> Dict[str, str]:
    """Record that the answer for `session_id` cited verified entry `title`.

    Unlike upsert_thread_tag, this one takes its own lock: invoke() is a sync
    route running in the threadpool, so concurrent answers would otherwise
    read-modify-write the same file and drop each other's keys.
    """
    tags_path = path or DEFAULT_ANSWER_TAGS_PATH
    with ANSWER_TAGS_LOCK:
        tags = _load_tags(tags_path)
        tags[str(session_id)] = str(title)
        atomic_write_json(tags_path, tags)
        return tags
