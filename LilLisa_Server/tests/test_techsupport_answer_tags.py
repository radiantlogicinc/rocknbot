"""
Unit tests for the answer-time tag store (techsupport_answer_tags.json in the
techsupport state folder):
{session_id (Slack thread ts): title of the verified techsupport entry the bot's
answer cited}. invoke() writes it; the nightly product-channel pass reads it to
route an expert reply to a correction instead of a new entry.

invoke() is a sync route running in FastAPI's threadpool, so unlike the
escalation store this one takes its own threading.Lock inside upsert.

src.main is deliberately NOT imported here (it pulls the whole server).

Run from LilLisa_Server:
    PYTHONPATH=. python3 tests/test_techsupport_answer_tags.py
"""

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.techsupport_thread_tags import (  # noqa: E402
    ANSWER_TAGS_LOCK,
    DEFAULT_ANSWER_TAGS_PATH,
    DEFAULT_THREAD_TAGS_PATH,
    load_answer_tags,
    upsert_answer_tag,
)


class AnswerTagsUpsertTests(unittest.TestCase):
    def test_sequential_upserts_keep_both_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "techsupport_answer_tags.json"
            upsert_answer_tag("111.1", "LDAP bind", path=path)
            upsert_answer_tag("222.2", "SSO certs", path=path)
            tags = load_answer_tags(path)
            self.assertEqual(tags, {"111.1": "LDAP bind", "222.2": "SSO certs"})

    def test_reanswer_overwrites_the_same_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "techsupport_answer_tags.json"
            upsert_answer_tag("111.1", "LDAP bind", path=path)
            upsert_answer_tag("111.1", "LDAP referrals", path=path)
            self.assertEqual(load_answer_tags(path), {"111.1": "LDAP referrals"})

    def test_missing_file_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_answer_tags(Path(tmp) / "nope.json"), {})

    def test_write_is_atomic_and_leaves_no_temp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "techsupport_answer_tags.json"
            upsert_answer_tag("111.1", "LDAP bind", path=path)
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))
            self.assertEqual([p.name for p in Path(tmp).iterdir()], [path.name])

    def test_parent_directory_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scripts" / "techsupport_answer_tags.json"
            upsert_answer_tag("111.1", "LDAP bind", path=path)
            self.assertTrue(path.exists())

    def test_lock_serializes_concurrent_upserts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "techsupport_answer_tags.json"
            path.write_text("{}\n", encoding="utf-8")
            start = threading.Barrier(8)

            def tag(index: int) -> None:
                start.wait()
                upsert_answer_tag(f"{index}.0", f"Entry {index}", path=path)

            threads = [threading.Thread(target=tag, args=(i,)) for i in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            tags = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(tags, {f"{i}.0": f"Entry {i}" for i in range(8)})

    def test_lock_is_not_held_after_an_upsert(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "techsupport_answer_tags.json"
            upsert_answer_tag("111.1", "LDAP bind", path=path)
            self.assertTrue(ANSWER_TAGS_LOCK.acquire(blocking=False))
            ANSWER_TAGS_LOCK.release()

    def test_default_path_is_its_own_store(self):
        self.assertNotEqual(DEFAULT_ANSWER_TAGS_PATH, DEFAULT_THREAD_TAGS_PATH)
        self.assertEqual(DEFAULT_ANSWER_TAGS_PATH.name, "techsupport_answer_tags.json")
        self.assertEqual(DEFAULT_ANSWER_TAGS_PATH.parent, DEFAULT_THREAD_TAGS_PATH.parent)


if __name__ == "__main__":
    unittest.main()
