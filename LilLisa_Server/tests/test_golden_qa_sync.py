"""
Unit tests for pushing expert-verified QA pairs back to the golden QA repo.

No network and no real GitHub: the "remote" is a bare git repo in a temp dir,
cloned over a file:// URL, so clone/commit/push all run for real.

Run from LilLisa_Server:
    PYTHONPATH=. python3 tests/test_golden_qa_sync.py
"""

import io
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src import golden_qa_sync  # noqa: E402

# The module logs an error for every deliberate push failure below; keep the
# test output readable.
logging.getLogger("RL_Logger").setLevel(logging.CRITICAL)

import backfill_expert_qa_pairs as backfill  # noqa: E402

# Exactly what _run_update_golden_qa_pairs_task() in src/main.py does.
REBUILD_SPLIT_TOKEN = "# Question/Answer Pair"
REBUILD_PATTERN = re.compile(r"Question:\s*(.*?)\nAnswer:\s*(.*)", re.DOTALL)


def rebuild_parse(file_content):
    """A verbatim copy of the rebuild task's parsing, so the round-trip test
    proves the format against main.py's logic and not a shared helper."""
    pairs = []
    for chunk in [p.strip() for p in file_content.split(REBUILD_SPLIT_TOKEN) if p.strip()]:
        if match := REBUILD_PATTERN.search(chunk):
            pairs.append((match[1].strip(), match[2].strip()))
    return pairs


def git_run(*args, cwd=None):
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


class FormatRoundTripTests(unittest.TestCase):
    def test_single_entry_round_trips_through_the_rebuild_parser(self):
        entry = golden_qa_sync.format_golden_qa_entry(
            "How do I set the bind DN?", "Set it in the data source configuration."
        )
        self.assertEqual(
            rebuild_parse(entry),
            [("How do I set the bind DN?", "Set it in the data source configuration.")],
        )

    def test_multiple_appended_entries_round_trip(self):
        content = "".join(
            golden_qa_sync.format_golden_qa_entry(f"Question {i}?", f"Answer {i}\nwith a second line.")
            for i in range(3)
        )
        expected = [(f"Question {i}?", f"Answer {i}\nwith a second line.") for i in range(3)]
        self.assertEqual(rebuild_parse(content), expected)
        # The module's own parser must agree with the rebuild task's.
        self.assertEqual(golden_qa_sync.parse_golden_qa_entries(content), expected)

    def test_multiline_question_round_trips(self):
        question = "Upgrade fails.\nWhat should I check first?"
        entry = golden_qa_sync.format_golden_qa_entry(question, "Check the schema version.")
        self.assertEqual(rebuild_parse(entry), [(question, "Check the schema version.")])


class LocalRemoteTestCase(unittest.TestCase):
    """Base: a bare repo in a temp dir, seeded and cloneable over file://."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="golden_qa_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.remote = Path(self.tmp) / "remote.git"
        git_run("init", "--bare", "--initial-branch=main", str(self.remote))
        self.repo_url = self.remote.as_uri()
        self._seed_remote()

    def _seed_remote(self):
        """A bare repo with no commits cannot be cloned usefully; give it a README."""
        seed = Path(self.tmp) / "seed"
        git_run("clone", self.repo_url, str(seed))
        git_run("config", "user.email", "test@example.com", cwd=seed)
        git_run("config", "user.name", "Test", cwd=seed)
        (seed / "README.md").write_text("golden qa pairs\n", encoding="utf-8")
        git_run("add", "README.md", cwd=seed)
        git_run("commit", "-m", "seed", cwd=seed)
        git_run("push", "origin", "HEAD:main", cwd=seed)

    def remote_file(self, filename):
        """The file's content at the remote's tip (empty string if absent)."""
        result = subprocess.run(
            ["git", "show", f"HEAD:{filename}"],
            cwd=self.remote,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.decode("utf-8") if result.returncode == 0 else ""

    def remote_log(self):
        result = subprocess.run(
            ["git", "log", "--format=%s"],
            cwd=self.remote,
            check=True,
            stdout=subprocess.PIPE,
        )
        return result.stdout.decode("utf-8").splitlines()


class RepoAppendTests(LocalRemoteTestCase):
    """clone -> append -> commit -> push, for real, against the local remote."""

    def test_append_creates_the_file_when_missing(self):
        result = golden_qa_sync.append_expert_qa_pair_to_repo(
            "IDDM", "How do I rotate the PAT?", "Update github_push.env.", repo_url=self.repo_url
        )
        self.assertTrue(result["pushed"], result)
        self.assertEqual(result["filename"], "iddm_qa_pairs.md")
        content = self.remote_file("iddm_qa_pairs.md")
        self.assertEqual(rebuild_parse(content), [("How do I rotate the PAT?", "Update github_push.env.")])
        self.assertIn("Add expert-verified QA pair (IDDM)", self.remote_log()[0])

    def test_append_keeps_existing_entries(self):
        first = golden_qa_sync.append_expert_qa_pair_to_repo(
            "IDA", "First question?", "First answer.", repo_url=self.repo_url
        )
        self.assertTrue(first["pushed"], first)
        second = golden_qa_sync.append_expert_qa_pair_to_repo(
            "IDA", "Second question?", "Second answer.", repo_url=self.repo_url
        )
        self.assertTrue(second["pushed"], second)

        content = self.remote_file("ida_qa_pairs.md")
        self.assertEqual(
            rebuild_parse(content),
            [("First question?", "First answer."), ("Second question?", "Second answer.")],
        )
        # Two separate commits, and nothing else clobbered.
        self.assertEqual(len(self.remote_log()), 3)  # seed + two appends
        self.assertEqual(self.remote_file("README.md"), "golden qa pairs\n")

    def test_append_to_a_pre_existing_file_without_trailing_newline(self):
        seed = Path(self.tmp) / "seed2"
        git_run("clone", self.repo_url, str(seed))
        git_run("config", "user.email", "test@example.com", cwd=seed)
        git_run("config", "user.name", "Test", cwd=seed)
        (seed / "iddm_qa_pairs.md").write_text(
            "# Question/Answer Pair\nQuestion: Old question?\nAnswer: Old answer.", encoding="utf-8"
        )
        git_run("add", "iddm_qa_pairs.md", cwd=seed)
        git_run("commit", "-m", "existing pairs", cwd=seed)
        git_run("push", "origin", "HEAD:main", cwd=seed)

        result = golden_qa_sync.append_expert_qa_pair_to_repo(
            "IDDM", "New question?", "New answer.", repo_url=self.repo_url
        )
        self.assertTrue(result["pushed"], result)
        self.assertEqual(
            rebuild_parse(self.remote_file("iddm_qa_pairs.md")),
            [("Old question?", "Old answer."), ("New question?", "New answer.")],
        )

    def test_read_repo_qa_pairs(self):
        self.assertEqual(golden_qa_sync.read_repo_qa_pairs("IDDM", repo_url=self.repo_url), [])
        golden_qa_sync.append_expert_qa_pair_to_repo(
            "IDDM", "Q1?", "A1.", repo_url=self.repo_url
        )
        self.assertEqual(
            golden_qa_sync.read_repo_qa_pairs("IDDM", repo_url=self.repo_url), [("Q1?", "A1.")]
        )


class PushFailureTests(unittest.TestCase):
    def test_unreachable_remote_returns_pushed_false(self):
        result = golden_qa_sync.append_expert_qa_pair_to_repo(
            "IDDM", "Q?", "A.", repo_url="file:///nonexistent/golden_qa_repo.git"
        )
        self.assertFalse(result["pushed"])
        self.assertTrue(result["error"])

    def test_push_error_is_caught_not_raised(self):
        with mock.patch.object(golden_qa_sync, "cloned_repo", side_effect=RuntimeError("boom")):
            result = golden_qa_sync.append_expert_qa_pair_to_repo("IDDM", "Q?", "A.")
        self.assertFalse(result["pushed"])
        self.assertIn("boom", result["error"])

    def test_missing_repo_url_returns_pushed_false(self):
        with mock.patch.object(golden_qa_sync, "resolve_repo_url", return_value=None):
            result = golden_qa_sync.append_expert_qa_pair_to_repo("IDDM", "Q?", "A.")
        self.assertFalse(result["pushed"])
        self.assertIn("QA_PAIRS_GITHUB_REPO_URL", result["error"])

    def test_empty_question_is_rejected_without_touching_git(self):
        with mock.patch.object(golden_qa_sync, "cloned_repo") as cloned:
            result = golden_qa_sync.append_expert_qa_pair_to_repo("IDDM", "   ", "A.")
        cloned.assert_not_called()
        self.assertFalse(result["pushed"])

    def test_non_https_remote_is_rejected(self):
        with self.assertRaises(ValueError):
            golden_qa_sync._needs_auth("git@github.com:org/repo.git")
        with self.assertRaises(ValueError):
            golden_qa_sync._needs_auth("https://token@github.com/org/repo.git")
        self.assertTrue(golden_qa_sync._needs_auth("https://github.com/org/repo.git"))


class TokenLookupTests(unittest.TestCase):
    def test_uses_github_token(self):
        with mock.patch.dict("os.environ", {"GITHUB_TOKEN": "ts-token"}, clear=False):
            self.assertEqual(golden_qa_sync.load_github_token(), "ts-token")

    def test_blank_github_token_is_missing(self):
        with mock.patch.dict("os.environ", {"GITHUB_TOKEN": "   "}, clear=False):
            self.assertIsNone(golden_qa_sync.load_github_token())


LOG_SAMPLE = """2026-09-01 10:00:00 INFO Some unrelated line
2026-09-01 10:00:01 INFO Expert QA Verification: {"timestamp": "t1", "action": "expert_qa_verification", "product": "IDDM", "question": "How do I reset?", "answer": "Run the reset tool."}
2026-09-01 10:05:00 INFO Expert QA Verification: {"timestamp": "t2", "action": "expert_qa_verification", "product": "IDDM", "question": "How do I reset?", "answer": "Run the reset tool."}
2026-09-01 10:06:00 INFO Expert QA Verification: {"timestamp": "t3", "action": "expert_qa_verification", "product": "IDA", "question": "Where are logs?", "answer": "Under /var/log."}
2026-09-01 10:07:00 INFO Expert QA Verification: not json at all
2026-09-01 10:08:00 INFO Expert QA Verification: {"product": "IDA", "question": "", "answer": "no question"}
"""


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="golden_qa_backfill_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.log_path = Path(self.tmp) / "server.log"
        self.log_path.write_text(LOG_SAMPLE, encoding="utf-8")

    def test_parse_log_lines_dedupes_and_skips_junk(self):
        pairs = backfill.parse_log_lines(LOG_SAMPLE.splitlines())
        self.assertEqual(
            pairs,
            [
                ("IDDM", "How do I reset?", "Run the reset tool."),
                ("IDA", "Where are logs?", "Under /var/log."),
            ],
        )

    def test_dry_run_pushes_nothing(self):
        with mock.patch.object(backfill.golden_qa_sync, "read_repo_qa_pairs", return_value=[]) as read_repo:
            with mock.patch.object(backfill.golden_qa_sync, "append_expert_qa_pair_to_repo") as append:
                summary = backfill.backfill(str(self.log_path), dry_run=True, repo_url="file:///fake")
        append.assert_not_called()
        self.assertTrue(read_repo.called)
        self.assertEqual(summary["found"], 2)
        self.assertEqual(summary["pushed"], 0)
        self.assertEqual(len(summary["would_push"]), 2)

    def test_skips_pairs_already_in_the_repo(self):
        def fake_read(product, repo_url=None, token=None):
            if product == "IDDM":
                return [("How do I reset?", "Run the reset tool.")]
            return []

        with mock.patch.object(backfill.golden_qa_sync, "read_repo_qa_pairs", side_effect=fake_read):
            with mock.patch.object(
                backfill.golden_qa_sync, "append_expert_qa_pair_to_repo", return_value={"pushed": True}
            ) as append:
                summary = backfill.backfill(str(self.log_path), repo_url="file:///fake")

        self.assertEqual(summary["skipped_existing"], 1)
        self.assertEqual(summary["pushed"], 1)
        self.assertEqual(append.call_count, 1)
        self.assertEqual(append.call_args[0][0], "IDA")

    def test_product_filter_and_push_failures_are_reported(self):
        with mock.patch.object(backfill.golden_qa_sync, "read_repo_qa_pairs", return_value=[]):
            with mock.patch.object(
                backfill.golden_qa_sync,
                "append_expert_qa_pair_to_repo",
                return_value={"pushed": False, "error": "RuntimeError: nope"},
            ) as append:
                summary = backfill.backfill(str(self.log_path), products=["iddm"], repo_url="file:///fake")

        self.assertEqual(summary["found"], 1)
        self.assertEqual(append.call_count, 1)
        self.assertEqual(summary["failed"], 1)
        self.assertIn("nope", summary["errors"][0])

    def test_main_dry_run_exits_zero(self):
        with mock.patch.object(backfill.golden_qa_sync, "read_repo_qa_pairs", return_value=[]):
            with mock.patch.object(backfill.golden_qa_sync, "append_expert_qa_pair_to_repo") as append:
                with redirect_stdout(io.StringIO()) as printed:
                    code = backfill.main(
                        ["--log-file", str(self.log_path), "--dry-run", "--repo-url", "file:///fake"]
                    )
        self.assertIn('"dry_run": true', printed.getvalue())
        append.assert_not_called()
        self.assertEqual(code, 0)


class EndToEndBackfillTests(LocalRemoteTestCase):
    """The backfill against the real local remote, no mocks."""

    def test_backfill_appends_only_missing_pairs(self):
        golden_qa_sync.append_expert_qa_pair_to_repo(
            "IDDM", "How do I reset?", "Run the reset tool.", repo_url=self.repo_url
        )
        log_path = Path(self.tmp) / "server.log"
        log_path.write_text(LOG_SAMPLE, encoding="utf-8")

        summary = backfill.backfill(str(log_path), repo_url=self.repo_url)
        self.assertEqual(summary["found"], 2)
        self.assertEqual(summary["skipped_existing"], 1)
        self.assertEqual(summary["pushed"], 1)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(
            rebuild_parse(self.remote_file("ida_qa_pairs.md")),
            [("Where are logs?", "Under /var/log.")],
        )


if __name__ == "__main__":
    unittest.main()
