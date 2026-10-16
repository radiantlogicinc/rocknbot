"""
Unit tests for github_sync GIT_ASKPASS auth (token never in clone URL).

Run from LilLisa_Server/cron:
    PYTHONPATH=. python3 tests/test_github_sync.py
"""

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

# github_sync -> paths -> src.techsupport_state_paths -> src.utils, whose
# module-level LILLISA_SERVER_ENV_DICT["LOG_LEVEL"] lookup is the one hard env
# requirement in that chain. The dotenv mock installed right below makes every
# dotenv_values() call (including src.utils's) return {}, so this must be set
# via os.environ instead -- same fix as test_techsupport_correct_entry.py's
# setUp comment describes, needed here too when this file runs standalone
# (not just under `unittest discover`, where some other file installs the
# mock first).
os.environ.setdefault("LOG_LEVEL", "INFO")

sys.modules.setdefault("git", MagicMock())
sys.modules["git"].Actor = MagicMock(return_value="actor")
sys.modules.setdefault("dotenv", MagicMock())
sys.modules["dotenv"].dotenv_values = MagicMock(return_value={})

import github_sync  # noqa: E402


class GitAskpassTests(unittest.TestCase):
    def test_askpass_script_reads_env_and_does_not_embed_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = github_sync._write_git_askpass(Path(tmp))
            text = path.read_text(encoding="utf-8")
            self.assertIn("$GITHUB_TOKEN", text)
            self.assertNotIn("ghp_", text)
            self.assertTrue(os.access(path, os.X_OK))
            self.assertTrue(stat.S_IMODE(path.stat().st_mode) & stat.S_IXUSR)

    def test_git_auth_env_sets_askpass_and_token(self):
        env = github_sync._git_auth_env("secret-token", Path("/tmp/git-askpass.sh"))
        self.assertEqual(env["GIT_ASKPASS"], "/tmp/git-askpass.sh")
        self.assertEqual(env["GITHUB_TOKEN"], "secret-token")
        self.assertEqual(env["GIT_TERMINAL_PROMPT"], "0")

    def test_https_url_with_embedded_user_is_rejected(self):
        with self.assertRaises(ValueError):
            github_sync._require_https_repo_url("https://ghp_abc@github.com/org/repo.git")

    def test_ssh_url_is_rejected(self):
        with self.assertRaises(ValueError):
            github_sync._require_https_repo_url("git@github.com:org/repo.git")

    def test_clone_uses_plain_url_and_askpass_env(self):
        repo_url = "https://github.com/org/techsupport-qa.git"
        token = "secret-token-not-for-url"

        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "verified"
            source_dir.mkdir()
            source_path = source_dir / github_sync.TECHSUPPORT_QA_MARKDOWN_FILENAME
            source_path.write_text("## New entry\n", encoding="utf-8")

            def fake_clone(url, to_path, **kwargs):
                self.assertEqual(url, repo_url)
                self.assertNotIn(token, url)
                self.assertEqual(kwargs["env"]["GITHUB_TOKEN"], token)
                self.assertTrue(kwargs["env"]["GIT_ASKPASS"].endswith("git-askpass.sh"))
                self.assertEqual(kwargs["multi_options"], ["--config", "credential.helper="])
                os.makedirs(to_path, exist_ok=True)
                Path(to_path, github_sync.TECHSUPPORT_QA_MARKDOWN_FILENAME).write_text("## old\n", encoding="utf-8")
                repo = MagicMock()
                repo.git.custom_environment.return_value.__enter__.return_value = None
                repo.git.custom_environment.return_value.__exit__.return_value = False
                return repo

            with patch.object(
                github_sync,
                "load_env",
                return_value={
                    "GITHUB_TOKEN": token,
                    "GITHUB_REPO_URL": repo_url,
                },
            ), patch.object(github_sync, "VERIFIED_TECHSUPPORT_QA_FOLDERPATH", source_dir), patch.object(
                github_sync.git.Repo, "clone_from", side_effect=fake_clone
            ):
                result = github_sync.push_verified_qa_pairs()

        self.assertTrue(result["pushed"])


class ShrinkGuardTests(unittest.TestCase):
    """push_verified_qa_pairs() must refuse to overwrite the repo's copy of
    techsupport_qa_pairs.md whenever the local file has fewer entries. The
    pipeline never deletes an entry, so any drop means the local file was
    lost or truncated. filecmp.cmp() alone treats that as a normal content
    change and would push it over the repo."""

    def _run_push(self, source_entries, dest_entries):
        """Shared harness: the mocked clone_from() writes `dest_entries`
        "## Old entry N" blocks into the cloned repo's copy of the markdown;
        the local copy gets `source_entries` "## Entry N" blocks. Returns
        (result_or_None, raised_exception_or_None, push_was_called)."""
        env = {"GITHUB_TOKEN": "secret-token-not-for-url", "GITHUB_REPO_URL": "https://github.com/org/repo.git"}

        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "verified"
            source_dir.mkdir()
            source_path = source_dir / github_sync.TECHSUPPORT_QA_MARKDOWN_FILENAME
            source_path.write_text(
                "".join(f"## Entry {i}\n\nBody {i}.\n\n" for i in range(source_entries)),
                encoding="utf-8",
            )

            pushed = {"called": False}

            def fake_clone(_url, to_path, **_kwargs):
                os.makedirs(to_path, exist_ok=True)
                Path(to_path, github_sync.TECHSUPPORT_QA_MARKDOWN_FILENAME).write_text(
                    "".join(f"## Old entry {i}\n\nOld body {i}.\n\n" for i in range(dest_entries)),
                    encoding="utf-8",
                )
                repo = MagicMock()
                repo.git.custom_environment.return_value.__enter__.return_value = None
                repo.git.custom_environment.return_value.__exit__.return_value = False
                repo.remote.return_value.push.side_effect = lambda: pushed.__setitem__("called", True)
                return repo

            with patch.object(github_sync, "load_env", return_value=env), patch.object(
                github_sync, "VERIFIED_TECHSUPPORT_QA_FOLDERPATH", source_dir
            ), patch.object(github_sync.git.Repo, "clone_from", side_effect=fake_clone):
                try:
                    result = github_sync.push_verified_qa_pairs()
                    return result, None, pushed["called"]
                except Exception as exc:  # noqa: BLE001 -- caller asserts the exact type
                    return None, exc, pushed["called"]

    def test_any_shrink_refuses_to_push(self):
        result, exc, pushed = self._run_push(source_entries=9, dest_entries=10)
        self.assertIsNone(result)
        self.assertIsInstance(exc, github_sync.ShrinkGuardError)
        self.assertIn("9 entries", str(exc))
        self.assertIn("10", str(exc))
        self.assertFalse(pushed)

    def test_empty_local_file_refuses_to_push_over_existing_repo(self):
        result, exc, pushed = self._run_push(source_entries=0, dest_entries=3)
        self.assertIsNone(result)
        self.assertIsInstance(exc, github_sync.ShrinkGuardError)
        self.assertFalse(pushed)

    def test_equal_count_with_rewritten_text_still_pushes(self):
        result, exc, pushed = self._run_push(source_entries=10, dest_entries=10)
        self.assertIsNone(exc)
        self.assertTrue(result["pushed"])
        self.assertTrue(pushed)

    def test_growth_is_always_allowed(self):
        result, exc, pushed = self._run_push(source_entries=201, dest_entries=200)
        self.assertIsNone(exc)
        self.assertTrue(result["pushed"])
        self.assertTrue(pushed)

    def test_first_ever_push_has_nothing_to_shrink_from(self):
        # Repo's copy has 0 entries (new/empty file) -- nothing to compare against.
        result, exc, pushed = self._run_push(source_entries=1, dest_entries=0)
        self.assertIsNone(exc)
        self.assertTrue(result["pushed"])
        self.assertTrue(pushed)


class PullTests(unittest.TestCase):
    """pull_verified_qa_pairs() treats GITHUB_REPO_URL as the source of truth
    -- techsupport_bootstrap.py clones it at startup and rebuilds
    TECHSUPPORT_QA_PAIRS from the result -- so the repo's copy normally wins
    over the local one. The one exception is a local file with MORE entries:
    the pipeline never deletes an entry, so those are unpushed entries that
    exist nowhere else and must survive the pull so sync_verified_qa_pairs()
    can push them."""

    def _run_pull(self, local_entries, remote_entries, remote_exists=True):
        """Returns (result, local_text_after). local_entries=None means no
        local file at all (the fresh-deployment case)."""
        env = {"GITHUB_TOKEN": "secret-token-not-for-url", "GITHUB_REPO_URL": "https://github.com/org/repo.git"}

        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "verified"
            local_dir.mkdir()
            local_path = local_dir / github_sync.TECHSUPPORT_QA_MARKDOWN_FILENAME
            if local_entries is not None:
                local_path.write_text(
                    "".join(f"## Local {i}\n\nLocal body {i}.\n\n" for i in range(local_entries)),
                    encoding="utf-8",
                )

            def fake_clone(_url, to_path, **_kwargs):
                os.makedirs(to_path, exist_ok=True)
                if remote_exists:
                    Path(to_path, github_sync.TECHSUPPORT_QA_MARKDOWN_FILENAME).write_text(
                        "".join(f"## Remote {i}\n\nRemote body {i}.\n\n" for i in range(remote_entries)),
                        encoding="utf-8",
                    )
                return MagicMock()

            with patch.object(github_sync, "load_env", return_value=env), patch.object(
                github_sync, "VERIFIED_TECHSUPPORT_QA_FOLDERPATH", local_dir
            ), patch.object(github_sync.git.Repo, "clone_from", side_effect=fake_clone):
                result = github_sync.pull_verified_qa_pairs()

            local_text = local_path.read_text(encoding="utf-8") if local_path.exists() else None
            return result, local_text

    def test_fresh_deployment_with_no_local_file_pulls_the_repo(self):
        result, local_text = self._run_pull(local_entries=None, remote_entries=638)
        self.assertTrue(result["pulled"])
        self.assertEqual(result["entry_count"], 638)
        self.assertIn("## Remote 0", local_text)

    def test_repo_ahead_overwrites_the_local_copy(self):
        result, local_text = self._run_pull(local_entries=2, remote_entries=638)
        self.assertTrue(result["pulled"])
        self.assertIn("## Remote 0", local_text)
        self.assertNotIn("## Local 0", local_text)

    def test_local_ahead_is_kept_for_sync_to_push(self):
        result, local_text = self._run_pull(local_entries=640, remote_entries=638)
        self.assertFalse(result["pulled"])
        self.assertEqual(result["reason"], "local_ahead")
        self.assertEqual(result["local_entry_count"], 640)
        self.assertEqual(result["remote_entry_count"], 638)
        self.assertIn("## Local 0", local_text)

    def test_equal_counts_but_different_text_still_takes_the_repo_copy(self):
        # Equal counts are NOT the local_ahead exception: the repo is the
        # source of truth, so its text wins when the two differ.
        result, local_text = self._run_pull(local_entries=5, remote_entries=5)
        self.assertTrue(result["pulled"])
        self.assertIn("## Remote 0", local_text)

    def test_identical_files_are_a_no_op(self):
        env = {"GITHUB_TOKEN": "t", "GITHUB_REPO_URL": "https://github.com/org/repo.git"}
        content = "## Same\n\nSame body.\n\n"

        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "verified"
            local_dir.mkdir()
            (local_dir / github_sync.TECHSUPPORT_QA_MARKDOWN_FILENAME).write_text(content, encoding="utf-8")

            def fake_clone(_url, to_path, **_kwargs):
                os.makedirs(to_path, exist_ok=True)
                Path(to_path, github_sync.TECHSUPPORT_QA_MARKDOWN_FILENAME).write_text(content, encoding="utf-8")
                return MagicMock()

            with patch.object(github_sync, "load_env", return_value=env), patch.object(
                github_sync, "VERIFIED_TECHSUPPORT_QA_FOLDERPATH", local_dir
            ), patch.object(github_sync.git.Repo, "clone_from", side_effect=fake_clone):
                result = github_sync.pull_verified_qa_pairs()

        self.assertFalse(result["pulled"])
        self.assertEqual(result["reason"], "unchanged")

    def test_empty_repo_never_clobbers_a_populated_local_file(self):
        result, local_text = self._run_pull(local_entries=3, remote_entries=0)
        self.assertFalse(result["pulled"])
        self.assertEqual(result["reason"], "remote_empty")
        self.assertIn("## Local 0", local_text)

    def test_missing_remote_file_never_clobbers_a_populated_local_file(self):
        result, local_text = self._run_pull(local_entries=3, remote_entries=0, remote_exists=False)
        self.assertFalse(result["pulled"])
        self.assertEqual(result["reason"], "remote_missing")
        self.assertIn("## Local 0", local_text)


class SyncTests(unittest.TestCase):
    """sync_verified_qa_pairs() reconciles in whichever direction has more
    entries: pull when the repo is ahead, push when local is ahead (including
    a repo that has no file or no entries yet)."""

    def _run_sync(self, pull_result):
        with patch.object(github_sync, "pull_verified_qa_pairs", return_value=pull_result), patch.object(
            github_sync, "push_verified_qa_pairs", return_value={"pushed": True}
        ) as push:
            return github_sync.sync_verified_qa_pairs(), push

    def test_repo_ahead_pulls_and_does_not_push(self):
        result, push = self._run_sync({"pulled": True, "entry_count": 638})
        self.assertEqual(result["direction"], "pull")
        push.assert_not_called()

    def test_local_ahead_pushes(self):
        result, push = self._run_sync(
            {"pulled": False, "reason": "local_ahead", "local_entry_count": 640, "remote_entry_count": 638}
        )
        self.assertEqual(result["direction"], "push")
        self.assertEqual(result["push"], {"pushed": True})
        push.assert_called_once_with()

    def test_populated_local_with_missing_or_empty_repo_pushes(self):
        for reason in ("remote_missing", "remote_empty"):
            with self.subTest(reason=reason):
                result, push = self._run_sync({"pulled": False, "reason": reason, "local_entry_count": 3})
                self.assertEqual(result["direction"], "push")
                push.assert_called_once_with()

    def test_both_sides_empty_does_nothing(self):
        result, push = self._run_sync({"pulled": False, "reason": "remote_missing", "local_entry_count": 0})
        self.assertEqual(result["direction"], "none")
        push.assert_not_called()

    def test_identical_does_nothing(self):
        result, push = self._run_sync(
            {"pulled": False, "reason": "unchanged", "entry_count": 5, "local_entry_count": 5}
        )
        self.assertEqual(result["direction"], "none")
        push.assert_not_called()

    def test_push_failure_propagates(self):
        with patch.object(
            github_sync,
            "pull_verified_qa_pairs",
            return_value={"pulled": False, "reason": "local_ahead", "local_entry_count": 2},
        ), patch.object(github_sync, "push_verified_qa_pairs", side_effect=RuntimeError("push rejected")):
            with self.assertRaises(RuntimeError):
                github_sync.sync_verified_qa_pairs()


class CountMarkdownEntriesTests(unittest.TestCase):
    def test_counts_heading_blocks(self):
        text = "## One\n\nBody.\n\n## Two\n\nBody.\n\n## Three\n\nBody.\n"
        self.assertEqual(github_sync._count_markdown_entries(text), 3)

    def test_zero_for_empty_or_headingless_text(self):
        self.assertEqual(github_sync._count_markdown_entries(""), 0)
        self.assertEqual(github_sync._count_markdown_entries("just prose, no headings"), 0)


class LoadEnvTests(unittest.TestCase):
    def test_load_env_overlays_process_environment_over_empty_file(self):
        with patch.object(
            github_sync,
            "dotenv_values",
            return_value={"GITHUB_TOKEN": "", "GITHUB_REPO_URL": "https://github.com/org/repo.git"},
        ), patch.dict(
            os.environ,
            {"GITHUB_TOKEN": "from-env-secret", "GITHUB_REPO_URL": "https://github.com/org/repo.git"},
            clear=False,
        ):
            env = github_sync.load_env()
        self.assertEqual(env["GITHUB_TOKEN"], "from-env-secret")
        self.assertEqual(env["GITHUB_REPO_URL"], "https://github.com/org/repo.git")

    def test_load_env_raises_when_token_missing_in_file_and_environment(self):
        with patch.object(
            github_sync,
            "dotenv_values",
            return_value={"GITHUB_TOKEN": "", "GITHUB_REPO_URL": ""},
        ), patch.dict(os.environ, {"GITHUB_TOKEN": "", "GITHUB_REPO_URL": ""}, clear=False):
            with self.assertRaises(RuntimeError) as ctx:
                github_sync.load_env()
        self.assertIn("GITHUB_TOKEN", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
