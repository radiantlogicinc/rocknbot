"""
github_sync.py
====================================
Pushes the verified-techsupport markdown file (techsupport_qa_pairs.md) to a
dedicated private GitHub repo whenever nightly_pipeline.py's run has actually
changed it -- resolves techsupport_qa_ingest.py's FLAGGED OPEN ITEM (there is
now a real repo, unlike the golden QA pairs situation that comment
describes).

Clones the repo fresh into a temp directory on every call rather than keeping
a persistent local clone -- same clone-fresh-every-time pattern src/main.py
already uses for the golden QA pairs repo (QA_PAIRS_GITHUB_REPO_URL) -- so
there's no local git state that can drift or get corrupted between runs.
Credentials are passed via GIT_ASKPASS (never embedded in the clone URL or
written into .git/config). The work dir is deleted immediately after use.

Required env vars (read from cron/env/github_push.env):
    GITHUB_TOKEN     - a GitHub personal access token with push access to
                        GITHUB_REPO_URL's repo
    GITHUB_REPO_URL  - HTTPS URL of the destination repo, e.g.
                        https://github.com/<org>/<repo>.git (no embedded
                        credentials)

Shrink guard: before copying the local markdown over the repo's copy,
push_verified_qa_pairs() compares entry counts and raises ShrinkGuardError
if the local file has fewer "## " entries than the repo copy already has.
The pipeline only ever appends an entry or rewrites one in place, so a
lower local count means the local file was lost or truncated (a state-folder
path change, a fresh pod, a crashed partial write), not a real edit.
filecmp.cmp() alone only answers "did the content change": a one-entry local
file still counts as changed and, without this guard, would be pushed over
the repo's full history. An equal count with rewritten text (replace, enrich,
correct) and any increase still push. Deleting an entry from the local file
is not a supported operation, so there is no threshold and no override.

Usage (as a library, e.g. from nightly_pipeline.py):
    from github_sync import push_verified_qa_pairs
    result = push_verified_qa_pairs()

pull_verified_qa_pairs() is the opposite-direction counterpart: GITHUB_REPO_URL
is the source of truth, so techsupport_bootstrap.py clones it at startup and
rebuilds TECHSUPPORT_QA_PAIRS from it -- the same contract src/main.py's
_run_update_golden_qa_pairs_task() applies to the golden QA pairs repo. The
local file is a working copy the pipeline appends to between pushes.
sync_verified_qa_pairs() combines the two for startup: whichever side has
more entries wins, so unpushed local entries are pushed rather than lost.

Usage (standalone):
    python github_sync.py
"""

import filecmp
import logging
import os
import re
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit

import git
from dotenv import dotenv_values
from paths import LILLISA_SERVER_ROOT, PACKAGE_ROOT, VERIFIED_TECHSUPPORT_QA_FOLDERPATH, ensure_import_paths

ensure_import_paths()
SCRIPT_DIR = PACKAGE_ROOT
PROJECT_ROOT = LILLISA_SERVER_ROOT
ENV_PATH = SCRIPT_DIR / "env" / "github_push.env"

# Deliberately duplicated from techsupport_qa_ingest.py rather than imported
# from it -- that module's top-level imports pull in the full LanceDB /
# llama_index / voyageai runtime stack (and the server env setup that
# requires), which this lightweight, standalone-runnable script has no need
# for. VERIFIED_TECHSUPPORT_QA_FOLDERPATH comes from the lightweight paths.py.
TECHSUPPORT_QA_MARKDOWN_FILENAME = "techsupport_qa_pairs.md"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("github_sync")

REQUIRED_ENV_VARS = ("GITHUB_TOKEN", "GITHUB_REPO_URL")


class ShrinkGuardError(RuntimeError):
    """Raised by push_verified_qa_pairs() when the local verified-techsupport
    markdown has fewer entries than the repo's current copy. The pipeline
    never deletes an entry, so a lower count means the local file was lost
    or truncated. Pushing it would overwrite the repo. The caller
    (nightly_pipeline.py's publish_changes_if_any()) logs any exception from
    this function and alerts via post_admin_alert() without failing the
    whole pipeline run, since the LanceDB write it follows already succeeded."""

# Explicit commit identity so this doesn't depend on whatever git config
# happens to be set on the machine running the nightly pipeline.
COMMIT_AUTHOR = git.Actor("LilLisa Techsupport Pipeline", "noreply@radiantlogic.com")


def load_env() -> Dict[str, str]:
    """Reads GITHUB_TOKEN / GITHUB_REPO_URL from env/github_push.env, then
    overlays os.environ so cron/k8s secrets win over empty file placeholders."""
    env = dict(dotenv_values(str(ENV_PATH)))
    env = {**env, **os.environ}
    missing = [key for key in REQUIRED_ENV_VARS if not env.get(key)]
    if missing:
        raise RuntimeError(
            f"Missing required env var(s) {missing} - expected in {ENV_PATH} or the process environment"
        )
    return env


def _require_https_repo_url(repo_url: str) -> str:
    """GIT_ASKPASS only works for HTTPS. Reject git:// and ssh:// so we never
    silently fall back to a scheme that can't use GITHUB_TOKEN."""
    parts = urlsplit(repo_url)
    if parts.scheme != "https":
        raise ValueError(f"GITHUB_REPO_URL must be an https:// URL, got: {repo_url}")
    if parts.username or parts.password:
        raise ValueError("GITHUB_REPO_URL must not include credentials; set GITHUB_TOKEN instead")
    return repo_url


def _write_git_askpass(directory: Path) -> Path:
    """Helper Git will exec for Username/Password prompts. Always prints
    $GITHUB_TOKEN (GitHub accepts the PAT as either field). The token is not
    written into this file."""
    path = Path(directory) / "git-askpass.sh"
    path.write_text(
        "#!/bin/sh\n" "printf '%s\\n' \"$GITHUB_TOKEN\"\n",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


def _git_auth_env(token: str, askpass_path: Path) -> dict:
    env = os.environ.copy()
    env["GIT_ASKPASS"] = str(askpass_path)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"
    env["GITHUB_TOKEN"] = token
    return env


def _count_markdown_entries(markdown_text: str) -> int:
    """Count '## ' heading blocks. Same split pattern
    techsupport_markdown.parse_summary_markdown() uses, duplicated here (not
    imported) to keep this script's only real dependencies git + dotenv --
    see the module docstring's note on why TECHSUPPORT_QA_MARKDOWN_FILENAME
    etc. are duplicated rather than imported from techsupport_qa_ingest.py."""
    return len(re.findall(r"(?m)^## ", markdown_text))


def _check_shrink_guard(source_path: Path, dest_path: Path) -> None:
    """Raise ShrinkGuardError if source_path has fewer entries than dest_path
    (the repo's current copy). A no-op when dest_path doesn't exist yet or
    has zero entries: a first push into an empty repo has nothing to lose."""
    if not dest_path.exists():
        return
    repo_entry_count = _count_markdown_entries(dest_path.read_text(encoding="utf-8"))
    if repo_entry_count <= 0:
        return
    source_entry_count = _count_markdown_entries(source_path.read_text(encoding="utf-8"))
    if source_entry_count < repo_entry_count:
        raise ShrinkGuardError(
            f"Refusing to push {TECHSUPPORT_QA_MARKDOWN_FILENAME}: local copy has "
            f"{source_entry_count} entries and the repo already has {repo_entry_count}. "
            "The pipeline only appends entries or rewrites one in place, so a lower "
            "local count means the local file was lost or truncated. Pushing it would "
            "overwrite the repo. Restore the local file (techsupport_bootstrap.py pulls "
            "it back when the local copy is empty) before the next run."
        )


def push_verified_qa_pairs() -> Dict[str, Any]:
    """Clones GITHUB_REPO_URL fresh into a temp dir, copies the current
    techsupport_qa_pairs.md over the repo's copy, and commits+pushes only if
    the content actually changed. Safe to call even when nothing changed
    (returns pushed=False rather than erroring)."""
    env = load_env()
    token = env["GITHUB_TOKEN"]
    repo_url = env["GITHUB_REPO_URL"]

    source_path = VERIFIED_TECHSUPPORT_QA_FOLDERPATH / TECHSUPPORT_QA_MARKDOWN_FILENAME
    if not source_path.exists():
        logger.info("No %s found at %s -- nothing to push", TECHSUPPORT_QA_MARKDOWN_FILENAME, source_path)
        return {"pushed": False, "reason": "source_missing"}

    repo_url = _require_https_repo_url(repo_url)

    work_dir = tempfile.mkdtemp(prefix="techsupport_github_sync_")
    try:
        # Askpass lives next to the clone, not inside it — git clone requires an
        # empty destination directory.
        askpass_path = _write_git_askpass(Path(work_dir))
        git_env = _git_auth_env(token, askpass_path)
        clone_dir = os.path.join(work_dir, "repo")
        logger.info("Cloning %s into %s", repo_url, clone_dir)
        repo = git.Repo.clone_from(
            repo_url,
            clone_dir,
            env=git_env,
            # Do not let a global credential.helper persist the PAT to disk.
            multi_options=["--config", "credential.helper="],
            # GitPython >= 3.1.31 lists --config as an "unsafe" clone option and
            # raises UnsafeOptionError without this flag (confirmed against the
            # pinned 3.1.43). The value above is a fixed literal, not user input.
            allow_unsafe_options=True,
        )

        dest_path = Path(clone_dir) / TECHSUPPORT_QA_MARKDOWN_FILENAME
        if dest_path.exists() and filecmp.cmp(source_path, dest_path, shallow=False):
            logger.info("%s unchanged -- skipping commit/push", TECHSUPPORT_QA_MARKDOWN_FILENAME)
            return {"pushed": False, "reason": "unchanged"}

        # Must run BEFORE the copy below overwrites dest_path with source_path's
        # content -- it needs to compare the two as they stood before this push.
        _check_shrink_guard(source_path, dest_path)

        shutil.copyfile(source_path, dest_path)
        repo.index.add([TECHSUPPORT_QA_MARKDOWN_FILENAME])

        commit_message = f"Update verified techsupport QA pairs - {date.today().isoformat()}"
        repo.index.commit(commit_message, author=COMMIT_AUTHOR, committer=COMMIT_AUTHOR)
        with repo.git.custom_environment(**git_env):
            repo.remote(name="origin").push()

        logger.info("Pushed commit: %s", commit_message)
        return {"pushed": True, "commit_message": commit_message}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def pull_verified_qa_pairs() -> Dict[str, Any]:
    """Clones GITHUB_REPO_URL fresh into a temp dir and copies its
    techsupport_qa_pairs.md down over the local copy.

    GITHUB_REPO_URL is the source of truth for verified-techsupport content,
    the same contract src/main.py's _run_update_golden_qa_pairs_task()
    applies to the golden QA pairs repo: the repo is cloned at startup and
    the LanceDB table is rebuilt from it (see techsupport_bootstrap.py, which
    calls this). The local file is a working copy that the pipeline appends
    to between pushes, not the authority.

    When the local file has MORE entries than the repo copy (including a
    repo with no file or no entries at all), the local file is left alone
    and the result says so via a reason in LOCAL_AHEAD_REASONS. A higher
    local count can only mean entries the pipeline added but has not pushed
    yet (the pipeline never deletes an entry), so overwriting would discard
    verified content that exists nowhere else -- exactly what a silently
    failing push leaves behind. sync_verified_qa_pairs() turns that result
    into a push.

    Returns {"pulled": False, "reason": ..., "local_entry_count": ...}
    whenever nothing was pulled.
    """
    env = load_env()
    token = env["GITHUB_TOKEN"]
    repo_url = _require_https_repo_url(env["GITHUB_REPO_URL"])

    local_path = VERIFIED_TECHSUPPORT_QA_FOLDERPATH / TECHSUPPORT_QA_MARKDOWN_FILENAME
    local_entry_count = (
        _count_markdown_entries(local_path.read_text(encoding="utf-8")) if local_path.exists() else 0
    )

    work_dir = tempfile.mkdtemp(prefix="techsupport_github_pull_")
    try:
        askpass_path = _write_git_askpass(Path(work_dir))
        git_env = _git_auth_env(token, askpass_path)
        clone_dir = os.path.join(work_dir, "repo")
        logger.info("Cloning %s into %s (pull)", repo_url, clone_dir)
        git.Repo.clone_from(
            repo_url,
            clone_dir,
            env=git_env,
            multi_options=["--config", "credential.helper="],
            allow_unsafe_options=True,
        )

        remote_path = Path(clone_dir) / TECHSUPPORT_QA_MARKDOWN_FILENAME
        if not remote_path.exists():
            logger.info("Repo has no %s yet -- nothing to pull", TECHSUPPORT_QA_MARKDOWN_FILENAME)
            return {"pulled": False, "reason": "remote_missing", "local_entry_count": local_entry_count}

        remote_entry_count = _count_markdown_entries(remote_path.read_text(encoding="utf-8"))
        if remote_entry_count <= 0:
            logger.info("Repo's %s has no entries -- nothing to pull", TECHSUPPORT_QA_MARKDOWN_FILENAME)
            return {"pulled": False, "reason": "remote_empty", "local_entry_count": local_entry_count}

        if local_entry_count > remote_entry_count:
            logger.warning(
                "Local %s has %d entries but the repo has %d -- keeping the local copy. The "
                "pipeline never deletes an entry, so the extra local entries are unpushed and "
                "would be lost by pulling; they need to be pushed instead.",
                TECHSUPPORT_QA_MARKDOWN_FILENAME,
                local_entry_count,
                remote_entry_count,
            )
            return {
                "pulled": False,
                "reason": "local_ahead",
                "local_entry_count": local_entry_count,
                "remote_entry_count": remote_entry_count,
            }

        if local_path.exists() and filecmp.cmp(remote_path, local_path, shallow=False):
            return {
                "pulled": False,
                "reason": "unchanged",
                "entry_count": remote_entry_count,
                "local_entry_count": local_entry_count,
            }

        VERIFIED_TECHSUPPORT_QA_FOLDERPATH.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote_path, local_path)
        logger.info(
            "Pulled %s (%d entries) from %s", TECHSUPPORT_QA_MARKDOWN_FILENAME, remote_entry_count, repo_url
        )
        return {"pulled": True, "entry_count": remote_entry_count}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


LOCAL_AHEAD_REASONS = frozenset({"local_ahead", "remote_missing", "remote_empty"})


def sync_verified_qa_pairs() -> Dict[str, Any]:
    """Two-way reconcile of techsupport_qa_pairs.md between the local copy
    and GITHUB_REPO_URL, run at startup by techsupport_bootstrap.py. The
    side with more entries wins: the pipeline never deletes an entry, so the
    larger file is always a superset of the smaller one.

      * repo ahead (or equal count, different text): pull_verified_qa_pairs()
        copies the repo down -- the repo is the source of truth.
      * local ahead (including a repo with no file or no entries yet):
        push_verified_qa_pairs() publishes the unpushed local entries.

    Exceptions from either direction propagate (including push's
    ShrinkGuardError); the caller decides whether they are fatal.
    """
    pull_result = pull_verified_qa_pairs()
    if pull_result.get("reason") not in LOCAL_AHEAD_REASONS or pull_result.get("local_entry_count", 0) <= 0:
        return {"direction": "pull" if pull_result.get("pulled") else "none", "pull": pull_result}

    logger.info(
        "Local %s is ahead of the repo (%s) -- pushing local entries",
        TECHSUPPORT_QA_MARKDOWN_FILENAME,
        pull_result["reason"],
    )
    return {"direction": "push", "pull": pull_result, "push": push_verified_qa_pairs()}


if __name__ == "__main__":
    result = push_verified_qa_pairs()
    print(result)
