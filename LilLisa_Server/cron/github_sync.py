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

Usage (as a library, e.g. from nightly_pipeline.py):
    from github_sync import push_verified_qa_pairs
    result = push_verified_qa_pairs()

Usage (standalone):
    python github_sync.py
"""

import filecmp
import logging
import os
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


if __name__ == "__main__":
    result = push_verified_qa_pairs()
    print(result)
