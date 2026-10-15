"""
golden_qa_sync.py
====================================
Makes expert thumbs-up QA pairs durable.

POST /add_expert_qa_pair/ appends the pair to the {PRODUCT}_QA_PAIRS LanceDB
table, but _run_update_golden_qa_pairs_task() drops that table and rebuilds it
from the markdown files in QA_PAIRS_GITHUB_REPO_URL. Anything that only ever
reached LanceDB is therefore silently lost on the next rebuild. This module
pushes the pair back to that repo so the rebuild stays a pure function of the
repo.

The markdown format written here is exactly what the rebuild parser reads:

    # Question/Answer Pair
    Question: <question>
    Answer: <answer>

(entries separated by blank lines; the rebuild splits on the
"# Question/Answer Pair" header and parses with QA_ENTRY_PATTERN below).

Auth: the repo is cloned over HTTPS with the PAT supplied through GIT_ASKPASS
-- never embedded in the URL, and with credential.helper= so nothing is
persisted to disk. The same GITHUB_TOKEN that pushes the techsupport markdown
(cron/github_sync.py) pushes here too. It is read from
LilLisa_Server/cron/env/github_push.env, then overlaid by os.environ.

Deliberately free of heavy imports (no lancedb / llama_index / torch) so it
can be unit tested and driven from scripts/backfill_expert_qa_pairs.py
without standing up the server.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import git
from dotenv import dotenv_values

from src import utils

SERVER_ROOT = Path(__file__).resolve().parent.parent
GITHUB_PUSH_ENV_PATH = SERVER_ROOT / "cron" / "env" / "github_push.env"

QA_PAIR_HEADER = "# Question/Answer Pair"
# Same pattern _run_update_golden_qa_pairs_task() in src/main.py uses.
QA_ENTRY_PATTERN = re.compile(r"Question:\s*(.*?)\nAnswer:\s*(.*)", re.DOTALL)

# Explicit identity so the commit does not depend on the git config of
# whatever host the API process happens to run on (same reasoning as
# cron/github_sync.COMMIT_AUTHOR).
COMMIT_AUTHOR = git.Actor("LilLisa Expert QA", "noreply@radiantlogic.com")


def markdown_filename(product: str) -> str:
    """The per-product file the rebuild task reads: iddm_qa_pairs.md etc."""
    return f"{product.lower()}_qa_pairs.md"


def format_golden_qa_entry(question: str, answer: str) -> str:
    """Render one entry in exactly the shape the rebuild parser expects.

    Ends with a trailing blank line so entries stay separated when appended
    one after another."""
    question = (question or "").strip()
    answer = (answer or "").strip()
    return f"{QA_PAIR_HEADER}\nQuestion: {question}\nAnswer: {answer}\n\n"


def parse_golden_qa_entries(file_content: str) -> List[Tuple[str, str]]:
    """Parse a *_qa_pairs.md file the way the rebuild task does.

    Kept here (rather than imported from main.py) so the backfill script and
    the tests can read the repo files without importing the server."""
    pairs: List[Tuple[str, str]] = []
    for chunk in file_content.split(QA_PAIR_HEADER):
        chunk = chunk.strip()
        if not chunk:
            continue
        if match := QA_ENTRY_PATTERN.search(chunk):
            pairs.append((match[1].strip(), match[2].strip()))
    return pairs


def load_github_token() -> Optional[str]:
    """GITHUB_TOKEN from cron/env/github_push.env, overlaid by os.environ
    (so a container secret beats a file placeholder). The same token pushes
    the golden QA repo and the techsupport markdown repo."""
    env = dict(dotenv_values(str(GITHUB_PUSH_ENV_PATH)))
    env = {**env, **os.environ}
    token = (env.get("GITHUB_TOKEN") or "").strip()
    return token or None


def resolve_repo_url() -> Optional[str]:
    """QA_PAIRS_GITHUB_REPO_URL, from the server env file overlaid by
    os.environ (utils.LILLISA_SERVER_ENV_DICT already does that overlay)."""
    url = (utils.LILLISA_SERVER_ENV_DICT.get("QA_PAIRS_GITHUB_REPO_URL") or "").strip()
    return url or None


def _needs_auth(repo_url: str) -> bool:
    """True for the real https:// remote; False for a local path or file://
    remote (used by the tests, and usable for a local mirror).

    Raises for any other scheme: GIT_ASKPASS only works over HTTPS, so a
    ssh:// or git:// URL would silently fail to authenticate, and a URL with
    embedded credentials would leak the PAT into .git/config and error logs."""
    parts = urlsplit(repo_url)
    if parts.scheme == "file":
        return False
    if not parts.scheme:
        # A bare filesystem path is fine; anything else with no scheme is an
        # scp-style ssh remote (git@github.com:org/repo.git), which must not
        # be mistaken for a local path.
        if repo_url.startswith(("/", "./", "../")):
            return False
        raise ValueError(f"QA_PAIRS_GITHUB_REPO_URL must be an https:// URL, got: {repo_url}")
    if parts.scheme != "https":
        raise ValueError(f"QA_PAIRS_GITHUB_REPO_URL must be an https:// URL, got: {repo_url}")
    if parts.username or parts.password:
        raise ValueError(
            "QA_PAIRS_GITHUB_REPO_URL must not include credentials; "
            "set GITHUB_TOKEN instead"
        )
    return True


# The next two helpers duplicate cron/github_sync._write_git_askpass and
# ._git_auth_env. They are re-implemented rather than imported because that
# module is part of the flat-import cron package (it imports `paths`, which
# only resolves once cron/ is on sys.path) and its load_env() also demands
# GITHUB_REPO_URL -- the techsupport repo, not this one. Keep the two in sync;
# the original is the reference implementation.
def _write_git_askpass(directory: Path) -> Path:
    """Helper Git execs for Username/Password prompts. Prints $GITHUB_TOKEN
    (GitHub accepts a PAT as either field). The token is not written here."""
    path = Path(directory) / "git-askpass.sh"
    path.write_text("#!/bin/sh\nprintf '%s\\n' \"$GITHUB_TOKEN\"\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _git_auth_env(token: str, askpass_path: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["GIT_ASKPASS"] = str(askpass_path)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "never"
    env["GITHUB_TOKEN"] = token
    return env


@contextmanager
def cloned_repo(repo_url: Optional[str] = None, token: Optional[str] = None):
    """Shallow-clone the golden QA repo into a temp dir, deleted on exit.

    Yields (git.Repo, clone_dir, git_env) where git_env is None for a local /
    file:// remote that needs no credentials."""
    repo_url = repo_url or resolve_repo_url()
    if not repo_url:
        raise RuntimeError("QA_PAIRS_GITHUB_REPO_URL is not configured")

    needs_auth = _needs_auth(repo_url)
    git_env: Optional[Dict[str, str]] = None

    work_dir = tempfile.mkdtemp(prefix="golden_qa_sync_")
    try:
        if needs_auth:
            token = token or load_github_token()
            if not token:
                raise RuntimeError(
                    "No GitHub token found -- set GITHUB_TOKEN "
                    f"in {GITHUB_PUSH_ENV_PATH} or the process environment"
                )
            # Askpass lives next to the clone: git clone needs an empty target.
            git_env = _git_auth_env(token, _write_git_askpass(Path(work_dir)))

        clone_dir = os.path.join(work_dir, "repo")
        utils.logger.info("Golden QA sync: cloning %s into %s", repo_url, clone_dir)
        clone_kwargs: Dict[str, Any] = {
            # Shallow: this runs inline on a Slack request, and we only ever
            # append to the tip.
            "depth": 1,
            # Do not let a global credential.helper persist the PAT to disk.
            "multi_options": ["--config", "credential.helper="],
            # GitPython >= 3.1.31 treats --config as an "unsafe" clone option
            # and refuses it unless this is set (UnsafeOptionError). The value
            # here is a fixed literal, not user input.
            "allow_unsafe_options": True,
        }
        if git_env:
            clone_kwargs["env"] = git_env
        repo = git.Repo.clone_from(repo_url, clone_dir, **clone_kwargs)
        yield repo, clone_dir, git_env
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def read_repo_qa_pairs(
    product: str,
    repo_url: Optional[str] = None,
    token: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """The (question, answer) pairs currently in the repo for one product.

    Used by scripts/backfill_expert_qa_pairs.py to skip pairs that are already
    there. Raises on clone failure -- the backfill wants to know."""
    with cloned_repo(repo_url, token) as (_repo, clone_dir, _git_env):
        path = Path(clone_dir) / markdown_filename(product)
        if not path.exists():
            return []
        return parse_golden_qa_entries(path.read_text(encoding="utf-8"))


def append_expert_qa_pair_to_repo(
    product: str,
    question: str,
    answer: str,
    repo_url: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Append one expert-verified pair to {product}_qa_pairs.md in the golden
    QA repo and push it.

    Never raises: the caller has already written the pair to LanceDB and that
    insert must stand even when the push fails. Returns
    {"pushed": True, "commit_message": ..., "filename": ...} on success and
    {"pushed": False, "error": ...} otherwise."""
    question = (question or "").strip()
    answer = (answer or "").strip()
    if not question or not answer:
        return {"pushed": False, "error": "question and answer are both required"}

    try:
        with cloned_repo(repo_url, token) as (repo, clone_dir, git_env):
            return _commit_and_push_entry(repo, clone_dir, git_env, product, question, answer)
    except Exception as exc:  # noqa: BLE001 -- a failed push must not fail the request
        utils.logger.error(
            "Golden QA sync: failed to push expert QA pair for %s: %s", product, exc, exc_info=True
        )
        return {"pushed": False, "error": f"{type(exc).__name__}: {exc}"}


def _commit_and_push_entry(
    repo: "git.Repo",
    clone_dir: str,
    git_env: Optional[Dict[str, str]],
    product: str,
    question: str,
    answer: str,
) -> Dict[str, Any]:
    """Write the entry into the clone, commit it, push it."""
    filename = markdown_filename(product)
    dest_path = Path(clone_dir) / filename
    entry = format_golden_qa_entry(question, answer)
    if dest_path.exists():
        existing = dest_path.read_text(encoding="utf-8")
        # Guarantee a blank line between the previous entry and this one.
        if not existing.strip() or existing.endswith("\n\n"):
            separator = ""
        elif existing.endswith("\n"):
            separator = "\n"
        else:
            separator = "\n\n"
        dest_path.write_text(existing + separator + entry, encoding="utf-8")
    else:
        dest_path.write_text(entry, encoding="utf-8")

    repo.index.add([filename])
    commit_message = f"Add expert-verified QA pair ({product}) - {date.today().isoformat()}"
    repo.index.commit(commit_message, author=COMMIT_AUTHOR, committer=COMMIT_AUTHOR)
    if git_env:
        with repo.git.custom_environment(**git_env):
            _push(repo)
    else:
        _push(repo)

    utils.logger.info("Golden QA sync: pushed %s (%s)", filename, commit_message)
    return {"pushed": True, "commit_message": commit_message, "filename": filename}


def _push(repo: "git.Repo") -> None:
    """Push origin and turn a *rejected* push into an exception -- GitPython
    reports rejections in the PushInfo flags rather than raising."""
    for info in repo.remote(name="origin").push():
        if info.flags & info.ERROR:
            raise RuntimeError(f"push rejected: {(info.summary or '').strip()}")
