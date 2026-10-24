"""
techsupport_preflight.py
====================================
Startup preflight for the in-process techsupport pipeline: checks, once and
up front, that every piece of configuration and every writable directory the
nightly pipeline depends on is actually in place, and turns any gap into one
prescriptive, secret-safe report instead of an hourly stack trace nobody reads.

Why this exists -- the staging incident it prevents:
  The API process runs the techsupport pipeline in-process
  (src/techsupport_cron.py ticks every TECHSUPPORT_PIPELINE_TICK_MINUTES). On
  staging, every tick for hours raised

      RuntimeError: Missing required env var(s) ['SLACK_BOT_TOKEN',
      'TECHSUPPORT_CHANNEL_ID'] - expected in /app/cron/env/techsupport_sync.env

  while the API kept answering questions and Kubernetes reported a healthy
  rollout. dockerfile_prod deliberately ships no cron/env/*.env files (they
  hold secrets), so a deployment that forgets to mount them is
  indistinguishable from a healthy one at the pod level: the pipeline was 100%
  dead and nothing failed the rollout. The only signal was a per-tick
  traceback naming two keys at a time -- fix those, redeploy, and the next run
  would have surfaced the next missing key, one rollout per problem.

  So this module:
    * validates everything at once and COLLECTS every problem (never raises on
      the first), so devops fixes one rollout rather than N sequential ones;
    * validates the RESOLVED configuration -- env-file values overlaid by
      os.environ, exactly the idiom every loader in this package already uses
      (nightly_techsupport_sync.load_env(), github_sync.load_env(),
      nightly_pipeline._lillisa_env(), ...) -- not mere file existence.
      Injecting container environment variables instead of mounting the env
      files is a documented, supported deployment route, so a value present
      only in os.environ satisfies validation even with no .env file at all;
    * splits problems into two severity tiers (see Severity below), so that
      a pipeline that cannot ingest, cannot push the verified markdown, or
      cannot put clickable source links on answers becomes a failed rollout
      in strict mode;
    * never echoes a value. Problems and the formatted report carry only KEY
      NAMES and filesystem PATHS -- no tokens, no os.environ dump, no URL that
      could contain credentials -- because this text goes straight into
      CRITICAL logs that are shipped and retained far more widely than the
      Secret it describes.

Deliberately self-contained (stdlib + dotenv only): no FastAPI, LanceDB,
llama_index, voyageai, dspy, requests, git, or src.* imports, for the same
reason nightly_techsupport_sync.py and github_anchor.py are standalone -- it
must be cheap to import at server startup and trivially unit-testable. That is
also why the GITHUB_REPO_URL rule (github_sync._require_https_repo_url) is
re-stated here rather than imported: importing that module would drag in
GitPython. Keep the copy aligned with the original.

What is intentionally NOT required:
  * The TECHSUPPORT_QA_PAIRS LanceDB table and techsupport_qa_pairs.md -- both
    are created by the pipeline's first successful run. A brand-new, correctly
    configured deployment must produce zero problems.
  * The state files (techsupport_sync_state.json, techsupport_reembed_state.json,
    techsupport_review_state.json) -- same reason; only their directory must be
    writable, since atomic_io writes a temp file beside them and os.replace()s.
    That directory is the techsupport state folder (src/techsupport_state_paths.py),
    by default techsupport_state/ next to LANCEDB_FOLDERPATH, i.e. on the same
    volume as the table.

Usage (the startup wiring lives in src/main.py, not here):
    from techsupport_preflight import run_preflight, format_preflight_report, is_strict_mode
    result = run_preflight()
    if result.problems:
        report = format_preflight_report(result, strict=is_strict_mode())
        ...log it; refuse to start if is_strict_mode() and not result.ok...
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from dotenv import dotenv_values
from paths import (
    LILLISA_SERVER_ENV_PATH,
    LILLISA_SERVER_ROOT,
    PACKAGE_ROOT,
    bot_env_path,
    read_lil_lisa_channel_ids,
)

from src.techsupport_state_paths import resolve_state_dir, resolve_verified_qa_dir


# Stable, machine-greppable code for alert routing. Do not reword: log-based
# alerts match on this exact string.
FAILURE_CODE = "TECHSUPPORT_CONFIG_MISSING"
# Emitted instead of FAILURE_CODE when every problem is DEGRADED, so a
# deployment whose only gaps are a working default or a skipped product-channel
# scan does not page on every boot through the FAILURE_CODE alert route.
DEGRADED_CODE = "TECHSUPPORT_CONFIG_DEGRADED"

STRICT_ENV_KEY = "TECHSUPPORT_PREFLIGHT_STRICT"
# Fail closed: a deployment that never heard of this knob gets the protective
# behaviour. Dev boxes opt out with TECHSUPPORT_PREFLIGHT_STRICT=false in their
# gitignored env/lillisa_server.env (or the process environment).
DEFAULT_STRICT = True
_TRUTHY = frozenset({"true", "1", "yes", "on"})
_FALSEY = frozenset({"false", "0", "no", "off"})

# Same three files the loaders read; the names match the Secret keys devops
# mounts with subPath.
SYNC_ENV_PATH = PACKAGE_ROOT / "env" / "techsupport_sync.env"
GITHUB_ENV_PATH = PACKAGE_ROOT / "env" / "github_push.env"

PRODUCTS: Tuple[str, ...] = ("IDA", "IDDM", "IDO")
# Products the product-channel scan cannot run without (expert_group.REQUIRED_PRODUCTS:
# the resolver raises ValueError when their group id is unset). IDO is only
# deployed for some workspaces, so its keys stay optional.
REQUIRED_SCAN_PRODUCTS: Tuple[str, ...] = ("IDA", "IDDM")
# Mirrors nightly_pipeline.DEFAULT_SCAN_PRODUCT_CHANNELS / FALSEY exactly --
# including "" meaning false -- so preflight and the pipeline always agree on
# whether the product-channel pass will run.
_DEFAULT_SCAN_PRODUCT_CHANNELS = "true"
_SCAN_FALSEY = frozenset({"false", "0", "no", "off", ""})


class Severity(str, Enum):
    """How bad a problem is, which decides whether strict mode refuses startup.

    BLOCKING: the pipeline cannot function at all, or would misbehave (e.g.
      ingest into LanceDB but never get the new rows reloaded into live
      retrieval). In strict mode any BLOCKING problem refuses startup -- this
      is the tier that turns the staging incident into a failed rollout.
    DEGRADED: the techsupport channel can still be ingested and served. What is
      lost is a product-channel correction scan, or a setting that already has
      a working default (LIL_LISA_SERVER_URL). Always a warning, never fatal.
    """

    BLOCKING = "BLOCKING"
    DEGRADED = "DEGRADED"


@dataclass(frozen=True)
class PreflightProblem:
    """One thing wrong with the deployment.

    keys: the config key name(s) involved (never their values).
    path: the filesystem path involved, for writability problems (paths are
      not secrets and the operator needs the exact one).
    problem: short statement of what is wrong.
    remediation: the concrete fix, phrased for whoever owns the deployment.
    """

    severity: Severity
    keys: Tuple[str, ...]
    problem: str
    remediation: str
    path: Optional[str] = None


@dataclass(frozen=True)
class PreflightResult:
    """Every problem found, plus the configuration sources that were consulted
    (so the report can tell the operator exactly where values are read from)."""

    problems: Tuple[PreflightProblem, ...]
    sources: Tuple[Path, ...]

    @property
    def blocking(self) -> Tuple[PreflightProblem, ...]:
        """Problems that refuse startup in strict mode."""
        return tuple(p for p in self.problems if p.severity is Severity.BLOCKING)

    @property
    def degraded(self) -> Tuple[PreflightProblem, ...]:
        """Problems that only ever warn."""
        return tuple(p for p in self.problems if p.severity is Severity.DEGRADED)

    @property
    def ok(self) -> bool:  # pylint: disable=invalid-name
        """True when nothing BLOCKING was found (DEGRADED problems may remain)."""
        return not self.blocking


def _clean(value: Optional[str]) -> Optional[str]:
    """Empty / whitespace-only counts as missing: the shipped .env.example
    files carry blank placeholders such as `GITHUB_TOKEN=`."""
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _read_env_file(path: Path) -> Dict[str, str]:
    """dotenv_values(path) minus None values; an absent file is simply empty,
    because env injection is a supported alternative to mounting it."""
    if not path.is_file():
        return {}
    return {k: v for k, v in dotenv_values(str(path)).items() if v is not None}


def _overlay(file_env: Mapping[str, Optional[str]], environ: Mapping[str, str]) -> Dict[str, str]:
    """The house idiom: env-file values, overridden by the process environment."""
    env = {k: v for k, v in file_env.items() if v is not None}
    env.update({k: v for k, v in environ.items() if v is not None})
    return env


def _parse_bool(raw: Optional[str]) -> Optional[bool]:
    value = (raw or "").strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSEY:
        return False
    return None


def is_strict_mode(
    *,
    environ: Optional[Mapping[str, str]] = None,
    lillisa_env: Optional[Mapping[str, Optional[str]]] = None,
    lillisa_env_path: Optional[Path] = None,
) -> bool:
    """Whether BLOCKING problems should refuse startup.

    Reads TECHSUPPORT_PREFLIGHT_STRICT from env/lillisa_server.env overlaid by
    the process environment (so a dev box can opt out in its gitignored env
    file, and a deployment can override via the container env). Accepts
    true/false/1/0/yes/no/on/off case-insensitively. Unset, blank, or
    unrecognised spellings all resolve to DEFAULT_STRICT (True): a typo must
    not silently disable the guard.
    """
    environ = os.environ if environ is None else environ
    if lillisa_env is None:
        lillisa_env = _read_env_file(lillisa_env_path or LILLISA_SERVER_ENV_PATH)
    parsed = _parse_bool(_overlay(lillisa_env, environ).get(STRICT_ENV_KEY))
    return DEFAULT_STRICT if parsed is None else parsed


def _scan_product_channels(lillisa: Mapping[str, str]) -> bool:
    raw = lillisa.get("TECHSUPPORT_SCAN_PRODUCT_CHANNELS", _DEFAULT_SCAN_PRODUCT_CHANNELS)
    return str(raw).strip().lower() not in _SCAN_FALSEY


def _resolve_path(raw: str, server_root: Path) -> Path:
    """Same rule as techsupport_qa_ingest._resolve_path: relative paths in
    lillisa_server.env are relative to the LilLisa_Server root, not the CWD."""
    path = Path(raw)
    return path if path.is_absolute() else (server_root / path)


def _writability_problem(path: Path) -> Optional[str]:
    """None if `path` is (or could be created as) a writable directory, else a
    short reason. Never creates anything: preflight must be side-effect free.

    A missing directory is fine as long as its nearest existing ancestor is a
    writable directory -- the pipeline mkdirs on first use, and a fresh
    deployment legitimately has no lancedb/ or techsupport_state/ yet.
    """
    if path.exists():
        if not path.is_dir():
            return "exists but is not a directory"
        if not os.access(path, os.W_OK | os.X_OK):
            return "is not writable by this process"
        return None
    ancestor = path.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        return f"cannot be created: nearest existing ancestor {ancestor} is not a directory"
    if not os.access(ancestor, os.W_OK | os.X_OK):
        return f"does not exist and cannot be created: {ancestor} is not writable by this process"
    return None


def _repo_url_problem(repo_url: str) -> Optional[str]:
    """github_sync._require_https_repo_url's rules, reported instead of raised.
    The returned text must never include the URL itself: a URL that fails the
    credentials check is, by definition, carrying a secret."""
    try:
        parts = urlsplit(repo_url)
        has_credentials = bool(parts.username or parts.password)
    except ValueError:
        return "is not a parseable URL"
    if parts.scheme != "https":
        return "must be an https:// URL (git://, ssh:// and scp-style git@host:org/repo URLs cannot use GITHUB_TOKEN)"
    if has_credentials:
        return "must not embed credentials (user:password@ or token@ in the URL); put the token in GITHUB_TOKEN"
    if not parts.hostname:
        return "has no host"
    return None


def _env_file_fix(key: str, env_path: Path) -> str:
    return (
        f"Mount Secret key {env_path.name} at {env_path} (volumeMount with subPath: {env_path.name}) "
        f"with {key} set, or inject {key} into the container environment."
    )


def _missing(severity: Severity, key: str, env_path: Path, why: str, extra_fix: str = "") -> PreflightProblem:
    return PreflightProblem(
        severity=severity,
        keys=(key,),
        problem=f"{key} is not set{why}.",
        remediation=_env_file_fix(key, env_path) + (f" {extra_fix}" if extra_fix else ""),
    )


def run_preflight(
    *,
    environ: Optional[Mapping[str, str]] = None,
    sync_env: Optional[Mapping[str, Optional[str]]] = None,
    github_env: Optional[Mapping[str, Optional[str]]] = None,
    lillisa_env: Optional[Mapping[str, Optional[str]]] = None,
    sync_env_path: Optional[Path] = None,
    github_env_path: Optional[Path] = None,
    lillisa_env_path: Optional[Path] = None,
    server_root: Optional[Path] = None,
) -> PreflightResult:
    """Validate the resolved techsupport pipeline configuration and return
    EVERY problem found (never raises for a configuration problem).

    All arguments are optional injection points so tests need not touch
    global state:
      environ       -- stands in for os.environ (the override layer).
      *_env         -- the parsed contents of each env file; when omitted the
                       file at *_env_path is read (an absent file is empty).
      *_env_path    -- where each env file lives; also what the report tells
                       the operator to mount.
      server_root   -- LilLisa_Server root, for resolving relative paths.
    """
    environ = os.environ if environ is None else environ
    sync_env_path = sync_env_path or SYNC_ENV_PATH
    github_env_path = github_env_path or GITHUB_ENV_PATH
    lillisa_env_path = lillisa_env_path or LILLISA_SERVER_ENV_PATH
    server_root = server_root or LILLISA_SERVER_ROOT

    if sync_env is None:
        # CHANNEL_ID_* live in lil-lisa.env. techsupport_sync.env and the
        # process environment still override them when set.
        sync_file = read_lil_lisa_channel_ids()
        sync_file.update(_read_env_file(sync_env_path))
    else:
        sync_file = {k: v for k, v in sync_env.items() if v is not None}
    sync = _overlay(sync_file, environ)
    github = _overlay(github_env if github_env is not None else _read_env_file(github_env_path), environ)
    lillisa = _overlay(lillisa_env if lillisa_env is not None else _read_env_file(lillisa_env_path), environ)

    problems: List[PreflightProblem] = []
    problems.extend(_check_slack(sync, sync_env_path))
    problems.extend(_check_product_scan(sync, lillisa, sync_env_path, lillisa_env_path))
    problems.extend(_check_server(lillisa, lillisa_env_path))
    problems.extend(_check_github(github, github_env_path))
    problems.extend(_check_writable_dirs(lillisa, lillisa_env_path, server_root))

    sources: List[Path] = [sync_env_path, github_env_path, lillisa_env_path]
    # Only when this call actually read lil-lisa.env. Injected sync_env (tests,
    # and any caller that already resolved the config) does not consult it.
    if sync_env is None and bot_env_path().is_file():
        sources.append(bot_env_path())
    return PreflightResult(
        problems=tuple(problems),
        sources=tuple(sources),
    )


def _check_slack(sync: Mapping[str, str], sync_env_path: Path) -> List[PreflightProblem]:
    """The staging failure itself: without these two keys every sync() call
    raises and the pipeline does nothing at all."""
    problems: List[PreflightProblem] = []
    for key in ("SLACK_BOT_TOKEN", "TECHSUPPORT_CHANNEL_ID"):
        if not _clean(sync.get(key)):
            problems.append(
                _missing(
                    Severity.BLOCKING,
                    key,
                    sync_env_path,
                    " -- nightly_techsupport_sync.load_env() raises on every pipeline tick, so nothing is ingested",
                )
            )

    return problems


def _check_product_scan(
    sync: Mapping[str, str],
    lillisa: Mapping[str, str],
    sync_env_path: Path,
    lillisa_env_path: Path,
) -> List[PreflightProblem]:
    """The product-channel expert-correction pass runs by default. With it on,
    a missing IDA/IDDM expert group makes the expert resolver raise
    (expert_group rule 3), and a missing IDA/IDDM channel silently drops that
    product's corrections -- either way the pass does not do its job. Only
    required while the pass is enabled; IDO stays optional.

    Both keys are DEGRADED, not BLOCKING, even though the pass cannot do its
    job without them. The deciding question for this tier is not "is this
    feature broken?" but "can the pipeline still ingest?", and here it can:
    run_pipeline() runs the techsupport channel loop (sync -> classify ->
    ingest) to completion FIRST, and only then calls
    run_product_channel_pass() inside a try/except that counts the failure,
    posts an admin alert, and returns -- so a missing expert group costs the
    product-correction pass and nothing else. Specifically:
      * EXPERT_GROUP_ID_* makes expert_group.ExpertResolver raise ValueError
        for that product, which run_product_channel_pass() already catches per
        product and reports as errored rather than quietly skipping.
      * CHANNEL_ID_* is an explicitly supported per-product opt-out:
        run_product_channel_pass()'s docstring states that "a product with no
        channel id is not scanned", and product_channel_ids() simply leaves
        it out of the dict. These are the Slack bot's channel variables
        (lil-lisa.env), not a second copy in techsupport_sync.env.

    Keeping these out of the BLOCKING tier matters for blast radius. The scan
    defaults to ON (DEFAULT_SCAN_PRODUCT_CHANNELS = "true"), so treating them
    as BLOCKING would make four extra keys mandatory for every deployment and
    would refuse to boot a pod whose primary job -- ingesting the techsupport
    channel -- is configured perfectly. BLOCKING stays scoped to "cannot
    ingest at all, or would lose/corrupt data".
    """
    if not _scan_product_channels(lillisa):
        return []
    disable_hint = (
        f"Or, to run without the product-channel pass, set TECHSUPPORT_SCAN_PRODUCT_CHANNELS=false in "
        f"{lillisa_env_path} or the container environment."
    )
    problems: List[PreflightProblem] = []
    for template, why, for_channel in (
        (
            "CHANNEL_ID_{product}",
            " but TECHSUPPORT_SCAN_PRODUCT_CHANNELS is enabled, so that product's channel is not scanned "
            "and its expert corrections are never picked up (every other product still scans)",
            True,
        ),
        (
            "EXPERT_GROUP_ID_{product}",
            " but TECHSUPPORT_SCAN_PRODUCT_CHANNELS is enabled, so the expert resolver raises for that "
            "product and its expert corrections are never picked up (techsupport channel ingest is "
            "unaffected)",
            False,
        ),
    ):
        for product in REQUIRED_SCAN_PRODUCTS:
            key = template.format(product=product)
            if _clean(sync.get(key)):
                continue
            if for_channel:
                problems.append(
                    PreflightProblem(
                        severity=Severity.DEGRADED,
                        keys=(key,),
                        problem=f"{key} is not set{why}.",
                        remediation=(
                            f"{key} is set in lil-lisa/app_envfiles/lil-lisa.env for the Slack bot. Inject that "
                            f"same {key} into this container's environment (the API image does not contain "
                            f"lil-lisa.env). Do not keep a second copy under another name. {disable_hint}"
                        ),
                    )
                )
            else:
                problems.append(_missing(Severity.DEGRADED, key, sync_env_path, why, disable_hint))
    return problems


def _check_server(lillisa: Mapping[str, str], lillisa_env_path: Path) -> List[PreflightProblem]:
    """Post-ingest index reload (nightly_pipeline.reload_techsupport_index):
    without AUTHENTICATION_KEY it cannot authenticate, so rows written to
    LanceDB stay invisible to live queries until a restart -- the pipeline
    "works" but its output never reaches users, which is misbehaviour, hence
    BLOCKING. LIL_LISA_SERVER_URL has a correct loopback default for the
    in-process pipeline, so its absence is only DEGRADED."""
    problems: List[PreflightProblem] = []
    if not _clean(lillisa.get("AUTHENTICATION_KEY")):
        problems.append(
            _missing(
                Severity.BLOCKING,
                "AUTHENTICATION_KEY",
                lillisa_env_path,
                " -- the post-ingest index reload cannot authenticate, so newly ingested entries stay "
                "invisible to live queries",
            )
        )
    if not _clean(lillisa.get("LIL_LISA_SERVER_URL")):
        problems.append(
            PreflightProblem(
                severity=Severity.DEGRADED,
                keys=("LIL_LISA_SERVER_URL",),
                problem=(
                    "LIL_LISA_SERVER_URL is not set; the index reload falls back to the loopback default "
                    "http://127.0.0.1:8000 (correct for the in-process pipeline). A present-but-blank value "
                    "falls back the same way (pr42-44j.9)."
                ),
                remediation=(
                    f"Set LIL_LISA_SERVER_URL in {lillisa_env_path} or the container environment if the "
                    "API is not reachable on the loopback default."
                ),
            )
        )
    return problems


def _check_github(github: Mapping[str, str], github_env_path: Path) -> List[PreflightProblem]:
    """GITHUB_TOKEN and GITHUB_REPO_URL are BLOCKING. Without them the verified
    markdown is never pushed and answers lose their clickable source links, so
    a deployment that forgot github_push.env must fail the rollout rather than
    serve. A present but malformed GITHUB_REPO_URL is the same tier: it is an
    operator mistake (possibly a token pasted into the URL)."""
    problems: List[PreflightProblem] = []
    lost = "the GitHub mirror of techsupport_qa_pairs.md and the clickable source links in answers are disabled"
    if not _clean(github.get("GITHUB_TOKEN")):
        problems.append(_missing(Severity.BLOCKING, "GITHUB_TOKEN", github_env_path, f"; {lost}"))
    repo_url = _clean(github.get("GITHUB_REPO_URL"))
    if not repo_url:
        problems.append(_missing(Severity.BLOCKING, "GITHUB_REPO_URL", github_env_path, f"; {lost}"))
    else:
        reason = _repo_url_problem(repo_url)
        if reason:
            problems.append(
                PreflightProblem(
                    severity=Severity.BLOCKING,
                    keys=("GITHUB_REPO_URL",),
                    problem=f"GITHUB_REPO_URL is set but malformed: it {reason}. (Value not shown.)",
                    remediation=(
                        "Set GITHUB_REPO_URL to a plain https://github.com/<org>/<repo>.git URL in "
                        f"{github_env_path} (Secret key {github_env_path.name}, mounted with subPath) or the "
                        "container environment, with the token in GITHUB_TOKEN. If a token was embedded in the "
                        "URL, rotate it."
                    ),
                )
            )
    return problems


def _check_writable_dirs(
    lillisa: Mapping[str, str],
    lillisa_env_path: Path,
    server_root: Path,
) -> List[PreflightProblem]:
    """Every directory the pipeline writes into. An unwritable one fails the
    run part-way (after Slack and the LLM have already been paid for), so it is
    BLOCKING. Read-only root filesystems with no volume mounted are the usual
    cause in Kubernetes."""
    problems: List[PreflightProblem] = []

    raw_lancedb = _clean(lillisa.get("LANCEDB_FOLDERPATH"))
    if not raw_lancedb:
        # load_pipeline_env() raises without it, so every ingest would fail.
        problems.append(
            _missing(
                Severity.BLOCKING,
                "LANCEDB_FOLDERPATH",
                lillisa_env_path,
                " -- techsupport_qa_ingest cannot open the vector store",
            )
        )
    targets: List[Tuple[Tuple[str, ...], Path, str]] = []
    if raw_lancedb:
        targets.append(
            (
                ("LANCEDB_FOLDERPATH",),
                _resolve_path(raw_lancedb, server_root),
                f"mount a writable volume there, or point LANCEDB_FOLDERPATH (in {lillisa_env_path} or the "
                "container environment) at a writable directory",
            )
        )
    # Resolved by the same functions the pipeline and the API get their paths
    # from (src/techsupport_state_paths.py, via paths.py), so preflight always
    # checks the folders that are actually written. That module is stdlib +
    # dotenv only, so this module stays self-contained.
    targets.append(
        (
            ("VERIFIED_TECHSUPPORT_QA_FOLDERPATH",),
            resolve_verified_qa_dir(lillisa, server_root),
            "mount a writable volume there, or set VERIFIED_TECHSUPPORT_QA_FOLDERPATH (in "
            f"{lillisa_env_path} or the container environment) to a writable directory",
        )
    )
    # The pipeline's own state files (techsupport_sync_state.json's dedup
    # flags, the reembed timestamp, review_state's node_id mapping) and the
    # API's two tag files. An unwritable folder means atomic_io cannot even
    # write its temp file, so every run would fail after doing real work.
    targets.append(
        (
            ("TECHSUPPORT_STATE_DIR",),
            resolve_state_dir(lillisa, server_root),
            "the techsupport state files (techsupport_sync_state.json, techsupport_reembed_state.json, "
            "techsupport_review_state.json, techsupport_thread_tags.json, techsupport_answer_tags.json) are "
            "written here, next to LANCEDB_FOLDERPATH by default; make it writable by the server process, or "
            f"set TECHSUPPORT_STATE_DIR (in {lillisa_env_path} or the container environment) to a writable "
            "directory on persistent storage",
        )
    )

    for keys, path, fix in targets:
        reason = _writability_problem(path)
        if reason:
            label = keys[0]
            problems.append(
                PreflightProblem(
                    severity=Severity.BLOCKING,
                    keys=keys,
                    path=str(path),
                    problem=f"{label} {path} {reason}.",
                    remediation=fix[0].upper() + fix[1:] + ".",
                )
            )
    return problems


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def format_preflight_report(result: PreflightResult, *, strict: bool, startup: bool = True) -> str:
    """Render `result` as one multi-line, operator-facing report.

    Built only from PreflightProblem fields and source paths, which by
    construction contain key names and paths, never values -- so it is safe to
    log at CRITICAL and ship to any log aggregator.

    `startup` picks the closing verdict's framing. True (the default) is the
    boot-time caller (techsupport_cron.run_startup_preflight), where the
    outcome is whether the process starts at all, and `strict` decides which
    of those two verdicts applies. False is the runtime net
    (techsupport_cron._log_run_failure), where the pod is already serving and
    the only honest verdict is "this tick failed and will keep failing";
    `strict` is then irrelevant to the closing line.
    """
    if not result.problems:
        return "techsupport pipeline preflight passed: no configuration problems found."

    blocking = result.blocking
    code = FAILURE_CODE if blocking else DEGRADED_CODE
    verdict = "failed" if blocking else "found degraded configuration"
    lines: List[str] = [
        f"{code}: techsupport pipeline preflight {verdict} "
        f"({_plural(len(result.problems), 'problem')}: {len(blocking)} blocking, "
        f"{len(result.degraded)} degraded).",
    ]
    # BLOCKING first: they are what the operator has to fix before anything else.
    ordered: Sequence[PreflightProblem] = blocking + result.degraded
    for index, problem in enumerate(ordered, start=1):
        lines.append(f"  {index}. [{problem.severity.value}] {problem.problem}")
        lines.append(f"     Fix: {problem.remediation}")

    lines.append("")
    lines.append("Configuration sources (both routes are supported):")
    for source in result.sources:
        lines.append(f"  - {source}")
    lines.append(
        "  Values in these files are overridden by container environment variables of the same name. "
        "A key injected only as an environment variable is valid even if the file is absent."
    )
    lines.append(
        "Kubernetes: mount each Secret key at the exact path above using a volumeMount with subPath "
        "(e.g. mountPath: <path above>, subPath: techsupport_sync.env), or inject the keys via env/envFrom. "
        "The image ships no cron/env/*.env files."
    )
    lines.append("")
    if not startup:
        # Runtime net (techsupport_cron._log_run_failure): this pod is already
        # serving, so none of the startup verdicts below apply -- saying
        # "Refusing to start" in an hourly tick log would be actively
        # misleading about what just happened.
        lines.append(
            f"This is a running pod: the techsupport pipeline tick just failed and will keep failing every "
            f"tick until the above is fixed. A configuration change needs a pod restart to take effect when "
            f"the env files are mounted from a Secret with subPath."
        )
    elif blocking and strict:
        lines.append(
            f"Refusing to start: {_plural(len(blocking), 'BLOCKING problem')} would leave the techsupport "
            f"pipeline broken. Fix the above, or set {STRICT_ENV_KEY}=false to start anyway "
            "(dev only -- the pipeline will stay broken)."
        )
    elif blocking:
        lines.append(
            f"Starting anyway because {STRICT_ENV_KEY}=false: the techsupport pipeline will stay broken "
            "until the BLOCKING problems above are fixed."
        )
    else:
        lines.append(
            "Starting normally: only DEGRADED problems. Ingestion and retrieval work; the affected features "
            "stay disabled until fixed."
        )
    return "\n".join(lines)
