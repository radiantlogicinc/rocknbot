"""
techsupport_bootstrap.py
====================================
Brings the verified-techsupport store in sync with GITHUB_REPO_URL at server
startup: clone the repo, take its techsupport_qa_pairs.md, and rebuild the
TECHSUPPORT_QA_PAIRS LanceDB table from it.

GITHUB_REPO_URL is the source of truth for this content, the same contract
src/main.py's _run_update_golden_qa_pairs_task() applies to the golden QA
pairs repo (QA_PAIRS_GITHUB_REPO_URL): that repo is cloned on startup and
each product's table is rebuilt from the markdown in it. This module is the
techsupport equivalent, called from src/techsupport_cron.run_startup_bootstrap()
in main.lifespan. The local markdown under VERIFIED_TECHSUPPORT_QA_FOLDERPATH
is a working copy the nightly pipeline appends to between pushes, not the
authority.

Why this needs its own startup step rather than riding on
init_lance_databases(): TECHSUPPORT_QA_PAIRS is deliberately NOT in
src/main.py's _REQUIRED_LANCEDB_TABLES, because a missing techsupport table
must not trigger a full documentation + golden-QA rebuild. That means a pod
whose docs and golden tables are intact takes main.lifespan's fast path and
would never build TECHSUPPORT_QA_PAIRS at all. So this runs on BOTH paths
and self-gates instead (step 2 below).

What run_bootstrap() does, each step idempotent -- safe on every container
start, not just the first:

  1. markdown_sync: reconcile techsupport_qa_pairs.md between the local copy
     and GITHUB_REPO_URL (github_sync.sync_verified_qa_pairs()). Whichever
     side has more entries wins: repo ahead -> pull it down; local ahead ->
     push the unpushed local entries up. See that function's docstring.
  2. lancedb_rebuild: compare TECHSUPPORT_QA_PAIRS's row count to the
     markdown's entry count and, when they differ, run ONE full contextual
     re-embed (techsupport_contextual_reembed.run_if_due(force=True)) to
     rebuild the table from the markdown. force=True is the sanctioned
     exception to that module's "a missing state file means do not rebuild"
     rule (see is_reembed_due()): here the markdown has just been
     reconciled against the source of truth, which is precisely the
     known-good reason to rebuild from it. Equal counts are a no-op, so a
     normal restart costs one clone and one row-count query.

techsupport_sync_state.json is deliberately NOT reconstructed here. The
markdown records no thread_ts, so there is no way to rebuild the per-thread
sync history from it; only techsupport_review_state.json ever held that, and
this scenario does not assume it survived. nightly_techsupport_sync.sync()
already starts a correctly-shaped v2 state against a missing file, and the
duplicate-add risk that causes is covered by
techsupport_markdown.find_duplicate_entry_index(), the content-based dedup
guard techsupport_qa_ingest.add_verified_qa_pair() applies before appending.

Required env vars: whatever sync_verified_qa_pairs() (cron/env/github_push.env:
GITHUB_TOKEN, GITHUB_REPO_URL) and techsupport_contextual_reembed.run_reembed()
(env/lillisa_server.env: LANCEDB_FOLDERPATH, VOYAGE_API_KEY_FILEPATH) already
require. Both are BLOCKING in techsupport_preflight.py, which main.lifespan
runs before this, so a misconfigured deployment fails there with a full report
rather than here.

Usage (as a library, from src/techsupport_cron.py at server startup):
    from techsupport_bootstrap import run_bootstrap
    result = run_bootstrap()

Usage (standalone):
    python techsupport_bootstrap.py
"""

import logging
from typing import Any, Dict

import lancedb
from paths import ensure_import_paths

ensure_import_paths()

from github_sync import sync_verified_qa_pairs  # noqa: E402

# Reused from techsupport_contextual_reembed.py rather than
# techsupport_qa_ingest.py -- same reasoning backfill_github_urls.py already
# documents for itself: avoids triggering techsupport_qa_ingest's
# module-level configure_dspy_lm(), an LLM dependency this script (which
# only counts entries and triggers a rebuild; it extracts nothing new) has
# no need for.
from techsupport_contextual_reembed import (  # noqa: E402
    TECHSUPPORT_QA_MARKDOWN_FILENAME,
    TECHSUPPORT_QA_TABLE_NAME,
    VERIFIED_TECHSUPPORT_QA_FOLDERPATH,
    _resolve_path,
    load_pipeline_env,
    parse_qa_pairs,
    run_if_due as run_reembed_if_due,
)

logger = logging.getLogger(__name__)

MARKDOWN_PATH = VERIFIED_TECHSUPPORT_QA_FOLDERPATH / TECHSUPPORT_QA_MARKDOWN_FILENAME


def _markdown_entry_count() -> int:
    if not MARKDOWN_PATH.exists():
        return 0
    return len(parse_qa_pairs(MARKDOWN_PATH.read_text(encoding="utf-8")))


def _lancedb_row_count() -> int:
    """0 if LANCEDB_FOLDERPATH/VOYAGE_API_KEY_FILEPATH aren't configured yet,
    or the table doesn't exist -- both are ordinary states for a genuinely
    fresh environment, not errors this function should raise on."""
    try:
        env = load_pipeline_env()
    except RuntimeError:
        return 0
    lancedb_folderpath = str(_resolve_path(env["LANCEDB_FOLDERPATH"]))
    db = lancedb.connect(lancedb_folderpath)
    if TECHSUPPORT_QA_TABLE_NAME not in db.table_names():
        return 0
    return db.open_table(TECHSUPPORT_QA_TABLE_NAME).count_rows()


def run_bootstrap() -> Dict[str, Any]:
    """Reconcile the local markdown with GITHUB_REPO_URL in whichever
    direction has more entries, then rebuild TECHSUPPORT_QA_PAIRS from it
    when the table doesn't already have one row per entry. See module
    docstring for the full rationale.

    Never raises: each step's exception is captured as an {"error": ...}
    entry in the returned dict so the caller gets one complete report.
    src/techsupport_cron.run_startup_bootstrap() turns any such entry into a
    startup abort. A failed sync skips the rebuild, since startup is about to
    abort anyway and rebuilding from an unreconciled file is wasted work."""
    result: Dict[str, Any] = {}

    try:
        result["markdown_sync"] = sync_verified_qa_pairs()
    except Exception as exc:  # noqa: BLE001 -- reported via the result; the caller aborts startup
        result["markdown_sync"] = {"direction": "none", "error": str(exc)}
        logger.exception("techsupport_bootstrap: syncing %s with the repo failed", TECHSUPPORT_QA_MARKDOWN_FILENAME)
        result["lancedb_rebuild"] = {"ran": False, "reason": "markdown_sync_failed"}
        return result

    entry_count = _markdown_entry_count()
    if entry_count == 0:
        result["lancedb_rebuild"] = {"ran": False, "reason": "no_markdown_entries_yet"}
        return result

    try:
        row_count = _lancedb_row_count()
    except Exception as exc:  # noqa: BLE001 -- reported via the result; the caller aborts startup
        result["lancedb_rebuild"] = {"ran": False, "error": str(exc)}
        logger.exception("techsupport_bootstrap: checking %s row count failed", TECHSUPPORT_QA_TABLE_NAME)
        return result

    if row_count == entry_count:
        result["lancedb_rebuild"] = {
            "ran": False,
            "reason": "already_in_sync",
            "entry_count": entry_count,
            "row_count": row_count,
        }
        return result

    logger.warning(
        "techsupport_bootstrap: %s has %d row(s) but %s has %d entries -- running a full "
        "contextual re-embed to bring LanceDB in sync",
        TECHSUPPORT_QA_TABLE_NAME,
        row_count,
        TECHSUPPORT_QA_MARKDOWN_FILENAME,
        entry_count,
    )
    try:
        result["lancedb_rebuild"] = run_reembed_if_due(force=True)
    except Exception as exc:  # noqa: BLE001 -- reported via the result; the caller aborts startup
        result["lancedb_rebuild"] = {"ran": False, "error": str(exc)}
        logger.exception("techsupport_bootstrap: forced re-embed failed")
    return result


if __name__ == "__main__":
    print(run_bootstrap())
