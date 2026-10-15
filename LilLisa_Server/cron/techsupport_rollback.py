"""
techsupport_rollback.py
====================================
Manual rollback tool for the TECHSUPPORT_QA_PAIRS LanceDB table, built on
LanceDB's native version history rather than any custom backup mechanism.

techsupport_contextual_reembed.py's rebuild_table() writes each re-embed into
the live table via create_table(mode="overwrite") instead of drop_table() +
rename, which means LanceDB keeps every prior version of the table on disk
(list_versions()) and lets you jump back to one (checkout() / restore()).
This script is a thin, explicit wrapper around that so someone can inspect
and roll back the table by hand without knowing the LanceDB API.

Coordinated restore (do not skip):
    Rolling back TECHSUPPORT_QA_PAIRS alone will drift from the verified
    markdown source and from techsupport_review_state.json (node_ids /
    thread_ts keyed by markdown entry index). Those three artifacts must
    be restored together from the same point in time:
        1. LanceDB table TECHSUPPORT_QA_PAIRS (this script)
        2. techsupport_qa_pairs.md under VERIFIED_TECHSUPPORT_QA_FOLDERPATH
           (by default techsupport_state/verified_techsupport/ next to the LanceDB folder)
        3. techsupport_review_state.json in the techsupport state folder
    This script does NOT auto-restore markdown or review_state from Lance
    versions. After rollback_to_version(), it prints a reminder listing
    the files you must restore yourself (git history for the markdown;
    backup/copy for review_state).

No index management is performed here: TECHSUPPORT_QA_PAIRS has no persistent
vector or scalar index today (confirmed via list_indices() -- the only index
in play is a full-text index LanceDBVectorStore.query() builds lazily, in the
querying process, on the first hybrid query after startup, with replace=True,
so it self-heals regardless of what this script does). Nothing here needs to
rebuild an index that no ingest/rebuild script maintains in the first place.

Required env vars (read from ../env/lillisa_server.env, same file the main
server and techsupport_contextual_reembed.py read):
    LANCEDB_FOLDERPATH - path (relative to LilLisa_Server project root) to the LanceDB folder

Usage (as a library):
    from techsupport_rollback import list_available_versions, rollback_to_version
    list_available_versions()
    rollback_to_version(3)

Usage (standalone):
    python techsupport_rollback.py list
    python techsupport_rollback.py rollback <version_number>
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import dotenv_values
from paths import (
    LILLISA_SERVER_ENV_PATH,
    LILLISA_SERVER_ROOT,
    PACKAGE_ROOT,
    REVIEW_STATE_PATH,
    VERIFIED_TECHSUPPORT_QA_FOLDERPATH,
    ensure_import_paths,
)

import lancedb

ensure_import_paths()
SCRIPT_DIR = PACKAGE_ROOT
PROJECT_ROOT = LILLISA_SERVER_ROOT

TECHSUPPORT_QA_TABLE_NAME = "TECHSUPPORT_QA_PAIRS"


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _open_table():
    # os.environ overlay so rollback still works on a deployment that injects
    # config as env vars instead of mounting env/ (see build/dockerfile_prod).
    env = {**dict(dotenv_values(str(LILLISA_SERVER_ENV_PATH))), **os.environ}
    if not env.get("LANCEDB_FOLDERPATH"):
        raise RuntimeError(f"LANCEDB_FOLDERPATH not found in {LILLISA_SERVER_ENV_PATH} or the process environment")
    lancedb_folderpath = str(_resolve_path(env["LANCEDB_FOLDERPATH"]))
    db = lancedb.connect(lancedb_folderpath)
    if TECHSUPPORT_QA_TABLE_NAME not in db.table_names():
        raise RuntimeError(f"Table '{TECHSUPPORT_QA_TABLE_NAME}' does not exist in {lancedb_folderpath}")
    return db.open_table(TECHSUPPORT_QA_TABLE_NAME)


def _print_coordinated_restore_reminder(restored_from_version: int) -> None:
    """Warn that markdown + review_state are not restored by this script."""
    print()
    print("=" * 72)
    print("IMPORTANT: LanceDB rollback is incomplete until you also restore")
    print("markdown + review_state from the SAME point in time as LanceDB")
    print(f"version {restored_from_version}. This script does NOT auto-restore them.")
    print()
    print("Restore together:")
    print(f"  1. LanceDB table {TECHSUPPORT_QA_TABLE_NAME}  (just done)")
    print("  2. techsupport_qa_pairs.md")
    print(f"     ({VERIFIED_TECHSUPPORT_QA_FOLDERPATH / 'techsupport_qa_pairs.md'};")
    print("      GitHub history in Section 7 of the deploy notes)")
    print(f"  3. {REVIEW_STATE_PATH}")
    print("=" * 72)


def list_available_versions() -> List[Dict[str, Any]]:
    """Print every version of TECHSUPPORT_QA_PAIRS currently on disk, newest
    last, so someone deciding whether/what to roll back to can see what's
    available. Returns the same list list_versions() gives, for callers that
    want it programmatically."""
    table = _open_table()
    versions = table.list_versions()
    current = table.version

    print(f"Table: {TECHSUPPORT_QA_TABLE_NAME}")
    print(f"Current version: {current}")
    print(f"{len(versions)} version(s) available:\n")
    for v in versions:
        marker = "  <- current" if v["version"] == current else ""
        print(f"  version {v['version']:>4}  {v['timestamp']}{marker}")

    return versions


def rollback_to_version(version_number: int) -> Dict[str, Any]:
    """Roll TECHSUPPORT_QA_PAIRS back to `version_number` via restore().

    restore() does not overwrite/delete the version it restores from, or any
    version in between -- it adds a new HEAD version whose data matches
    `version_number` (confirmed empirically), so this operation is itself
    reversible: list_available_versions() will show the pre-rollback state as
    a still-checkoutable earlier version, and rolling back again (to a
    version before *this* rollback) undoes it.

    After restore, prints a reminder that techsupport_qa_pairs.md and
    techsupport_review_state.json must be restored together with this table.
    This function does not auto-restore those files.
    """
    table = _open_table()
    available = {v["version"] for v in table.list_versions()}
    if version_number not in available:
        raise ValueError(
            f"Version {version_number} does not exist for {TECHSUPPORT_QA_TABLE_NAME}. "
            f"Available versions: {sorted(available)}"
        )

    before_version = table.version
    before_rows = table.count_rows()

    table.restore(version_number)

    after_version = table.version
    after_rows = table.count_rows()

    print(f"Rolled back {TECHSUPPORT_QA_TABLE_NAME}:")
    print(f"  was version {before_version} ({before_rows} rows)")
    print(f"  restored data from version {version_number}")
    print(f"  now version {after_version} ({after_rows} rows) <- new HEAD")
    _print_coordinated_restore_reminder(version_number)

    return {
        "table_name": TECHSUPPORT_QA_TABLE_NAME,
        "restored_from_version": version_number,
        "before_version": before_version,
        "after_version": after_version,
        "row_count": after_rows,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "rollback"):
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "list":
        list_available_versions()
    else:
        if len(sys.argv) != 3:
            print("Usage: python techsupport_rollback.py rollback <version_number>")
            sys.exit(1)
        rollback_to_version(int(sys.argv[2]))
