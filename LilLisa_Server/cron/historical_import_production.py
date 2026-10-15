"""
historical_import_production.py
====================================
One-time bulk import of data/historical_import/production_1year.txt (a real
Slack export already threaded with slack_harvester.py's exact reply format --
see the file structure check that preceded this script) into the SAME shared
verified-techsupport store (VERIFIED_TECHSUPPORT_QA_FOLDERPATH/techsupport_qa_pairs.md
+ the TECHSUPPORT_QA_PAIRS LanceDB table) that nightly_pipeline.py maintains
for live Slack threads.

Unlike historical_import.py (which converts last year's already-summarized,
already-filtered "useful and conclusive" text into this format), this script
starts from RAW threaded conversations, so it reuses nightly_pipeline.py's
actual live-thread pipeline end to end, per conversation:
    1. techsupport_classifier.classify_thread() -- IsUsefulConversation /
       IsConclusiveConversation, gated the same way (conclusive is only
       checked if useful).
    2. techsupport_qa_ingest.add_verified_qa_pair() -- SummarizeConversationThread
       + GenerateTechsupportTitle, then append to techsupport_qa_pairs.md and
       insert into TECHSUPPORT_QA_PAIRS, exactly like nightly_pipeline.py's
       process_thread() does for a real Slack thread.
There is no conversation-grouping/extraction step to port from historical_import.py
(or the older extracting-conversations project) -- the input already has real
thread structure, one top-level message plus its indented "↳" replies.

Parsing: production_1year.txt has 179 lines that look like a bracketed
timestamp at the start of a line, but only 171 of those are real top-level
messages ("[YYYY-MM-DD HH:MM:SS] USERID: text"); the other 8 are pasted
service-log lines (e.g. "[2025-11-26 10:19:21] [info]  [11244] Apache Commons
Daemon...") embedded inside a message's own multi-line text that happen to
start with a bracket + timestamp but have no "USERID: " after it. parse_conversations()
below requires the full "[timestamp] USERID: " prefix (matching
slack_harvester.py's save_messages_to_file() output exactly -- see its
f"\\n[{timestamp}] {user}: {text}\\n" / f"    ↳ [{reply_timestamp}] {reply_user}: {reply_text}\\n"
lines) for both top-level and "    ↳ "-prefixed reply lines, so those 8 lines
are correctly folded into their parent message's text as continuation
content instead of being split into bogus extra conversations. This gives
exactly EXPECTED_CONVERSATION_COUNT = 171 conversations; run() refuses to
proceed if a re-parse of the file ever produces a different count (the file
may have changed since this script was written).

Each conversation's messages are converted into the same Slack-message-JSON
shape (list of {"ts", "user", "text"} dicts) that classify_thread() /
format_thread_messages() already consume for live threads -- "ts" is a
synthesized unix-epoch string (parsed from the human timestamp), which is all
format_thread_messages() actually needs (it sorts by float(ts) and prints
"[{ts}] {user}: {text}"); this is a closer match to a real Slack thread's
actual "ts" field than the human "YYYY-MM-DD HH:MM:SS" string would be, and
SummarizeConversationThread's prompt already explicitly asks the model not to
reference timestamps at all, so the exact ts format has no bearing on output
quality. Best-effort Slack username resolution (slack_token from
nightly_techsupport_sync.load_env(), same SLACK_BOT_TOKEN nightly_pipeline.py
uses) is attempted the same way process_thread() does; if that env isn't
configured, classification/summarization proceeds fine with raw user IDs
instead (format_thread_messages()'s own fallback).

Resumable / idempotent: every parsed conversation gets a stable id
"production_1year.txt::<conversation_number>" (1-indexed, file order), and
its outcome is recorded in historical_import_production_state.json (this
directory). "added", "skipped_not_useful", and "skipped_not_conclusive" are
all terminal -- unlike nightly_pipeline.py's live threads, this text is
static and will never receive "new replies" that would justify
re-classifying it, so a skip here is permanent, not just "not yet". "error"
is not terminal, so a transient failure (e.g. a rate limit) gets retried on
the next run without needing --limit gymnastics. Nothing is written to
techsupport_qa_pairs.md, TECHSUPPORT_QA_PAIRS, or the state file for a given
conversation until add_verified_qa_pair() has fully returned (title AND
summary generated, markdown appended, LanceDB row inserted) -- matching the
same small-crash-window tradeoff historical_import.py and
techsupport_qa_ingest.add_verified_qa_pair() already accept (see their
docstrings): a kill between the markdown-append and the state-save could
duplicate at most one entry on retry, not more.

Usage:
    # Dry run: parse + classify + generate title/summary only, print results,
    # touch nothing on disk (no markdown/LanceDB/state writes). Use this for
    # the initial 5-10 conversation sanity check.
    python historical_import_production.py --dry-run --limit 8

    # Real run against everything not yet processed, resumable, skipping the
    # final reembed (recommended while still testing on a small --limit):
    python historical_import_production.py --limit 8 --skip-reembed

    # Full real run, resumable, with a final full contextual reembed of
    # TECHSUPPORT_QA_PAIRS once all conversations are processed:
    python historical_import_production.py

    # Real run without the final reembed (e.g. to split "import" and
    # "reembed" into separate invocations):
    python historical_import_production.py --skip-reembed
    python historical_import_production.py --reembed-only
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from paths import LILLISA_SERVER_ROOT, PACKAGE_ROOT, ensure_import_paths

ensure_import_paths()
SCRIPT_DIR = PACKAGE_ROOT
PROJECT_ROOT = LILLISA_SERVER_ROOT
PRODUCTION_FILE = PROJECT_ROOT / "data" / "historical_import" / "production_1year.txt"
STATE_PATH = SCRIPT_DIR / "historical_import_production_state.json"
SOURCE_ID = "production_1year.txt"
EXPECTED_CONVERSATION_COUNT = 171

import techsupport_contextual_reembed  # noqa: E402
from nightly_techsupport_sync import load_env  # noqa: E402
from techsupport_classifier import classify_thread  # noqa: E402
from techsupport_qa_ingest import (  # noqa: E402
    add_verified_qa_pair,
    generate_verified_title_and_summary,
)

# --- Parsing: production_1year.txt -> flat, ordered list of conversations,
# each one top-level message plus its "    ↳ "-prefixed replies. See module
# docstring for why the "[timestamp] USERID: " prefix must be matched in
# full (not just a leading bracketed timestamp). ---

_TOP_PATTERN = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (\S+): (.*)$")
_REPLY_PATTERN = re.compile(r"^    ↳ \[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (\S+): (.*)$")


def parse_conversations(path: Path) -> List[Dict[str, Any]]:
    """Parse the whole file into an ordered list of conversations, each
    {"conversation_number", "id", "top": {"ts_str","user","text"}, "replies": [...]}.

    A line is only ever treated as starting a new message if it matches the
    full "[timestamp] USERID: " (or "    ↳ [timestamp] USERID: ") prefix;
    every other line -- including blank lines, which can occur INSIDE a
    single multi-paragraph message -- is appended as continuation text to
    whichever message is currently being accumulated.
    """
    lines = path.read_text(encoding="utf-8").split("\n")

    conversations: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None  # buffer for the message currently being accumulated

    def flush(buf: Optional[Dict[str, Any]]) -> None:
        if buf is None:
            return
        text = "\n".join(buf["lines"]).strip()
        message = {"ts_str": buf["ts_str"], "user": buf["user"], "text": text}
        if buf["is_reply"]:
            buf["conv"]["replies"].append(message)
        else:
            buf["conv"]["top"] = message

    for line in lines:
        top_match = _TOP_PATTERN.match(line)
        if top_match:
            flush(current)
            conv: Dict[str, Any] = {"top": None, "replies": []}
            conversations.append(conv)
            current = {
                "ts_str": top_match[1],
                "user": top_match[2],
                "lines": [top_match[3]],
                "is_reply": False,
                "conv": conv,
            }
            continue

        reply_match = _REPLY_PATTERN.match(line)
        if reply_match:
            flush(current)
            if not conversations:
                raise RuntimeError("Found a '↳' reply line before any top-level message -- malformed input")
            current = {
                "ts_str": reply_match[1],
                "user": reply_match[2],
                "lines": [reply_match[3]],
                "is_reply": True,
                "conv": conversations[-1],
            }
            continue

        if current is not None:
            current["lines"].append(line)

    flush(current)

    for i, conv in enumerate(conversations, 1):
        conv["conversation_number"] = i
        conv["id"] = f"{SOURCE_ID}::{i}"
        if conv["top"] is None:
            raise RuntimeError(f"Conversation {conv['id']} has no top-level message -- malformed input")

    return conversations


def _to_epoch_str(ts_str: str) -> str:
    return str(datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").timestamp())


def to_slack_messages(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One conversation -> the same Slack-message-JSON shape (list of
    {"ts", "user", "text"}) classify_thread()/format_thread_messages()
    already consume for live conversations.replies() output."""
    messages = [{"ts": _to_epoch_str(conv["top"]["ts_str"]), "user": conv["top"]["user"], "text": conv["top"]["text"]}]
    for reply in conv["replies"]:
        messages.append({"ts": _to_epoch_str(reply["ts_str"]), "user": reply["user"], "text": reply["text"]})
    return messages


# --- Per-conversation processing: classify_thread() + add_verified_qa_pair(),
# the exact same functions nightly_pipeline.py's process_thread() calls for a
# live Slack thread. ---


def process_conversation(conv: Dict[str, Any], slack_token: Optional[str], dry_run: bool) -> Dict[str, Any]:
    """Returns {"outcome": "added", "title", "summary"} or
    {"outcome": "skipped_not_useful"} or {"outcome": "skipped_not_conclusive"}.
    Raises on LLM/parsing failure -- the caller records that as a retryable
    "error" outcome.

    In dry-run mode, an "added" outcome still runs the same public
    generate_verified_title_and_summary() path add_verified_qa_pair() uses
    (summarize, title, PII redaction) so the preview matches persisted
    content -- it just skips the markdown/LanceDB writes.
    """
    messages = to_slack_messages(conv)
    result = classify_thread(messages, slack_token=slack_token)

    if not result["is_useful"]:
        return {"outcome": "skipped_not_useful"}
    if not result["is_conclusive"]:
        return {"outcome": "skipped_not_conclusive"}

    if dry_run:
        generated = generate_verified_title_and_summary(result["conversation_thread"])
        return {"outcome": "added", "title": generated["title"], "summary": generated["summary"]}

    add_result = add_verified_qa_pair(result["conversation_thread"], thread_ts=f"historical::{conv['id']}")
    return {"outcome": "added", "title": add_result["title"], "summary": add_result["summary"]}


# --- State tracking (mirrors historical_import.py's pattern) ---


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"entries": {}}
    with open(STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


_TERMINAL_OUTCOMES = {"added", "skipped_not_useful", "skipped_not_conclusive"}


def is_done(state: Dict[str, Any], entry_id: str) -> bool:
    record = state["entries"].get(entry_id)
    return bool(record) and record.get("outcome") in _TERMINAL_OUTCOMES


# --- Main run ---


def _load_slack_token() -> Optional[str]:
    """Best-effort: username resolution is a readability nice-to-have, not
    required for classification/summarization to work (same fallback
    process_thread() relies on via format_thread_messages())."""
    try:
        return load_env()["SLACK_BOT_TOKEN"]
    except Exception:  # noqa: BLE001 -- proceed with raw user IDs if unavailable
        return None


def run(limit: Optional[int] = None, dry_run: bool = False, skip_reembed: bool = False) -> Dict[str, Any]:
    all_conversations = parse_conversations(PRODUCTION_FILE)
    if len(all_conversations) != EXPECTED_CONVERSATION_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_CONVERSATION_COUNT} conversations, found "
            f"{len(all_conversations)} -- refusing to proceed ({PRODUCTION_FILE} may have "
            "changed since this script was written)."
        )

    slack_token = _load_slack_token()
    state = load_state()

    pending = [c for c in all_conversations if not is_done(state, c["id"])]
    already_done = len(all_conversations) - len(pending)
    if limit is not None:
        pending = pending[:limit]

    counts = {"added": 0, "skipped_not_useful": 0, "skipped_not_conclusive": 0, "error": 0}
    examples: Dict[str, List[Dict[str, Any]]] = {"added": [], "skipped_not_useful": [], "skipped_not_conclusive": []}
    errors: List[Dict[str, str]] = []

    for i, conv in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {conv['id']} ...", end=" ", flush=True)
        try:
            result = process_conversation(conv, slack_token=slack_token, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001 -- one bad conversation must not abort the batch
            print(f"ERROR: {exc}")
            counts["error"] += 1
            errors.append({"id": conv["id"], "error": str(exc)})
            if not dry_run:
                state["entries"][conv["id"]] = {"outcome": "error", "error": str(exc)}
                save_state(state)
            continue

        outcome = result["outcome"]
        counts[outcome] += 1
        print(outcome)

        if len(examples[outcome]) < 3:
            examples[outcome].append(
                {"id": conv["id"], "top_text": conv["top"]["text"], "num_replies": len(conv["replies"]), **result}
            )

        if not dry_run:
            entry: Dict[str, Any] = {"outcome": outcome}
            if outcome == "added":
                entry["title"] = result["title"]
                entry["summary"] = result["summary"]
            state["entries"][conv["id"]] = entry
            save_state(state)

    reembed_result = None
    if not dry_run and not skip_reembed and counts["added"] > 0:
        print("\nTriggering full contextual re-embed of TECHSUPPORT_QA_PAIRS ...")
        reembed_result = techsupport_contextual_reembed.run_reembed()
        reembed_state = techsupport_contextual_reembed.load_state()
        reembed_state["last_reembed_timestamp"] = f"{time.time():.6f}"
        techsupport_contextual_reembed.save_state(reembed_state)
        print(f"Reembed done: {reembed_result}")

    return {
        "total_conversations": len(all_conversations),
        "already_done_before_this_run": already_done,
        "processed_this_run": len(pending),
        "counts": counts,
        "errors": errors,
        "examples": examples,
        "reembed_result": reembed_result,
        "dry_run": dry_run,
    }


def print_summary(result: Dict[str, Any]) -> None:
    print("\n=== Historical Production Import Summary ===")
    print(f"Total conversations parsed: {result['total_conversations']}")
    print(f"Already done (from previous runs, not reprocessed): {result['already_done_before_this_run']}")
    print(f"Processed this run: {result['processed_this_run']}")
    print(f"  Added: {result['counts']['added']}")
    print(f"  Skipped (not useful): {result['counts']['skipped_not_useful']}")
    print(f"  Skipped (not conclusive): {result['counts']['skipped_not_conclusive']}")
    print(f"  Errors: {result['counts']['error']}")
    if result["errors"]:
        print("  Error detail:")
        for err in result["errors"]:
            print(f"    {err['id']}: {err['error']}")
    if result["dry_run"]:
        print("(dry run -- nothing was written to markdown, state, or LanceDB)")
    elif result["reembed_result"] is not None:
        print(f"Reembed: {result['reembed_result']}")
    elif result["counts"]["added"] == 0:
        print("Reembed: skipped (no new additions this run)")
    else:
        print("Reembed: skipped (--skip-reembed)")

    for label in ("added", "skipped_not_useful", "skipped_not_conclusive"):
        if result["examples"][label]:
            print(f"\n--- Example {label} conversations ---")
            for ex in result["examples"][label]:
                print(f"\n[{ex['id']}] ({ex['num_replies']} replies)")
                print(f"Opening message: {ex['top_text'][:300]}")
                if label == "added":
                    print(f"Title: {ex['title']}")
                    print(f"Summary: {ex['summary']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of not-yet-done conversations to process this run"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Classify and generate title/summary only; write nothing to disk"
    )
    parser.add_argument("--skip-reembed", action="store_true", help="Skip the final full contextual re-embed step")
    parser.add_argument(
        "--reembed-only", action="store_true", help="Skip parsing/import; just force a full re-embed now"
    )
    args = parser.parse_args()

    if args.reembed_only:
        print("Triggering full contextual re-embed of TECHSUPPORT_QA_PAIRS ...")
        reembed_result = techsupport_contextual_reembed.run_reembed()
        reembed_state = techsupport_contextual_reembed.load_state()
        reembed_state["last_reembed_timestamp"] = f"{time.time():.6f}"
        techsupport_contextual_reembed.save_state(reembed_state)
        print(f"Reembed done: {reembed_result}")
        return

    result = run(limit=args.limit, dry_run=args.dry_run, skip_reembed=args.skip_reembed)
    print_summary(result)


if __name__ == "__main__":
    main()
