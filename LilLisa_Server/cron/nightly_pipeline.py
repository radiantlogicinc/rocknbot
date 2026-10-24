"""
nightly_pipeline.py
====================================
End-to-end nightly techsupport pipeline, tying together the three pieces
built in earlier steps:
    1. nightly_techsupport_sync.sync() -- detect new/updated threads in the
       one shared techsupport Slack channel (TECHSUPPORT_CHANNEL_ID in
       cron/env/techsupport_sync.env; the same variable lil-lisa uses for
       every product).
    2. techsupport_classifier.classify_thread() -- classify each changed
       thread as useful / conclusive.
    3. techsupport_qa_ingest.add_verified_qa_pair() -- for threads that are
       both useful and conclusive, extract a Q&A pair and add it to the
       single shared verified-techsupport store (product-agnostic; there is
       no per-product classification step). Entries are retrievable
       immediately -- there is no expert-review gate.

Each thread is processed independently (its own try/except) so one bad
thread can't take down the whole run. A per-run summary is logged, and if
any thread errored, the error details are posted to the shared admin Slack
channel (ADMIN_CHANNEL_ID in cron/env/techsupport_sync.env -- reuses the
same channel lil-lisa's ADMIN_CHANNEL_ID_IDA/ADMIN_CHANNEL_ID_IDDM already
point at).

How often step 1 actually checks the Slack channel is itself interval-gated
by TECHSUPPORT_SYNC_INTERVAL_HOURS (env/lillisa_server.env, default 24, i.e.
once a day) -- is_channel_check_due() below checks elapsed time since the
last check (tracked in techsupport_sync_state.json's last_run_timestamp,
which nightly_techsupport_sync.sync() already maintains) and skips calling
sync() entirely if the interval hasn't elapsed, the same
"check-elapsed-time-skip-if-not-due" pattern
techsupport_contextual_reembed.is_reembed_due() uses. This is independent of
TECHSUPPORT_REEMBED_INTERVAL_DAYS below, which controls how often the
contextual re-embed runs, not how often the channel gets checked.

Dedup: whether a thread has already been added to the verified DB is tracked
in techsupport_sync_state.json (the same per-thread state file
nightly_techsupport_sync.py already maintains for last_seen_reply_ts), via an
"added_to_verified_db": true field. A thread is only marked this way AFTER
add_verified_qa_pair() returns successfully, and the flag is saved to disk
immediately (not batched until the end of the run) so a crash partway through
a run doesn't lose track of threads that were already added -- a retry will
correctly skip those and only reconsider threads that weren't marked.

An already-added thread is only ever reconsidered when sync() itself detects
new reply activity on it (i.e. it shows up in updated_thread_ids); a thread
with no new activity is never touched, added_to_verified_db=true or not. When
one IS reconsidered, it is re-classified against the complete, current
conversation. If still useful+conclusive, the existing verified entry is
REPLACED in place (full regeneration from the whole updated conversation,
old entry fully discarded) -- see process_thread()'s already_added branch and
techsupport_qa_ingest.replace_verified_qa_pair(). If no longer useful or
conclusive, the existing entry is deliberately left as-is rather than
removed, since a changed classification isn't evidence the old answer became
wrong. Replacement relies on the entry's source thread_ts having been
recorded when it was added (techsupport_review_state.json); entries added
before that tracking existed can't be located and are left untouched, logged
as an error for that thread rather than silently skipped.

A separate pass, run_product_channel_pass(), runs after the techsupport loop
and before the GitHub push: the product channels (CHANNEL_ID_IDA / _IDDM /
_IDO from lil-lisa.env, or the process environment) are scanned for threads
where a member of that product's expert user group replied to a Lil Lisa answer. A
thread with no expert reply never reaches an LLM call. When the answer in that
thread cited a verified entry (recorded at answer time by invoke(), read back
via get_cited_entry_title), the expert's reply SUPERSEDES that entry
(correct_verified_entry); otherwise a useful+conclusive thread is added like
any other. The pass shares this run's error list, admin alert, GitHub push and
index reload, is gated per channel by TECHSUPPORT_SYNC_INTERVAL_HOURS, capped
by PRODUCT_SCAN_MAX_THREADS_PER_RUN, and turned off entirely by
TECHSUPPORT_SCAN_PRODUCT_CHANNELS=false (env/lillisa_server.env). The
hot/catch-up windows nightly_techsupport_sync.py applies (TECHSUPPORT_SYNC_HOT_DAYS,
TECHSUPPORT_SYNC_CATCHUP_AGE_DAYS, TECHSUPPORT_SYNC_CATCHUP_INTERVAL_DAYS) are
per channel, so they govern each product channel exactly as they govern the
techsupport channel. The very first scan of a product channel is additionally
bounded by PRODUCT_SCAN_INITIAL_LOOKBACK_DAYS (default 30): a channel with no
state would otherwise be synced from timestamp 0, i.e. its entire history, so
the pass seeds that channel's last_run_timestamp to now minus the lookback
before the first sync() call. The techsupport channel is untouched by this.

run_product_channel_scan() runs the product pass on its own, without the
techsupport loop, the review sync or the re-embed, and is what POST
/run_product_channel_scan/ triggers (see src/techsupport_cron.py). It still
does the GitHub push and index reload when the pass changed something, and
force=True makes it ignore the per-channel interval gate.

A fourth, less-frequent piece also runs at the end of every invocation:
techsupport_contextual_reembed.run_if_due() -- a periodic full re-embed of
the TECHSUPPORT_QA_PAIRS table using Voyage's contextual embedding (whole
techsupport_qa_pairs.md as one document, so every entry benefits from
full-document context, not just what existed when it was individually
added). It only actually re-embeds once every TECHSUPPORT_REEMBED_INTERVAL_DAYS
(env/lillisa_server.env, default 7); otherwise it no-ops. Keeping this call
here instead of a separate cron entry keeps all techsupport operations
consolidated into the one nightly script.

A fifth piece, techsupport_review_sync.sync_edited_entries(), runs after every
invocation too: it picks up any markdown entries an expert has since edited
directly (see techsupport_review_sync.py) and syncs the current text into
LanceDB. This is purely optional/best-effort -- there is no gating tied to it.

A sixth piece, github_sync.push_verified_qa_pairs() (see FLAGGED OPEN ITEM in
techsupport_qa_ingest.py), pushes techsupport_qa_pairs.md to a dedicated
private GitHub repo (GITHUB_TOKEN / GITHUB_REPO_URL in
cron/env/github_push.env) -- but only when this run actually added,
enriched, or replaced an entry (counts["added"] or counts["enriched"] or
counts["replaced"]), since that's the only way the markdown file's content
changes. push_verified_qa_pairs() itself
also no-ops if the file turns out to be byte-identical to what's already in
the repo, so this is a double-safe skip against pushing empty commits.

Scheduling: the API process runs this on its own tick, via
LilLisa_Server/src/techsupport_cron.py -- there is no crontab. POST
/run_nightly_pipeline/ forces a full run against a live server, and POST
/run_product_channel_scan/ forces just the product-channel pass.

Usage (manual/debugging only; don't run this while the API is up, since the
in-process lock cannot see a second OS process):
    python nightly_pipeline.py
"""

import logging
import os
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import jwt
import requests
from dotenv import dotenv_values
from paths import (
    LILLISA_SERVER_ENV_PATH,
    LILLISA_SERVER_ROOT,
    PACKAGE_ROOT,
    ensure_import_paths,
)

ensure_import_paths()
SCRIPT_DIR = PACKAGE_ROOT
PROJECT_ROOT = LILLISA_SERVER_ROOT

import expert_group  # noqa: E402
from github_sync import push_verified_qa_pairs  # noqa: E402
from nightly_techsupport_sync import (  # noqa: E402
    channel_state,
    load_env,
    load_state,
    paginate_messages,
    parent_activity_ts,
    product_channel_ids,
    save_state,
    sync,
)
from pipeline_summary import format_pipeline_counts  # noqa: E402
from techsupport_classifier import (  # noqa: E402
    classify_thread,
    format_thread_messages,
    has_expert_insight,
)
from techsupport_contextual_reembed import (  # noqa: E402
    run_if_due as run_reembed_if_due,
)
from techsupport_qa_ingest import (  # noqa: E402
    add_verified_qa_pair,
    correct_verified_entry,
    enrich_verified_entry,
    get_cited_entry_title,
    get_related_entry_title,
    replace_verified_qa_pair,
)
from techsupport_review_sync import sync_edited_entries  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("nightly_pipeline")

SLACK_API_BASE = "https://slack.com/api"
DEFAULT_SYNC_INTERVAL_HOURS = 24
DEFAULT_SCAN_PRODUCT_CHANNELS = "true"
DEFAULT_PRODUCT_SCAN_MAX_THREADS_PER_RUN = 50
DEFAULT_PRODUCT_SCAN_INITIAL_LOOKBACK_DAYS = 30.0
SECONDS_PER_DAY = 86400
FALSEY = {"false", "0", "no", "off", ""}

# Outcomes process_product_thread() can return, in summary order.
PRODUCT_COUNT_KEYS = (
    "checked",
    "corrected",
    "added",
    "replaced",
    "left_as_is_not_useful",
    "left_as_is_not_conclusive",
    "skipped_no_expert_reply",
    "skipped_no_expert_insight",
    "skipped_not_useful",
    "skipped_not_conclusive",
    "errored",
)
# Outcomes that changed techsupport_qa_pairs.md, so the GitHub push and the
# index reload below have something to do.
PRODUCT_CHANGE_KEYS = ("corrected", "added", "replaced")
# Same default lil-lisa/app_envfiles/lil-lisa.env's LIL_LISA_SERVER_URL uses --
# this script and the Slack bot normally run alongside the same LilLisa_Server
# instance on the same host. Override via LIL_LISA_SERVER_URL if not.
DEFAULT_LIL_LISA_SERVER_URL = "http://127.0.0.1:8000"


def get_sync_interval_hours() -> float:
    """Expert-adjustable knob: how many hours must elapse between checks of
    the techsupport Slack channel. Overridable per-process via env var
    without editing the env file -- same pattern as
    techsupport_contextual_reembed.get_reembed_interval_days()."""
    env = dict(dotenv_values(str(LILLISA_SERVER_ENV_PATH)))
    env = {**env, **os.environ}
    raw = env.get("TECHSUPPORT_SYNC_INTERVAL_HOURS", str(DEFAULT_SYNC_INTERVAL_HOURS))
    return float(raw)


def _lillisa_env() -> Dict[str, str]:
    """env/lillisa_server.env overlaid by the process environment."""
    env = dict(dotenv_values(str(LILLISA_SERVER_ENV_PATH)))
    env.update(os.environ)
    return env


def get_scan_product_channels() -> bool:
    """TECHSUPPORT_SCAN_PRODUCT_CHANNELS (default true): master switch for the
    product-channel expert-correction pass. Set it to false to turn the whole
    pass off without unsetting the per-product channel ids."""
    raw = _lillisa_env().get("TECHSUPPORT_SCAN_PRODUCT_CHANNELS", DEFAULT_SCAN_PRODUCT_CHANNELS)
    return str(raw).strip().lower() not in FALSEY


def get_product_scan_max_threads_per_run() -> int:
    """PRODUCT_SCAN_MAX_THREADS_PER_RUN (default 50): per-product cap on how
    many changed threads one run will classify. Excess threads are remembered
    in the channel's state (pending_thread_ids) and picked up next run, hottest
    first, so a busy day cannot turn into an unbounded LLM bill."""
    raw = _lillisa_env().get("PRODUCT_SCAN_MAX_THREADS_PER_RUN", str(DEFAULT_PRODUCT_SCAN_MAX_THREADS_PER_RUN))
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError):
        logger.warning(
            "PRODUCT_SCAN_MAX_THREADS_PER_RUN=%r is not a number -- using %s",
            raw,
            DEFAULT_PRODUCT_SCAN_MAX_THREADS_PER_RUN,
        )
        return DEFAULT_PRODUCT_SCAN_MAX_THREADS_PER_RUN


def get_product_scan_initial_lookback_days() -> float:
    """PRODUCT_SCAN_INITIAL_LOOKBACK_DAYS (default 30): how far back the FIRST
    scan of a product channel looks. A channel with no state has
    last_run_timestamp "0", which would make nightly_techsupport_sync.sync()
    pull the channel's whole history and hand every old thread with an expert
    reply to the classifier. Seeding the window keeps that first run bounded;
    every later run just picks up where the previous one stopped."""
    raw = _lillisa_env().get("PRODUCT_SCAN_INITIAL_LOOKBACK_DAYS", str(DEFAULT_PRODUCT_SCAN_INITIAL_LOOKBACK_DAYS))
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        logger.warning(
            "PRODUCT_SCAN_INITIAL_LOOKBACK_DAYS=%r is not a number -- using %s",
            raw,
            DEFAULT_PRODUCT_SCAN_INITIAL_LOOKBACK_DAYS,
        )
        return DEFAULT_PRODUCT_SCAN_INITIAL_LOOKBACK_DAYS


def reload_techsupport_index() -> Dict[str, Any]:
    """Tell the running LilLisa_Server to rebuild its in-memory QA pairs
    retrievers/indices (including TECHSUPPORT_QA_PAIRS) from the current
    on-disk LanceDB tables, via the /reload_techsupport_qa_pairs/ endpoint.

    add_verified_qa_pair()/replace_verified_qa_pair() above write straight to
    LanceDB, entirely out-of-process from LilLisa_Server -- confirmed live
    (see investigation notes) that a long-running server process's
    TECHSUPPORT_QA_PAIRS_RETRIEVER stays frozen at whatever it was at last
    startup, so newly-added content is invisible to real queries until
    something rebuilds it. Without this call, only a manual server restart
    would pick up tonight's changes.

    Best-effort: mirrors the other post-run steps below (github push, review
    sync, reembed) -- a failure here must not fail the whole pipeline run,
    since the LanceDB write itself already succeeded and is durable; it just
    means the running server won't see it until its next restart or reload."""
    env = dict(dotenv_values(str(LILLISA_SERVER_ENV_PATH)))
    env = {**env, **os.environ}
    authentication_key = env.get("AUTHENTICATION_KEY")
    if not authentication_key:
        return {"reloaded": False, "reason": "AUTHENTICATION_KEY not found in lillisa_server.env"}

    # env.get(key, default) only falls back when the key is ABSENT. A key that is
    # PRESENT but blank or whitespace-only -- LIL_LISA_SERVER_URL= in lillisa_server.env,
    # or an empty string injected via the container environment, both trivially easy to
    # produce in a Kubernetes manifest -- must behave the same as absent, or the POST
    # below targets a bare path with no host and this reload silently fails, leaving
    # freshly ingested entries invisible to live queries (pr42-44j.9).
    raw_base_url = (env.get("LIL_LISA_SERVER_URL") or "").strip()
    base_url = (raw_base_url or DEFAULT_LIL_LISA_SERVER_URL).rstrip("/")
    encrypted_key = jwt.encode({"some": "payload"}, authentication_key, algorithm="HS256")

    resp = requests.post(
        f"{base_url}/reload_techsupport_qa_pairs/",
        params={"encrypted_key": encrypted_key},
        timeout=30,
    )
    resp.raise_for_status()
    return {"reloaded": True, "response": resp.json()}


def is_channel_check_due(channel_sync_state: Dict[str, Any]) -> bool:
    """True if TECHSUPPORT_SYNC_INTERVAL_HOURS have elapsed since the last time
    THIS channel was checked. `channel_sync_state` is one channel's slice of
    techsupport_sync_state.json (nightly_techsupport_sync.channel_state()); its
    last_run_timestamp is maintained by nightly_techsupport_sync.sync(). The
    same interval gates the techsupport channel and each product channel,
    independently of one another."""
    last = channel_sync_state.get("last_run_timestamp", "0")
    interval_seconds = get_sync_interval_hours() * 3600
    return (time.time() - float(last)) >= interval_seconds


def post_admin_alert(token: str, admin_channel_id: str, text: str) -> None:
    """Best-effort post to the shared admin channel. Alerting failures are
    logged but never allowed to fail the pipeline run itself."""
    if not admin_channel_id:
        logger.warning("ADMIN_CHANNEL_ID not configured -- skipping admin alert:\n%s", text)
        return
    try:
        resp = requests.post(
            f"{SLACK_API_BASE}/chat.postMessage",
            headers={"Authorization": f"Bearer {token}"},
            json={"channel": admin_channel_id, "text": text},
            timeout=10,
        )
        data = resp.json()
        if not data.get("ok"):
            logger.error("Failed to post admin alert: %s", data.get("error"))
    except requests.RequestException as exc:
        logger.error("Failed to post admin alert: %s", exc)


def process_thread(thread_ts: str, token: str, channel_id: str, already_added: bool) -> str:
    """Fetch, classify, and (if it clears the bar) ingest, enrich, or replace one thread.

    `already_added` is True when this thread previously had
    added_to_verified_db=true and is being reprocessed because sync()
    detected new reply activity since then. In that case a still
    useful+conclusive thread REPLACES its existing verified entry -- full
    regeneration from the complete, current conversation (original question
    plus every reply, including the new one), not a patch of just the new
    detail -- instead of being added as a new (duplicate) entry. If it's no
    longer useful/conclusive, the existing entry is deliberately left as-is:
    there's no reliable signal that the old resolution stopped being correct,
    so nothing is deleted.

    A brand-new (not already_added) thread that was tagged (via
    /tag_techsupport_thread/) as related to an existing verified entry --
    i.e. Lil Lisa answered by citing that entry and the user escalated
    anyway -- gets a LIGHTER bar: only IsUsefulConversation is required, not
    IsConclusiveConversation, since enrichment is adding supplementary
    insight to an already-resolved topic rather than needing to
    independently resolve something from scratch. Untagged threads (the
    normal add path) are unaffected -- both useful AND conclusive are still
    required, exactly as before.

    Returns one of: "added", "enriched", "replaced", "skipped_not_useful",
    "skipped_not_conclusive", "left_as_is_not_useful", "left_as_is_not_conclusive".
    Raises on any failure -- the caller is responsible for catching per-thread
    errors so one bad thread doesn't abort the whole run.
    """
    messages = paginate_messages(
        "conversations.replies", token, {"channel": channel_id, "ts": thread_ts, "limit": 200}
    )
    if not messages:
        raise RuntimeError(f"No messages found for thread {thread_ts}")

    # Enrichment only ever applies to a thread's first pass (already_added is only
    # True for a replace-path reprocessing, which keeps the full bar) -- see
    # docstring above.
    related_entry_title = None if already_added else get_related_entry_title(thread_ts)

    result = classify_thread(messages, slack_token=token, skip_conclusive=bool(related_entry_title))

    if not result["is_useful"]:
        return "left_as_is_not_useful" if already_added else "skipped_not_useful"

    if related_entry_title:
        try:
            enrich_verified_entry(
                related_entry_title,
                result["conversation_thread"],
                source_channel_id=channel_id,
                source_thread_ts=thread_ts,
            )
            return "enriched"
        except LookupError as exc:
            logger.warning(
                "Thread %s tagged related_entry_title=%r but enrich failed (%s) -- " "falling back to normal add",
                thread_ts,
                related_entry_title,
                exc,
            )
            # The enrich attempt's lighter bar doesn't carry over to the add fallback --
            # it still needs the full useful+conclusive bar, which wasn't evaluated
            # above (conclusiveness was deliberately skipped). Re-classify with the
            # default (skip_conclusive=False) rather than duplicating the
            # IsConclusiveConversation "yes"/"no" comparison here.
            full_result = classify_thread(messages, slack_token=token)
            if not full_result["is_conclusive"]:
                return "skipped_not_conclusive"
            add_verified_qa_pair(full_result["conversation_thread"], thread_ts=thread_ts)
            return "added"

    if not result["is_conclusive"]:
        return "left_as_is_not_conclusive" if already_added else "skipped_not_conclusive"

    if already_added:
        replace_verified_qa_pair(thread_ts, result["conversation_thread"])
        return "replaced"

    add_verified_qa_pair(result["conversation_thread"], thread_ts=thread_ts)
    return "added"


def new_product_counts() -> Dict[str, Any]:
    return {key: 0 for key in PRODUCT_COUNT_KEYS}


def has_expert_reply(messages: Sequence[Dict[str, Any]], thread_ts: str, expert_ids: Iterable[str]) -> bool:
    """True if some message OTHER than the thread parent was posted by an expert.

    An expert opening a thread with a question of their own is not a
    correction, so the parent never counts -- which is also what keeps the
    common case (an ordinary product-channel question) from ever reaching an
    LLM call.
    """
    experts = {expert for expert in (expert_ids or ()) if expert}
    if not experts:
        return False
    for index, message in enumerate(messages):
        if index == 0 or message.get("ts") == thread_ts:
            continue
        if message.get("user") in experts:
            return True
    return False


def process_product_thread(
    thread_ts: str,
    token: str,
    channel_id: str,
    product: str,
    already_added: bool,
    *,
    expert_ids: Sequence[str],
    bot_user_id: Optional[str] = None,
    previous_outcome: Optional[str] = None,
) -> str:
    """Handle one product-channel thread that sync() flagged as new/updated.

    The use case: Lil Lisa answers a user in the IDA/IDDM/IDO channel, and a
    member of that product's expert user group replies in the same thread to
    correct or augment the answer.

    Routing (see techsupport_qa_ingest.py's module docstring):
      * The bot's answer cited a verified entry (get_cited_entry_title, from the
        tag invoke() writes at answer time) -> correct_verified_entry(), the
        SUPERSEDE path: the contradicted content must not survive. Only
        IsUsefulConversation is required here -- the same lighter bar the
        escalation-enrichment path uses, since the topic is already resolved and
        this thread only has to be worth folding in. If the cited entry is gone
        from the markdown (LookupError), fall back to the normal add path, which
        re-classifies against the FULL useful+conclusive bar.
      * No cited entry -> a plain new Q&A: full useful+conclusive bar, then
        add_verified_qa_pair().
      * `already_added` (this thread already produced or corrected an entry and
        an expert has since said something new) -> redo the same operation:
        correct_verified_entry() again when `previous_outcome` was "corrected"
        (replace_verified_qa_pair() would not find that entry -- a corrected
        entry keeps the thread_ts of the thread that originally created it, not
        this product-channel thread's), otherwise replace_verified_qa_pair().
        `previous_outcome` comes from the per-thread state; without it, the
        presence of a cited-entry tag decides.

    Two gates run before any of that routing. has_expert_reply() is the cheap
    one: no expert has posted a reply, no LLM call, "skipped_no_expert_reply".
    has_expert_insight() is the second: an expert did reply, but the reply only
    asks a question, requests clarification, or is small talk, so there is
    nothing to fold into the knowledge base and the thread is left alone as
    "skipped_no_expert_insight". Nothing is written and no flag is set, so the
    thread is reconsidered automatically the next time sync() sees new reply
    activity on it -- which is the point: the expert's later correction is
    picked up on the run after it is posted.

    Returns one of: "corrected", "added", "replaced", "skipped_no_expert_reply",
    "skipped_no_expert_insight", "skipped_not_useful", "skipped_not_conclusive",
    "left_as_is_not_useful", "left_as_is_not_conclusive". Raises on any failure
    -- the caller catches per-thread errors so one bad thread doesn't abort the
    run.
    """
    messages = paginate_messages(
        "conversations.replies", token, {"channel": channel_id, "ts": thread_ts, "limit": 200}
    )
    if not messages:
        raise RuntimeError(f"No messages found for thread {thread_ts}")

    if not has_expert_reply(messages, thread_ts, expert_ids):
        # The cheap exit, and the common one: no LLM call for a thread no
        # expert has weighed in on.
        return "skipped_no_expert_reply"

    if not has_expert_insight(
        format_thread_messages(
            messages,
            slack_token=token,
            bot_user_id=bot_user_id,
            expert_user_ids=expert_ids,
        )
    ):
        # The expert only asked a question or chatted: nothing to ingest, and
        # nothing recorded, so new activity on this thread brings it back.
        return "skipped_no_expert_insight"

    def classify(skip_conclusive: bool = False) -> Dict[str, Any]:
        return classify_thread(
            messages,
            slack_token=token,
            skip_conclusive=skip_conclusive,
            bot_user_id=bot_user_id,
            expert_user_ids=expert_ids,
        )

    cited_title = get_cited_entry_title(thread_ts)

    if already_added:
        prior = previous_outcome or ("corrected" if cited_title else "added")
        if prior == "corrected" and cited_title:
            result = classify(skip_conclusive=True)
            if not result["is_useful"]:
                return "left_as_is_not_useful"
            try:
                correct_verified_entry(
                    cited_title,
                    result["conversation_thread"],
                    source_channel_id=channel_id,
                    source_thread_ts=thread_ts,
                )
                return "corrected"
            except LookupError as exc:
                logger.warning(
                    "Thread %s previously corrected %r but the entry is gone (%s) -- trying replace",
                    thread_ts,
                    cited_title,
                    exc,
                )
        full_result = classify()
        if not full_result["is_useful"]:
            return "left_as_is_not_useful"
        if not full_result["is_conclusive"]:
            return "left_as_is_not_conclusive"
        replace_verified_qa_pair(thread_ts, full_result["conversation_thread"])
        return "replaced"

    if cited_title:
        result = classify(skip_conclusive=True)
        if not result["is_useful"]:
            return "skipped_not_useful"
        try:
            correct_verified_entry(
                cited_title,
                result["conversation_thread"],
                source_channel_id=channel_id,
                source_thread_ts=thread_ts,
            )
            return "corrected"
        except LookupError as exc:
            logger.warning(
                "Thread %s cited entry %r but correct failed (%s) -- falling back to normal add",
                thread_ts,
                cited_title,
                exc,
            )
            # The correction path's lighter bar doesn't carry over to the add
            # fallback: conclusiveness was deliberately skipped above, so
            # re-classify with the default rather than duplicating the
            # IsConclusiveConversation comparison here (same shape as
            # process_thread()'s enrich fallback).
            full_result = classify()
            if not full_result["is_conclusive"]:
                return "skipped_not_conclusive"
            add_verified_qa_pair(full_result["conversation_thread"], thread_ts=thread_ts)
            return "added"

    result = classify()
    if not result["is_useful"]:
        return "skipped_not_useful"
    if not result["is_conclusive"]:
        return "skipped_not_conclusive"
    add_verified_qa_pair(result["conversation_thread"], thread_ts=thread_ts)
    return "added"


def seed_initial_scan_window(state: Dict[str, Any], per_channel: Dict[str, Any], channel_id: str, product: str) -> bool:
    """Bound a product channel's FIRST sync to PRODUCT_SCAN_INITIAL_LOOKBACK_DAYS.

    A channel nothing has synced yet has last_run_timestamp "0", which
    nightly_techsupport_sync.find_new_threads() reads as "everything ever
    posted here" -- every old thread that happens to carry an expert reply
    would then be classified and ingested, a few dozen per run, for as long as
    the backlog lasts. Seeding the timestamp to now minus the lookback means
    the first run only sees threads created inside that window.

    Product channels only: run_pipeline()'s techsupport channel keeps its
    original from-the-beginning first sync. Returns True if a seed was written.
    """
    try:
        last_run = float(per_channel.get("last_run_timestamp") or 0)
    except (TypeError, ValueError):
        last_run = 0.0
    if last_run > 0:
        return False

    lookback_days = get_product_scan_initial_lookback_days()
    oldest = max(0.0, time.time() - lookback_days * SECONDS_PER_DAY)
    per_channel["last_run_timestamp"] = f"{oldest:.6f}"
    save_state(state)
    logger.info(
        "Product channel %s (%s) has no sync state: scanning it for the first time with a "
        "PRODUCT_SCAN_INITIAL_LOOKBACK_DAYS=%s day lookback instead of its whole history",
        channel_id,
        product,
        lookback_days,
    )
    return True


def run_product_channel_pass(
    token: str,
    *,
    slack_env: Optional[Mapping[str, str]] = None,
    errors: Optional[List[Dict[str, str]]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Scan each configured product channel for expert corrections.

    One pass per product (CHANNEL_ID_IDA / _IDDM / _IDO, the same variables the
    Slack bot uses; a product with no channel id is not scanned). Each channel
    gets its own slice of techsupport_sync_state.json
    and its own TECHSUPPORT_SYNC_INTERVAL_HOURS gate, and is capped at
    PRODUCT_SCAN_MAX_THREADS_PER_RUN threads per run. A channel being scanned
    for the first time is bounded by PRODUCT_SCAN_INITIAL_LOOKBACK_DAYS
    (seed_initial_scan_window above).

    `force` skips the per-channel interval gate, which is what an operator
    triggering a scan by hand (POST /run_product_channel_scan/) wants; nothing
    else about the pass changes, so the caps and the dedup state still apply.

    Per-thread failures are appended to `errors` (the caller's list, so they
    ride along in the same admin alert as the techsupport loop's) and never
    abort the pass. So does a product whose expert user group is unconfigured
    (ValueError) or unreadable (expert_group.ExpertLookupError): that product
    is counted as errored and reported rather than quietly skipped.

    Returns {"enabled": bool, "products": {product: counts}, "totals": counts}.
    """
    totals = new_product_counts()
    if not get_scan_product_channels():
        logger.info("Product-channel scan disabled (TECHSUPPORT_SCAN_PRODUCT_CHANNELS is false)")
        return {"enabled": False, "products": {}, "totals": totals}

    slack_env = slack_env if slack_env is not None else load_env()
    bot_user_id = (slack_env.get("LIL_LISA_SLACK_USERID") or "").strip() or None
    channels = product_channel_ids(slack_env)
    if not channels:
        logger.info("No CHANNEL_ID_IDA / _IDDM / _IDO configured -- nothing to scan for expert corrections")
        return {"enabled": True, "products": {}, "totals": totals, "reason": "no_product_channels"}

    if errors is None:
        errors = []
    max_threads = get_product_scan_max_threads_per_run()
    products: Dict[str, Any] = {}

    for product, channel_id in channels.items():
        counts = new_product_counts()
        counts["channel_id"] = channel_id
        products[product] = counts

        # A configured product channel whose expert group is missing or
        # unreadable is a configuration/outage problem, not a "nothing to do":
        # skipping it silently would look like a clean run that found no
        # corrections. Record it as an error for this product (which feeds the
        # admin alert) and move on to the next channel.
        try:
            expert_ids = expert_group.expert_user_ids(product)
        except (ValueError, expert_group.ExpertLookupError) as exc:
            counts["errored"] += 1
            errors.append({"thread_ts": f"channel {channel_id}", "error": f"[{product}] {exc}"})
            logger.exception("Could not resolve experts for %s (channel %s)", product, channel_id)
            continue

        state_before = load_state()
        per_channel_before = channel_state(state_before, channel_id)
        if force:
            logger.info(
                "Product channel %s (%s): TECHSUPPORT_SYNC_INTERVAL_HOURS gate bypassed (forced scan)",
                channel_id,
                product,
            )
        elif not is_channel_check_due(per_channel_before):
            logger.info(
                "Product channel %s (%s) check skipped: TECHSUPPORT_SYNC_INTERVAL_HOURS (%s) not yet elapsed",
                channel_id,
                product,
                get_sync_interval_hours(),
            )
            counts["skipped_reason"] = "interval_not_elapsed"
            continue

        # Must happen before sync(), which is what reads last_run_timestamp.
        seed_initial_scan_window(state_before, per_channel_before, channel_id, product)

        sync_result = sync(channel_id=channel_id)

        # sync() just wrote this channel's last_seen_reply_ts values; re-load so
        # they're combined with the added_to_verified_db/outcome flags earlier
        # runs left behind (same reason the techsupport loop re-loads).
        state = load_state()
        per_channel = channel_state(state, channel_id)
        threads_state = per_channel["threads"]

        candidates: List[str] = []
        for thread_ts in (
            list(per_channel.get("pending_thread_ids") or [])
            + sync_result["new_thread_ids"]
            + sync_result["updated_thread_ids"]
        ):
            if thread_ts not in candidates:
                candidates.append(thread_ts)
        # Hottest first, so the cap defers the least recently active threads.
        candidates.sort(key=lambda ts: parent_activity_ts(ts, threads_state.get(ts) or {}), reverse=True)
        selected, deferred = candidates[:max_threads], candidates[max_threads:]
        if deferred:
            logger.info(
                "Product channel %s (%s): %d thread(s) over PRODUCT_SCAN_MAX_THREADS_PER_RUN=%d deferred to next run",
                channel_id,
                product,
                len(deferred),
                max_threads,
            )
        # Persist the deferral before doing any work: sync() has already moved
        # this channel's last_seen_reply_ts forward, so a deferred thread would
        # otherwise never be reported as updated again.
        per_channel["pending_thread_ids"] = deferred
        save_state(state)

        for thread_ts in selected:
            counts["checked"] += 1
            thread_info = threads_state.get(thread_ts) or {}
            already_added = bool(thread_info.get("added_to_verified_db"))
            try:
                outcome = process_product_thread(
                    thread_ts,
                    token,
                    channel_id,
                    product,
                    already_added,
                    expert_ids=expert_ids,
                    bot_user_id=bot_user_id,
                    previous_outcome=thread_info.get("outcome"),
                )
                counts[outcome] += 1
                logger.info("Product %s thread %s: %s", product, thread_ts, outcome)
                if outcome in PRODUCT_CHANGE_KEYS:
                    # Mark and persist immediately, like the techsupport loop, so a
                    # crash later in the run can't cause a re-add. "outcome" records
                    # WHICH operation to repeat next time this thread changes:
                    # a corrected entry can only be corrected again (replace can't
                    # find it), an added one is replaced.
                    entry = threads_state.setdefault(thread_ts, {})
                    entry["added_to_verified_db"] = True
                    entry["outcome"] = "corrected" if outcome == "corrected" else "added"
                    save_state(state)
            except Exception as exc:  # noqa: BLE001 -- one bad thread must not abort the pass
                counts["errored"] += 1
                errors.append({"thread_ts": thread_ts, "error": f"[{product}] {exc}"})
                logger.exception("Product %s thread %s: errored", product, thread_ts)

        logger.info("Product channel %s (%s): %s", channel_id, product, format_pipeline_counts(counts))

    for key in PRODUCT_COUNT_KEYS:
        totals[key] = sum(int(counts.get(key, 0)) for counts in products.values())

    return {"enabled": True, "products": products, "totals": totals}


def publish_changes_if_any(
    token: str, admin_channel_id: Optional[str], changed_this_run: bool
) -> Dict[str, Any]:
    """Publish this run's markdown changes: send techsupport_qa_pairs.md to the
    private GitHub repo, then tell the running server to rebuild its in-memory
    techsupport retrievers so the new content is queryable without a restart.

    Both steps only have something to do when the run actually added, enriched,
    replaced or corrected an entry, and both are best-effort: a failure is
    logged and alerted but never raised, because the LanceDB write it follows
    already succeeded and is durable. Shared by run_pipeline() and
    run_product_channel_scan(), so the manual product scan publishes its
    corrections exactly the way the nightly run does.

    Returns {"github_push": ..., "techsupport_index_reload_after_ingest": ...},
    the two keys both callers put straight into their summary.
    """
    try:
        if changed_this_run:
            github_push_result = push_verified_qa_pairs()
            logger.info("GitHub push of verified techsupport QA pairs: %s", github_push_result)
        else:
            github_push_result = {"pushed": False, "reason": "no_changes_this_run"}
    except Exception as exc:  # noqa: BLE001 -- a push failure must not fail the whole pipeline run
        github_push_result = {"pushed": False, "error": str(exc)}
        logger.exception("GitHub push of verified techsupport QA pairs errored")
        post_admin_alert(
            token, admin_channel_id, f"nightly_pipeline.py: GitHub push of verified techsupport QA pairs failed: {exc}"
        )

    try:
        if changed_this_run:
            reload_after_ingest_result = reload_techsupport_index()
            logger.info("LilLisa_Server techsupport index reload (after ingest): %s", reload_after_ingest_result)
            if not reload_after_ingest_result.get("reloaded"):
                post_admin_alert(
                    token,
                    admin_channel_id,
                    f"nightly_pipeline.py: LilLisa_Server techsupport index reload (after ingest) did not run: {reload_after_ingest_result}",
                )
        else:
            reload_after_ingest_result = {"reloaded": False, "reason": "no_changes_this_run"}
    except Exception as exc:  # noqa: BLE001 -- a reload failure must not fail the whole pipeline run
        reload_after_ingest_result = {"reloaded": False, "error": str(exc)}
        logger.exception("LilLisa_Server techsupport index reload (after ingest) errored")
        post_admin_alert(
            token,
            admin_channel_id,
            f"nightly_pipeline.py: LilLisa_Server techsupport index reload (after ingest) failed: {exc}",
        )
    return {
        "github_push": github_push_result,
        "techsupport_index_reload_after_ingest": reload_after_ingest_result,
    }


def run_product_channel_scan(force: bool = False) -> Dict[str, Any]:
    """Run ONLY the product-channel expert-correction pass, then publish.

    What POST /run_product_channel_scan/ triggers, via src/techsupport_cron.py
    and under the same in-process lock as the nightly run. The techsupport
    loop, the review sync and the contextual re-embed are deliberately not run:
    an operator asking for a product-channel scan wants that scan, not a whole
    nightly cycle. The publish step above IS run, on the same "only when
    something changed" condition run_pipeline() uses, so a correction made here
    is live and backed up immediately.

    `force=True` bypasses the per-channel TECHSUPPORT_SYNC_INTERVAL_HOURS gate.

    Returns {"product_channels", "github_push",
    "techsupport_index_reload_after_ingest", "errors"}.
    """
    slack_env = load_env()
    token = slack_env["SLACK_BOT_TOKEN"]
    admin_channel_id = slack_env.get("ADMIN_CHANNEL_ID")

    errors: List[Dict[str, str]] = []
    try:
        product_result = run_product_channel_pass(token, slack_env=slack_env, errors=errors, force=force)
    except Exception as exc:  # noqa: BLE001 -- report the failure, never raise into a background task
        product_result = {"enabled": True, "products": {}, "totals": new_product_counts(), "error": str(exc)}
        logger.exception("Product-channel expert-correction pass errored")
        post_admin_alert(token, admin_channel_id, f"nightly_pipeline.py: product-channel pass failed: {exc}")

    product_totals = product_result.get("totals") or {}
    changed_this_run = bool(any(product_totals.get(key) for key in PRODUCT_CHANGE_KEYS))
    post_ingest = publish_changes_if_any(token, admin_channel_id, changed_this_run)

    if product_result.get("enabled"):
        logger.info("Product-channel scan summary: %s", format_pipeline_counts(product_totals))

    if errors:
        error_lines = "\n".join(f"- {e['thread_ts']}: {e['error']}" for e in errors)
        alert_text = (
            f"nightly_pipeline.py (product-channel scan): {int(product_totals.get('errored', 0))} thread(s) "
            f"errored out of {int(product_totals.get('checked', 0))} checked "
            f"({format_pipeline_counts(product_totals, omit=('checked',))}).\n{error_lines}"
        )
        post_admin_alert(token, admin_channel_id, alert_text)

    return {"product_channels": product_result, "errors": errors, **post_ingest}


def run_pipeline(force: bool = False) -> Dict[str, Any]:
    """Run the complete techsupport pipeline.

    Scheduled runs preserve the normal per-channel interval gates. An operator
    request uses ``force=True`` to collect the current techsupport and product
    channel activity immediately, while retaining sync()'s thread-level
    deduplication.
    """
    slack_env = load_env()
    token = slack_env["SLACK_BOT_TOKEN"]
    channel_id = slack_env["TECHSUPPORT_CHANNEL_ID"]
    admin_channel_id = slack_env.get("ADMIN_CHANNEL_ID")

    sync_state_before = load_state()
    if force or is_channel_check_due(channel_state(sync_state_before, channel_id)):
        sync_result = sync()
    else:
        logger.info(
            "Techsupport channel check skipped: TECHSUPPORT_SYNC_INTERVAL_HOURS (%s) not yet elapsed since last check",
            get_sync_interval_hours(),
        )
        sync_result = {"new_thread_ids": [], "updated_thread_ids": []}

    thread_ids = sync_result["new_thread_ids"] + sync_result["updated_thread_ids"]

    # sync() (when it ran) already wrote its own updates (last_seen_reply_ts
    # etc.) to techsupport_sync_state.json, so re-load it fresh here to get
    # those plus any added_to_verified_db flags from previous pipeline runs.
    state = load_state()
    threads_state = channel_state(state, channel_id)["threads"]

    counts = {
        "checked": 0,
        "added": 0,
        "enriched": 0,
        "replaced": 0,
        "left_as_is_not_useful": 0,
        "left_as_is_not_conclusive": 0,
        "skipped_not_useful": 0,
        "skipped_not_conclusive": 0,
        "errored": 0,
    }
    errors: List[Dict[str, str]] = []

    for thread_ts in thread_ids:
        counts["checked"] += 1

        # A thread only reaches here (i.e. is in thread_ids at all) if it's brand new
        # OR sync() detected new reply activity on it -- an already-added thread with no
        # new activity was never included in new_thread_ids/updated_thread_ids to begin
        # with, so it's implicitly left alone without any check needed here.
        already_added = bool(threads_state.get(thread_ts, {}).get("added_to_verified_db"))

        try:
            outcome = process_thread(thread_ts, token, channel_id, already_added)
            counts[outcome] += 1
            logger.info("Thread %s: %s", thread_ts, outcome)

            if outcome in ("added", "enriched"):
                # Mark and persist immediately (not batched) so a crash on a
                # later thread doesn't lose track of this one having already
                # been added -- a retry must not re-add/re-enrich it. "enriched"
                # is marked the same way as "added" since this thread_ts was never
                # previously added -- without this, an enriched thread with no new
                # activity would be reconsidered as a fresh add candidate forever.
                threads_state.setdefault(thread_ts, {})["added_to_verified_db"] = True
                save_state(state)
        except Exception as exc:  # noqa: BLE001 -- one bad thread must not abort the run
            counts["errored"] += 1
            errors.append({"thread_ts": thread_ts, "error": str(exc)})
            logger.exception("Thread %s: errored", thread_ts)

    try:
        product_result = run_product_channel_pass(token, slack_env=slack_env, errors=errors, force=force)
    except Exception as exc:  # noqa: BLE001 -- the product pass must not fail the whole run
        product_result = {"enabled": True, "products": {}, "totals": new_product_counts(), "error": str(exc)}
        logger.exception("Product-channel expert-correction pass errored")
        post_admin_alert(token, admin_channel_id, f"nightly_pipeline.py: product-channel pass failed: {exc}")
    product_totals = product_result.get("totals") or {}
    # The product pass runs BEFORE these blocks so its corrections/adds count
    # toward "did anything change this run?".
    changed_this_run = bool(
        counts["added"]
        or counts["enriched"]
        or counts["replaced"]
        or any(product_totals.get(key) for key in PRODUCT_CHANGE_KEYS)
    )

    post_ingest = publish_changes_if_any(token, admin_channel_id, changed_this_run)
    github_push_result = post_ingest["github_push"]
    reload_after_ingest_result = post_ingest["techsupport_index_reload_after_ingest"]

    try:
        review_sync_result = sync_edited_entries()
        logger.info("Techsupport review sync: %s", review_sync_result)
    except Exception as exc:  # noqa: BLE001 -- a review-sync failure must not fail the whole pipeline run
        review_sync_result = {"error": str(exc)}
        logger.exception("Techsupport review sync errored")
        post_admin_alert(token, admin_channel_id, f"nightly_pipeline.py: techsupport review sync failed: {exc}")

    try:
        reembed_result = run_reembed_if_due()
        logger.info("Contextual re-embed: %s", reembed_result)
    except Exception as exc:  # noqa: BLE001 -- a reembed failure must not fail the whole pipeline run
        reembed_result = {"ran": False, "error": str(exc)}
        logger.exception("Contextual re-embed errored")
        post_admin_alert(
            token, admin_channel_id, f"nightly_pipeline.py: contextual re-embed of TECHSUPPORT_QA_PAIRS failed: {exc}"
        )

    try:
        if reembed_result.get("ran"):
            # A full re-embed rewrites every row's vector (rebuild_table), same
            # staleness root cause as add/replace above -- the running server's
            # in-memory TECHSUPPORT_QA_PAIRS_RETRIEVER would otherwise keep
            # serving similarity scores computed against the OLD vectors until
            # a manual restart.
            reload_after_reembed_result = reload_techsupport_index()
            logger.info("LilLisa_Server techsupport index reload (after reembed): %s", reload_after_reembed_result)
            if not reload_after_reembed_result.get("reloaded"):
                post_admin_alert(
                    token,
                    admin_channel_id,
                    f"nightly_pipeline.py: LilLisa_Server techsupport index reload (after reembed) did not run: {reload_after_reembed_result}",
                )
        else:
            reload_after_reembed_result = {"reloaded": False, "reason": "reembed_did_not_run"}
    except Exception as exc:  # noqa: BLE001 -- a reload failure must not fail the whole pipeline run
        reload_after_reembed_result = {"reloaded": False, "error": str(exc)}
        logger.exception("LilLisa_Server techsupport index reload (after reembed) errored")
        post_admin_alert(
            token,
            admin_channel_id,
            f"nightly_pipeline.py: LilLisa_Server techsupport index reload (after reembed) failed: {exc}",
        )

    summary = {
        "new_threads": len(sync_result["new_thread_ids"]),
        "updated_threads": len(sync_result["updated_thread_ids"]),
        **counts,
        "product_channels": product_result,
        "errors": errors,
        "github_push": github_push_result,
        "techsupport_index_reload_after_ingest": reload_after_ingest_result,
        "review_sync": review_sync_result,
        "reembed": reembed_result,
        "techsupport_index_reload_after_reembed": reload_after_reembed_result,
    }

    logger.info("Pipeline summary: %s", format_pipeline_counts(counts))
    if product_result.get("enabled"):
        logger.info("Product-channel summary: %s", format_pipeline_counts(product_totals))

    if errors:
        error_lines = "\n".join(f"- {e['thread_ts']}: {e['error']}" for e in errors)
        # Product-channel threads share this alert, so count both loops.
        total_errored = counts["errored"] + int(product_totals.get("errored", 0))
        total_checked = counts["checked"] + int(product_totals.get("checked", 0))
        # `checked` is already in the sentence; omit it from the parenthetical.
        alert_text = (
            f"nightly_pipeline.py: {total_errored} thread(s) errored out of {total_checked} checked "
            f"({format_pipeline_counts(counts, omit=('checked',))}).\n{error_lines}"
        )
        post_admin_alert(token, admin_channel_id, alert_text)

    return summary


if __name__ == "__main__":
    result = run_pipeline()
    print(result)
