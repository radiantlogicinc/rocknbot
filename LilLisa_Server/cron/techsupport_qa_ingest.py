"""
techsupport_qa_ingest.py
====================================
Step 6 of the techsupport pipeline: takes a Slack thread that has already been
classified as useful + conclusive (techsupport_classifier.classify_thread) and
adds it to a new, separate "verified techsupport" content store -- incrementally,
one thread at a time. This deliberately does NOT use the drop-and-recreate
pattern that _run_update_golden_qa_pairs_task (src/main.py) uses for the
golden QA pairs -- per Dhar's instruction, this step is about nailing the
"add one row" path first.

This store is a single table/file SHARED across IDA and IDDM (and IDO),
not split per product. There is only one techsupport Slack channel
(TECHSUPPORT_CHANNEL_ID in cron/env/techsupport_sync.env is the same
channel for all products -- see lil-lisa/app_envfiles/lil-lisa.env's
TECHSUPPORT_CHANNEL_ID_IDA/IDDM/IDO, which are all equal), and IDA/IDDM
content has real overlap, so a verified thread is useful to both bots.
There's no product classification step here at all.

Content is stored as a prose technical summary of the resolved conversation
(NOT a Question/Answer pair) plus a short generated title used as a markdown
heading -- this is blended directly into normal document retrieval
(src/agent_and_tools.py) rather than shown as a separate FAQ-style match.

*** FLAGGED OPEN ITEM for Dhar's team -- now resolved, see github_sync.py /
github_anchor.py ***
A dedicated private GitHub repo now exists for verified-techsupport content
(GITHUB_REPO_URL in cron/env/github_push.env). Unlike the golden QA pairs
pipeline -- which treats its GitHub repo (QA_PAIRS_GITHUB_REPO_URL in
lillisa_server.env) as the source of truth, cloning it fresh on every
rebuild -- this module still treats the local markdown file under
VERIFIED_TECHSUPPORT_QA_FOLDERPATH below as the source of truth for
ingestion. The GitHub repo is a one-way push destination for
backup/visibility only: github_sync.push_verified_qa_pairs() copies the
current markdown file into a fresh clone of it and pushes, called from
nightly_pipeline.py after any add/replace.

Each entry's title is used to build a real GitHub anchor link (e.g.
".../blob/main/techsupport_qa_pairs.md#{slugified-title}", see
github_anchor.py for the slug algorithm) set into node.metadata["github_url"]
alongside "title"/"source" by insert_summary_into_lancedb() below, following
the exact same pattern src/main.py already uses for "webportal_url" --
consumed by src/agent_and_tools.py's answer_from_document_retrieval the same
way. Computing the slug requires every title in the file in order (duplicate
headings get "-1", "-2", ... suffixes, numbered by position), so
add_verified_qa_pair()/replace_verified_qa_pair() re-parse the markdown file
after writing to it and pass the freshly computed URL through -- see
_compute_github_url_for_entry(). Best-effort: any failure (e.g.
GITHUB_REPO_URL not yet configured) is caught and logged, falling back to no
github_url, since a missing link must never block the actual QA pair from
being added.

Entries added here are immediately retrievable -- there is no review/approval
gate. An expert can still optionally edit an entry's summary text directly in
the markdown file at any time; techsupport_review_sync.py picks up those edits
on the next nightly run and syncs them into LanceDB.

Also unlike techsupport_classifier.py / nightly_techsupport_sync.py, this
module DOES depend on LilLisa_Server's runtime stack (llama_index, lancedb,
voyageai) because it writes directly into the same LanceDB tables the
server's retrieval code reads from. It intentionally does NOT import
src.main (that builds the whole FastAPI app and needs far more env/setup
than ingesting one QA pair requires) -- instead it imports VoyageEmbedding
from src/embedding_config.py, a small standalone module with no FastAPI/app
dependencies that main.py itself now also imports from. Both places share
one implementation (voyage-context-3, input_type="query", 2048 dims --
confirmed against the actual IDA_QA_PAIRS/IDDM_QA_PAIRS tables' schema in the
local ./lancedb), so there's no risk of the two drifting out of sync.

Nightly insert embeddings use that query-space helper. Weekly
techsupport_contextual_reembed.py re-embeds the whole markdown file with
contextual input_type="document". Until that job runs, new rows are not in
the same space as older re-embedded rows (pr42-mp.1.8; full fix is
pr42-enhancements.3).

Generated title/summary text is run through redact_obvious_pii() before
markdown/LanceDB/GitHub (emails, private IPs, Slack mentions, obvious
secrets, INC/SR/HD/TICKET ids, internal FQDNs). Hostnames, tenant names,
and product-specific ticket keys still need a real techsupport_qa_pairs.md
to tune -- see beads pr42-blockers.2.3. The summarize/merge prompts also
ask the model to omit that material; regex is the backstop.

Routing rule for an incoming classified thread (nightly_pipeline.py decides,
this module supplies the three entry points):
  * A CORRECTION thread -- an expert replying in a product channel to say a
    cited verified entry is wrong -- goes to correct_verified_entry(), which
    SUPERSEDES: contradicted content is dropped/rewritten in the expert's
    favor (CorrectTechsupportSummary).
  * An ESCALATION thread tagged as related to an existing entry via
    /tag_techsupport_thread/ goes to enrich_verified_entry(), which APPEND
    MERGES: existing content is preserved and the new insight added
    (MergeTechsupportSummaries).
  * A thread with no cited/tracked entry (or whose cited title no longer
    exists, i.e. either call above raised LookupError) falls back to
    add_verified_qa_pair(), a brand-new entry.
Both update paths reuse the existing title verbatim so the GitHub anchor stays
stable, and record provenance (source channel/thread + timestamp) under
"corrections"/"enrichments" in techsupport_review_state.json.

Required env vars (read from ../env/lillisa_server.env, same file the main
server reads):
    LANCEDB_FOLDERPATH      - path (relative to LilLisa_Server project root) to the LanceDB folder
    VOYAGE_API_KEY_FILEPATH - path (relative to LilLisa_Server project root) to a file
                               containing the Voyage AI API key

Usage (as a library):
    from techsupport_qa_ingest import add_verified_qa_pair, generate_verified_title_and_summary
    result = add_verified_qa_pair(conversation_thread)
    preview = generate_verified_title_and_summary(conversation_thread)  # no markdown/LanceDB writes
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy
from dotenv import dotenv_values
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from paths import (
    LILLISA_SERVER_ENV_PATH,
    LILLISA_SERVER_ROOT,
    PACKAGE_ROOT,
    ANSWER_TAGS_PATH,
    REVIEW_STATE_PATH,
    THREAD_TAGS_PATH,
    VERIFIED_TECHSUPPORT_QA_FOLDERPATH,
    ensure_import_paths,
)

import lancedb

ensure_import_paths()
SCRIPT_DIR = PACKAGE_ROOT
PROJECT_ROOT = LILLISA_SERVER_ROOT

from github_anchor import compute_github_urls_for_titles  # noqa: E402
from techsupport_classifier import configure_dspy_lm  # noqa: E402
from techsupport_markdown import (  # noqa: E402
    find_duplicate_entry_index,
    parse_summary_markdown,
    prepare_title_and_summary_for_markdown,
    sanitize_techsupport_summary,
    sanitize_techsupport_title,
)
from techsupport_pii import redact_obvious_pii  # noqa: E402
from techsupport_review_state import (  # noqa: E402
    node_ids_from_review_entry,
    review_entry_state,
    with_provenance,
)

from atomic_io import atomic_write_json, atomic_write_text  # noqa: E402

from src.embedding_config import VoyageEmbedding  # noqa: E402
from src.llama_index_lancedb_vector_store import LanceDBVectorStore  # noqa: E402

# No basicConfig() call here -- this is a library module imported by
# orchestrator scripts (nightly_pipeline.py, historical_import_production.py)
# that already configure logging themselves; this just needs a named logger
# that inherits their configuration.
logger = logging.getLogger(__name__)

# Stand-in for a real dedicated GitHub repo (see FLAGGED OPEN ITEM above) --
# override with VERIFIED_TECHSUPPORT_QA_FOLDERPATH env var once one exists.
# VERIFIED_TECHSUPPORT_QA_FOLDERPATH comes from paths.py (default: the
# verified_techsupport/ folder inside the techsupport state folder).

# Single shared store across IDA/IDDM/IDO -- see module docstring.
TECHSUPPORT_QA_MARKDOWN_FILENAME = "techsupport_qa_pairs.md"
TECHSUPPORT_QA_TABLE_NAME = "TECHSUPPORT_QA_PAIRS"

# Local, git-ignored record of which LanceDB node_id(s) correspond to which
# (0-indexed) markdown entry -- same "small local state file" pattern as
# techsupport_sync_state.json / techsupport_reembed_state.json. Needed by
# techsupport_review_sync.py to find the right row(s) to update when an expert
# edits an entry's question/answer text directly in the markdown file (so
# matching on current row content alone wouldn't work).
#
# Keyed by the entry's stringified 0-indexed markdown position rather than
# stored as a parallel list -- deliberately sparse, since entries added before
# this feature existed (e.g. the original hand-seeded LDAP entry) have no
# tracked node_id(s) and must be safely skippable by index lookup, not
# silently misaligned by a list-position off-by-one.
#
# When an entry was added from a nightly_pipeline.py thread, the entry also
# records "thread_ts" (the source thread's ts). This is what
# find_entry_index_for_thread() uses to locate an already-added thread's
# entry again later, so a late reply to that thread can REPLACE the existing
# entry in place instead of creating a duplicate. Entries with no "thread_ts"
# (added before this field existed, or added via some other path) can't be
# located this way and are left alone by the replace pipeline.
# REVIEW_STATE_PATH is defined in paths.py (the techsupport state folder).

# Local, git-ignored map of {thread_ts: related_entry_title}, written by
# LilLisa_Server's /tag_techsupport_thread/ endpoint (called from lil-lisa's
# handle_escalate_to_techsupport) when a NEW escalation thread was created
# because the user escalated after Lil Lisa answered by citing an existing
# verified techsupport entry. Read by get_related_entry_title() below, which
# nightly_pipeline.py consults to decide whether a freshly-classified thread
# should be merged into that existing entry (enrich_verified_entry) instead
# of added as a new one.
# THREAD_TAGS_PATH is defined in paths.py (the techsupport state folder).
#
# Its twin, ANSWER_TAGS_PATH (also paths.py), is the local, git-ignored map of
# {session_id: cited_entry_title} that src/main.py's invoke() writes whenever
# an answer's top reranked node was a verified techsupport entry. session_id is
# the Slack thread ts, so get_cited_entry_title() below can ask "did the bot's
# answer in this product-channel thread cite a verified entry?" -- which is what
# nightly_pipeline.py's product-channel pass uses to route an expert reply to
# correct_verified_entry() (supersede) instead of add_verified_qa_pair().


def load_review_state() -> Dict[str, Any]:
    """entries[str(i)] describes the markdown entry at (0-indexed) position i,
    if and only if that entry was added via add_verified_qa_pair (or later
    synced by techsupport_review_sync.py) since this feature existed."""
    if not REVIEW_STATE_PATH.exists():
        return {"entries": {}}
    with open(REVIEW_STATE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_related_entry_title(thread_ts: str) -> Optional[str]:
    """Return the title of the existing verified entry this thread_ts was tagged as
    related to (via /tag_techsupport_thread/), or None if untagged."""
    if not THREAD_TAGS_PATH.exists():
        return None
    with open(THREAD_TAGS_PATH, "r", encoding="utf-8") as file:
        tags = json.load(file)
    return tags.get(thread_ts)


def get_cited_entry_title(thread_ts: str) -> Optional[str]:
    """Return the title of the verified entry the bot's answer in this thread cited
    (recorded at answer time by invoke()), or None if the answer cited none."""
    if not ANSWER_TAGS_PATH.exists():
        return None
    with open(ANSWER_TAGS_PATH, "r", encoding="utf-8") as file:
        tags = json.load(file)
    return tags.get(thread_ts)


def save_review_state(state: Dict[str, Any]) -> None:
    atomic_write_json(REVIEW_STATE_PATH, state)


# --- Markdown parsing (see techsupport_markdown.py) ---


def _redact_generated_title_and_summary(title: str, summary: str) -> tuple[str, str]:
    """Run obvious-PII regex on LLM output immediately before persist."""
    redacted_title = redact_obvious_pii(title)
    redacted_summary = redact_obvious_pii(summary)
    if redacted_title != title or redacted_summary != summary:
        logger.info("Redacted obvious PII from generated techsupport title/summary")
    return redacted_title, redacted_summary


def load_pipeline_env() -> Dict[str, str]:
    """Load LANCEDB_FOLDERPATH / VOYAGE_API_KEY_FILEPATH the same way
    techsupport_classifier.load_llm_env() loads its own required vars."""
    env = dict(dotenv_values(str(LILLISA_SERVER_ENV_PATH)))
    env = {**env, **os.environ}
    for key in ("LANCEDB_FOLDERPATH", "VOYAGE_API_KEY_FILEPATH"):
        if not env.get(key):
            raise RuntimeError(f"{key} not found in {LILLISA_SERVER_ENV_PATH}")
    return env


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


_embedding_configured = False


def configure_embedding_model() -> None:
    """Point Settings.embed_model at the same Voyage model/config the golden
    QA pairs tables were built with.

    This is query-space (`input_type="query"`). Weekly contextual reembed uses
    document-space; new rows match retrieval until that job runs, then match
    the rest of the techsupport table. See module docstring / pr42-mp.1.8.
    """
    global _embedding_configured
    if _embedding_configured:
        return

    env = load_pipeline_env()
    api_key_path = _resolve_path(env["VOYAGE_API_KEY_FILEPATH"])
    if not api_key_path.exists():
        raise FileNotFoundError(f"Voyage API key file not found: {api_key_path}")
    os.environ["VOYAGE_API_KEY"] = api_key_path.read_text(encoding="utf-8").strip()

    Settings.embed_model = VoyageEmbedding()
    _embedding_configured = True


# --- DSPy signatures: conversation thread -> prose summary, summary -> title ---


class SummarizeConversationThread(dspy.Signature):
    """Given a resolved technical support conversation thread, produce a
    single self-contained technical summary capturing the problem and its
    resolution -- prose, not a question/answer pair."""

    conversation_thread: str = dspy.InputField()
    summary: str = dspy.OutputField(
        desc=(
            "A clear, self-contained technical summary of the problem and its resolution, "
            "including the specific steps or fix that resolved the issue, phrased generally "
            "enough to be useful for similar future issues. Write it as standalone prose -- "
            "do not frame it as a question/answer pair, and do not reference the Slack "
            "thread, usernames, timestamps, or the fact this came from a conversation. "
            "Omit emails, hostnames, IP addresses, ticket IDs, tenant/customer names, and secrets. "
            "Content from a speaker tagged '(bot)' is AI-generated and unverified, so where a later "
            "message from a speaker tagged '(expert)' corrects it, the expert's version is "
            "authoritative and supersedes the bot's content."
        )
    )


class GenerateTechsupportTitle(dspy.Signature):
    """Given a technical support summary, produce a short descriptive title for
    it, suitable for use as a markdown heading."""

    summary: str = dspy.InputField()
    title: str = dspy.OutputField(
        desc=(
            "A short (roughly 3-7 word) descriptive title capturing the summary's technical "
            "topic, e.g. 'Zookeeper GC Logging Configuration'. Do not use markdown formatting "
            "(no '#' characters) and do not start the title itself with a heading marker. "
            "Do not include emails, hostnames, IP addresses, ticket IDs, or customer names."
        )
    )


class MergeTechsupportSummaries(dspy.Signature):
    """Given an existing verified technical support summary and new, additional
    insight from a separate but related conversation, produce an updated summary
    that incorporates the new insight into the existing one. This is a merge/append,
    not a rewrite: preserve the existing content and add the new information to it,
    rather than regenerating the summary from scratch."""

    existing_summary: str = dspy.InputField()
    new_insight: str = dspy.InputField()
    merged_summary: str = dspy.OutputField(
        desc=(
            "The existing summary, updated in place to also incorporate the new insight. "
            "Keep all still-relevant original content -- do not drop details from the "
            "existing summary that the new insight doesn't contradict. Write it as a single "
            "self-contained technical summary, standalone prose, not a question/answer pair, "
            "and do not reference the Slack thread, usernames, timestamps, or the fact this "
            "came from a conversation. Omit emails, hostnames, IP addresses, ticket IDs, "
            "tenant/customer names, and secrets. "
            "Content from a speaker tagged '(bot)' is AI-generated and unverified, so where a later "
            "message from a speaker tagged '(expert)' corrects it, the expert's version is "
            "authoritative and supersedes the bot's content."
        )
    )


class CorrectTechsupportSummary(dspy.Signature):
    """Given an existing verified technical support summary and a conversation in
    which a human expert states that it is wrong, produce a corrected summary.
    This is a supersede, not a merge/append: content the expert contradicts must
    NOT survive into the corrected summary."""

    existing_summary: str = dspy.InputField()
    correction_thread: str = dspy.InputField()
    corrected_summary: str = dspy.OutputField(
        desc=(
            "The existing summary rewritten so that anything the expert contradicts is "
            "removed or replaced with the expert's version, and everything the expert did "
            "not contradict is kept unchanged. Content from a speaker tagged '(expert)' is "
            "authoritative; content from a speaker tagged '(bot)' is AI-generated and "
            "unverified, so where the two disagree the expert wins and the bot's version "
            "must be dropped. Write it as a single self-contained technical summary, "
            "standalone prose, not a question/answer pair, and do not reference the Slack "
            "thread, usernames, timestamps, or the fact this came from a conversation. "
            "Omit emails, hostnames, IP addresses, ticket IDs, tenant/customer names, and secrets."
        )
    )


summarize_conversation = dspy.Predict(SummarizeConversationThread)
generate_title = dspy.Predict(GenerateTechsupportTitle)
merge_techsupport_summaries = dspy.Predict(MergeTechsupportSummaries)
correct_summary = dspy.Predict(CorrectTechsupportSummary)


def generate_verified_title_and_summary(conversation_thread: str) -> Dict[str, str]:
    """Public API: summarize a classified (useful + conclusive) thread and
    generate a short title, then run obvious-PII redaction.

    This is the generation path add_verified_qa_pair() / replace_verified_qa_pair()
    use before they persist. Callers that must not write markdown or LanceDB
    (e.g. historical_import_production.py --dry-run) should use this rather
    than invoking the module-level dspy.Predict instances directly.

    Returns {"title", "summary"}.
    """
    configure_dspy_lm()
    summary = summarize_conversation(conversation_thread=conversation_thread).summary.strip()
    title = generate_title(summary=summary).title.strip()
    title, summary = _redact_generated_title_and_summary(title, summary)
    title = sanitize_techsupport_title(title)
    summary = sanitize_techsupport_summary(summary)
    return {"title": title, "summary": summary}


# --- Markdown append ---


def append_summary_to_markdown(title: str, summary: str) -> Path:
    """Append one title+summary entry to the single shared
    techsupport_qa_pairs.md. Never overwrites existing entries. Entries are
    retrievable immediately -- there is no review/approval status tracked in
    the file."""
    VERIFIED_TECHSUPPORT_QA_FOLDERPATH.mkdir(parents=True, exist_ok=True)
    filepath = VERIFIED_TECHSUPPORT_QA_FOLDERPATH / TECHSUPPORT_QA_MARKDOWN_FILENAME

    block = f"## {title}\n\n{summary}\n\n"
    existing = filepath.read_text(encoding="utf-8") if filepath.exists() else ""
    atomic_write_text(filepath, existing + block)
    return filepath


def replace_summary_in_markdown(index: int, title: str, summary: str) -> Path:
    """Replace the (0-indexed) markdown entry at `index` in place -- same file
    position, fresh title/summary text -- instead of appending a new block.

    Implemented as parse-all / rewrite-whole-file rather than a surgical
    string splice: the file is small (one block per verified thread) and this
    guarantees the replaced block is byte-for-byte the same canonical format
    append_summary_to_markdown() already produces, with no risk of drifting
    out of sync with parse_summary_markdown()'s block-splitting regex.
    """
    filepath = VERIFIED_TECHSUPPORT_QA_FOLDERPATH / TECHSUPPORT_QA_MARKDOWN_FILENAME
    entries = parse_summary_markdown(filepath.read_text(encoding="utf-8"))
    if index >= len(entries):
        raise IndexError(f"Markdown entry index {index} out of range (file has {len(entries)} entries)")

    entries[index] = {"title": title, "summary": summary}
    content = "".join(f"## {entry['title']}\n\n{entry['summary']}\n\n" for entry in entries)
    atomic_write_text(filepath, content)
    return filepath


def _compute_github_url_for_entry(entry_index: int, markdown_filepath: Path) -> Optional[str]:
    """Best-effort GitHub anchor link for the (0-indexed) markdown entry at
    `entry_index`, computed by re-parsing the CURRENT file content (which
    must already include this entry -- call this after
    append_summary_to_markdown()/replace_summary_in_markdown(), not before)
    and running GithubAnchorSlugger over every title in the file in order,
    since a later entry's slug can depend on how many earlier entries share
    its base slug (see github_anchor.py).

    Returns None on any failure (e.g. GITHUB_REPO_URL not configured yet) --
    a missing/broken link must never block the actual QA pair from being
    added, so callers should treat this as advisory, not required.

    Known limitation: replace_verified_qa_pair() changes a title at an
    EXISTING position, not the end of the file. If that change happens to
    create or remove a duplicate-heading collision with some LATER entry,
    that later entry's already-stored github_url metadata (computed at ITS
    own insert/replace time) would go stale until the next full contextual
    re-embed (techsupport_contextual_reembed.py, which recomputes every
    entry's github_url from scratch every run). Accepted as out of scope:
    exact duplicate auto-generated titles are rare in practice (zero found
    among the first 672 entries), and this module's UI for detecting/fixing
    it would cost more than the risk it guards against.
    """
    try:
        titles_in_order = [
            entry["title"] for entry in parse_summary_markdown(markdown_filepath.read_text(encoding="utf-8"))
        ]
        return compute_github_urls_for_titles(titles_in_order)[entry_index]
    except Exception:  # noqa: BLE001 -- a broken GitHub link must not break ingest
        logger.warning("Could not compute github_url for markdown entry %d", entry_index, exc_info=True)
        return None


# --- Incremental LanceDB insert ---


def insert_summary_into_lancedb(title: str, summary: str, github_url: Optional[str] = None) -> Dict[str, Any]:
    """Insert one title+summary entry into the single shared
    TECHSUPPORT_QA_PAIRS table, using LanceDBVectorStore.add()'s natural
    create-or-append behavior instead of _run_update_golden_qa_pairs_task's
    drop_table()-then-rebuild pattern.

    LanceDBVectorStore.add() (src/llama_index_lancedb_vector_store.py) already
    does exactly what "add, don't rebuild" requires: on __init__ it opens the
    table if one already exists, and add() creates the table only when
    _table is None, otherwise calls `self._table.add(data, mode="append")`.
    So simply never calling db.drop_table() first is enough to get true
    incremental inserts -- there's no separate "append mode" to opt into.

    The node's `text` is the full "## {title}\\n\\n{summary}" block -- the same
    text written to markdown -- so a techsupport node looks like a normal
    document chunk to downstream retrieval/reranking/answer-synthesis code
    (src/agent_and_tools.py), with no special-casing required there. `title`,
    `source`, and (when given) `github_url` are kept in metadata (excluded
    from embedding, since title/source are already in the embedded text and
    github_url has no bearing on retrieval relevance) -- github_url is what
    lets agent_and_tools.py surface a real link to this entry's GitHub anchor,
    the same way it already does for documentation nodes' webportal_url.

    Note: the table's underlying "metadata" column has a FIXED Arrow struct
    schema (established by the very first row ever written) with no
    "github_url" field -- LanceDB silently drops metadata keys that aren't in
    that fixed schema when adding rows (confirmed empirically), so
    doc.metadata["github_url"] never survives as a queryable top-level
    column. It DOES survive inside metadata["_node_content"] (a plain string
    column holding this node's full serialized state, schema-invisible to
    Arrow), which is what metadata_dict_to_node() -- used by every read path
    in src/llama_index_lancedb_vector_store.py -- actually parses to
    reconstruct node.metadata. So this still works correctly end-to-end; it
    just isn't SQL-filterable via `metadata.github_url = ...`, which nothing
    here needs.
    """
    configure_embedding_model()
    env = load_pipeline_env()
    lancedb_folderpath = str(_resolve_path(env["LANCEDB_FOLDERPATH"]))
    table_name = TECHSUPPORT_QA_TABLE_NAME

    db = lancedb.connect(lancedb_folderpath)

    doc = Document(text=f"## {title}\n\n{summary}")
    doc.metadata["title"] = title
    doc.metadata["source"] = "techsupport"
    doc.excluded_embed_metadata_keys.extend(["title", "source"])
    doc.excluded_llm_metadata_keys.extend(["title", "source"])
    if github_url:
        doc.metadata["github_url"] = github_url
        doc.excluded_embed_metadata_keys.append("github_url")
        doc.excluded_llm_metadata_keys.append("github_url")

    splitter = SentenceSplitter(chunk_size=10000)  # summaries are short; matches golden QA pairs pipeline
    nodes = splitter.get_nodes_from_documents(documents=[doc], show_progress=False)

    vector_store = LanceDBVectorStore(
        connection=db, uri=lancedb_folderpath, table_name=table_name, query_type="hybrid"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    _ = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

    table = db.open_table(table_name)
    return {
        "table_name": table_name,
        "row_count": table.count_rows(),
        "node_ids": [node.node_id for node in nodes],
    }


def add_verified_qa_pair(conversation_thread: str, thread_ts: Optional[str] = None) -> Dict[str, Any]:
    """Full step-6 add pipeline for one classified (useful + conclusive)
    thread: summarize it into prose, generate a short title, append both to
    the shared markdown file, and insert into the shared TECHSUPPORT_QA_PAIRS
    table incrementally. Product-agnostic -- there is one shared techsupport
    channel and one shared verified-techsupport store used by both IDA and
    IDDM (and IDO).

    (Function name kept as `add_verified_qa_pair` for backward compatibility
    with nightly_pipeline.py's import -- it no longer extracts a Q&A pair,
    it produces a title+summary; see module docstring.)

    New entries are retrievable immediately -- there is no review/approval
    gate. An expert can still optionally edit the entry's text later directly
    in the markdown file; techsupport_review_sync.py picks up such edits.

    `thread_ts`, when given (nightly_pipeline.py always passes it), is
    recorded alongside the entry's node_id(s) in review_state so that a later
    reply to the same thread can be found again and used to REPLACE this
    entry in place -- see replace_verified_qa_pair() / find_entry_index_for_thread().

    Defense in depth against techsupport_review_state.json being missing,
    reset, or never having tracked this thread (e.g. a state-folder
    migration, or an entry added before thread_ts tracking existed): the
    thread_ts check above is the normal dedup guard, but it can only catch a
    repeat ingest if review_state still has this thread's mapping. If that
    mapping is gone, the SAME conversation gets re-summarized into
    near-identical prose. Before ever appending to the markdown file or
    writing to LanceDB, that regenerated summary is compared by content
    (find_duplicate_entry_index(), see techsupport_markdown.py) against every
    existing entry. A match means this thread's content is already verified
    and on file -- rather than appending a duplicate block (which would
    otherwise get a "- 2" title suffix from uniquify_techsupport_title(),
    since the title usually regenerates identically too), the existing
    entry's review_state record is backfilled with this thread_ts so future
    runs can find it the normal way, and nothing else is touched.
    """
    if thread_ts and find_entry_index_for_thread(thread_ts) is not None:
        # Retry after a crash that wrote markdown/LanceDB but not added_to_verified_db:
        # replace in place instead of appending a duplicate.
        return replace_verified_qa_pair(thread_ts, conversation_thread)

    generated = generate_verified_title_and_summary(conversation_thread)
    title, summary = generated["title"], generated["summary"]

    markdown_filepath = VERIFIED_TECHSUPPORT_QA_FOLDERPATH / TECHSUPPORT_QA_MARKDOWN_FILENAME
    existing_entries: List[Dict[str, str]] = []
    if markdown_filepath.exists():
        existing_entries = parse_summary_markdown(markdown_filepath.read_text(encoding="utf-8"))
    existing_titles = [entry["title"] for entry in existing_entries]

    duplicate_index = find_duplicate_entry_index(summary, existing_entries)
    if duplicate_index is not None:
        logger.warning(
            "add_verified_qa_pair: generated summary for thread_ts=%r matches existing markdown "
            "entry %d (%r) almost exactly -- treating as already-added instead of appending a "
            "duplicate entry. This usually means techsupport_review_state.json lost this "
            "thread's thread_ts mapping; backfilling it now.",
            thread_ts,
            duplicate_index,
            existing_entries[duplicate_index]["title"],
        )
        review_state = load_review_state()
        existing_entry_state = review_entry_state(review_state, duplicate_index)
        review_state["entries"][str(duplicate_index)] = with_provenance(
            existing_entry_state, node_ids_from_review_entry(existing_entry_state), thread_ts
        )
        save_review_state(review_state)
        return {
            "title": existing_entries[duplicate_index]["title"],
            "summary": existing_entries[duplicate_index]["summary"],
            "markdown_path": str(markdown_filepath),
            "table_name": TECHSUPPORT_QA_TABLE_NAME,
            "node_ids": node_ids_from_review_entry(existing_entry_state),
            "deduped_existing_entry_index": duplicate_index,
        }

    title, summary = prepare_title_and_summary_for_markdown(title, summary, existing_titles)

    # The new entry's ordinal is its position in the markdown file as it
    # exists right now -- NOT len(review_state["entries"]), which would be
    # wrong (and silently misaligned forever after) if any earlier entries in
    # the file predate this feature and were never recorded in review_state.
    entry_index = len(existing_titles)

    markdown_path = append_summary_to_markdown(title, summary)
    github_url = _compute_github_url_for_entry(entry_index, markdown_path)
    lancedb_result = insert_summary_into_lancedb(title, summary, github_url=github_url)

    review_state = load_review_state()
    review_state["entries"][str(entry_index)] = with_provenance(
        review_entry_state(review_state, entry_index), lancedb_result["node_ids"], thread_ts
    )
    save_review_state(review_state)

    return {
        "title": title,
        "summary": summary,
        "markdown_path": str(markdown_path),
        **lancedb_result,
    }


def find_entry_index_for_thread(thread_ts: str) -> Optional[int]:
    """Look up which (0-indexed) markdown entry / LanceDB row(s) an
    already-added thread corresponds to, via the "thread_ts" recorded in
    techsupport_review_state.json when it was added (see add_verified_qa_pair).
    Returns None if this thread_ts has no tracked entry -- e.g. it was added
    before thread_ts tracking existed, or wasn't added via this pipeline at
    all -- in which case there is no reliable way to find its entry."""
    review_state = load_review_state()
    for index_str, entry in review_state["entries"].items():
        if entry.get("thread_ts") == thread_ts:
            return int(index_str)
    return None


def replace_verified_qa_pair(thread_ts: str, conversation_thread: str) -> Dict[str, Any]:
    """Replace pipeline for an already-added thread that received new reply
    activity: regenerate the title+summary from the FULL updated conversation
    (`conversation_thread` -- original question plus all replies, including
    the new one, the same complete-thread text classify_thread() always
    produces) and replace the existing entry tied to this thread_ts in place,
    both in the markdown (same file position, not appended) and in LanceDB
    (old row(s) deleted, new row(s) inserted) -- the old entry is fully
    discarded in favor of the freshly regenerated one.

    (Function name kept as `replace_verified_qa_pair` for backward
    compatibility with nightly_pipeline.py's import; see module docstring.)

    Raises LookupError if this thread_ts has no tracked entry -- the caller
    must not guess which entry to overwrite in that case.
    """
    entry_index = find_entry_index_for_thread(thread_ts)
    if entry_index is None:
        raise LookupError(
            f"No tracked markdown entry for thread_ts={thread_ts!r} -- cannot replace "
            "(likely added before thread_ts tracking existed, or added outside this pipeline)"
        )

    generated = generate_verified_title_and_summary(conversation_thread)
    title, summary = generated["title"], generated["summary"]
    markdown_filepath = VERIFIED_TECHSUPPORT_QA_FOLDERPATH / TECHSUPPORT_QA_MARKDOWN_FILENAME
    existing_titles = [
        entry["title"]
        for i, entry in enumerate(parse_summary_markdown(markdown_filepath.read_text(encoding="utf-8")))
        if i != entry_index
    ]
    title, summary = prepare_title_and_summary_for_markdown(title, summary, existing_titles)

    review_state = load_review_state()
    existing_entry_state = review_entry_state(review_state, entry_index)
    old_node_ids = node_ids_from_review_entry(existing_entry_state)

    configure_embedding_model()
    env = load_pipeline_env()
    lancedb_folderpath = str(_resolve_path(env["LANCEDB_FOLDERPATH"]))
    db = lancedb.connect(lancedb_folderpath)
    vector_store = LanceDBVectorStore(
        connection=db, uri=lancedb_folderpath, table_name=TECHSUPPORT_QA_TABLE_NAME, query_type="hybrid"
    )

    markdown_path = replace_summary_in_markdown(entry_index, title, summary)
    github_url = _compute_github_url_for_entry(entry_index, markdown_path)
    lancedb_result = insert_summary_into_lancedb(title, summary, github_url=github_url)

    review_state["entries"][str(entry_index)] = with_provenance(
        existing_entry_state, lancedb_result["node_ids"], thread_ts
    )
    save_review_state(review_state)
    # Insert-then-delete: a crash in the gap may leave a brief duplicate row, not a hole
    # with nothing retrievable until retry. Skip delete when the state file has no
    # tracked node_ids (corrupt/partial write, or an entry predating this field).
    if old_node_ids:
        vector_store.delete_nodes(old_node_ids)

    return {
        "title": title,
        "summary": summary,
        "markdown_path": str(markdown_path),
        **lancedb_result,
    }


def _provenance_record(
    source_channel_id: Optional[str], source_thread_ts: Optional[str], timestamp_key: str
) -> Optional[Dict[str, Any]]:
    """Build one {source_channel_id, source_thread_ts, <timestamp_key>} record,
    or None when the caller supplied no source at all (the pre-provenance
    callers, whose review-state entries must keep their original shape)."""
    if source_channel_id is None and source_thread_ts is None:
        return None
    return {
        "source_channel_id": source_channel_id,
        "source_thread_ts": source_thread_ts,
        timestamp_key: datetime.now(timezone.utc).isoformat(),
    }


def enrich_verified_entry(
    existing_title: str,
    new_thread_conversation: str,
    *,
    source_channel_id: Optional[str] = None,
    source_thread_ts: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge/append pipeline for a NEW escalation thread that was tagged (via
    /tag_techsupport_thread/) as related to an already-existing verified entry --
    i.e. Lil Lisa answered by citing that entry, the user escalated anyway, and the
    resulting thread turned out useful+conclusive. Unlike replace_verified_qa_pair()
    (which fully regenerates an entry from its own updated thread), this locates the
    existing entry by TITLE -- the escalation thread is unrelated to whichever
    thread originally created that entry, so thread_ts can't be used to find it --
    and asks the LLM to merge the new insight into the existing summary rather than
    regenerating from scratch.

    The title is deliberately reused verbatim (never regenerated): the markdown
    heading and the GitHub anchor slug derived from it must stay stable so the
    existing GitHub link keeps working after the update.

    Raises LookupError if the markdown file doesn't exist or has no entry with this
    exact title -- the caller (nightly_pipeline.py) falls back to a normal add in
    that case rather than failing the whole thread.

    `source_channel_id` / `source_thread_ts` are optional provenance: when either
    is given, an {"source_channel_id", "source_thread_ts", "enriched_at"} record
    is appended to this entry's "enrichments" list in review_state. Callers that
    pass neither (the original nightly_pipeline.py call) leave the entry's shape
    exactly as before.
    """
    configure_dspy_lm()
    markdown_filepath = VERIFIED_TECHSUPPORT_QA_FOLDERPATH / TECHSUPPORT_QA_MARKDOWN_FILENAME
    if not markdown_filepath.exists():
        raise LookupError(f"No verified techsupport markdown file found -- cannot enrich {existing_title!r}")

    entries = parse_summary_markdown(markdown_filepath.read_text(encoding="utf-8"))
    entry_index = next((i for i, entry in enumerate(entries) if entry["title"] == existing_title), None)
    if entry_index is None:
        raise LookupError(f"No existing verified entry titled {existing_title!r} -- cannot enrich")

    existing_summary = entries[entry_index]["summary"]
    new_insight_summary = redact_obvious_pii(
        summarize_conversation(conversation_thread=new_thread_conversation).summary.strip()
    )
    merged_summary = merge_techsupport_summaries(
        existing_summary=existing_summary, new_insight=new_insight_summary
    ).merged_summary.strip()
    merged_summary = redact_obvious_pii(merged_summary)
    merged_summary = sanitize_techsupport_summary(merged_summary)

    review_state = load_review_state()
    existing_entry_state = review_entry_state(review_state, entry_index)
    old_node_ids = node_ids_from_review_entry(existing_entry_state)

    configure_embedding_model()
    env = load_pipeline_env()
    lancedb_folderpath = str(_resolve_path(env["LANCEDB_FOLDERPATH"]))
    db = lancedb.connect(lancedb_folderpath)
    vector_store = LanceDBVectorStore(
        connection=db, uri=lancedb_folderpath, table_name=TECHSUPPORT_QA_TABLE_NAME, query_type="hybrid"
    )

    # Title unchanged -- see docstring: this is what keeps the GitHub anchor stable.
    markdown_path = replace_summary_in_markdown(entry_index, existing_title, merged_summary)
    github_url = _compute_github_url_for_entry(entry_index, markdown_path)
    lancedb_result = insert_summary_into_lancedb(existing_title, merged_summary, github_url=github_url)

    # Preserve the entry's original thread_ts (the thread that first created it),
    # NOT the new escalation thread_ts -- replace_verified_qa_pair() still needs it
    # if the original thread later gets its own new replies.
    review_state["entries"][str(entry_index)] = with_provenance(
        existing_entry_state,
        lancedb_result["node_ids"],
        existing_entry_state.get("thread_ts"),
        enrichment=_provenance_record(source_channel_id, source_thread_ts, "enriched_at"),
    )
    save_review_state(review_state)
    if old_node_ids:
        vector_store.delete_nodes(old_node_ids)

    return {
        "title": existing_title,
        "summary": merged_summary,
        "markdown_path": str(markdown_path),
        **lancedb_result,
    }


def correct_verified_entry(
    existing_title: str,
    correction_thread: str,
    *,
    source_channel_id: str,
    source_thread_ts: str,
) -> Dict[str, Any]:
    """SUPERSEDE pipeline for an expert correction raised in a product channel:
    Lil Lisa answered a user by citing a verified techsupport entry, and an
    expert replied in that same thread to say the answer is wrong.

    Structurally the twin of enrich_verified_entry() -- same locate-by-title,
    same title-kept-verbatim (so the GitHub anchor stays stable), same
    replace-markdown / re-embed / delete-old-node-ids ordering -- but it runs
    CorrectTechsupportSummary instead of MergeTechsupportSummaries.
    MergeTechsupportSummaries is deliberately append-only ("preserve the
    existing content"), which is right for escalation enrichment and wrong
    here: the whole point of a correction is that the contradicted content
    must NOT survive. See the routing rule in the module docstring.

    `correction_thread` is the role-tagged thread text produced by
    techsupport_classifier.format_thread_messages() -- the user's question, the
    bot's (unverified) answer, and the expert's correction, each tagged so the
    prompt can tell which speaker is authoritative.

    Provenance ({"source_channel_id", "source_thread_ts", "superseded_at"}) is
    appended to this entry's "corrections" list in review_state. The entry's
    original "thread_ts" is preserved -- the correction happened in a different
    (product) channel thread, and replace_verified_qa_pair() still needs the
    original if that thread later gets new replies.

    Raises LookupError if the markdown file doesn't exist or has no entry with
    this exact title -- same contract as enrich_verified_entry(), so the caller
    can fall back to add_verified_qa_pair() rather than failing the thread.
    """
    configure_dspy_lm()
    markdown_filepath = VERIFIED_TECHSUPPORT_QA_FOLDERPATH / TECHSUPPORT_QA_MARKDOWN_FILENAME
    if not markdown_filepath.exists():
        raise LookupError(f"No verified techsupport markdown file found -- cannot correct {existing_title!r}")

    entries = parse_summary_markdown(markdown_filepath.read_text(encoding="utf-8"))
    entry_index = next((i for i, entry in enumerate(entries) if entry["title"] == existing_title), None)
    if entry_index is None:
        raise LookupError(f"No existing verified entry titled {existing_title!r} -- cannot correct")

    existing_summary = entries[entry_index]["summary"]
    corrected_summary = correct_summary(
        existing_summary=existing_summary, correction_thread=correction_thread
    ).corrected_summary.strip()
    corrected_summary = redact_obvious_pii(corrected_summary)
    corrected_summary = sanitize_techsupport_summary(corrected_summary)

    review_state = load_review_state()
    existing_entry_state = review_entry_state(review_state, entry_index)
    old_node_ids = node_ids_from_review_entry(existing_entry_state)

    configure_embedding_model()
    env = load_pipeline_env()
    lancedb_folderpath = str(_resolve_path(env["LANCEDB_FOLDERPATH"]))
    db = lancedb.connect(lancedb_folderpath)
    vector_store = LanceDBVectorStore(
        connection=db, uri=lancedb_folderpath, table_name=TECHSUPPORT_QA_TABLE_NAME, query_type="hybrid"
    )

    # Title unchanged -- see docstring: this is what keeps the GitHub anchor stable.
    markdown_path = replace_summary_in_markdown(entry_index, existing_title, corrected_summary)
    github_url = _compute_github_url_for_entry(entry_index, markdown_path)
    lancedb_result = insert_summary_into_lancedb(existing_title, corrected_summary, github_url=github_url)

    review_state["entries"][str(entry_index)] = with_provenance(
        existing_entry_state,
        lancedb_result["node_ids"],
        existing_entry_state.get("thread_ts"),
        correction=_provenance_record(source_channel_id, source_thread_ts, "superseded_at"),
    )
    save_review_state(review_state)
    # Insert-then-delete, same as enrich/replace: a crash in the gap leaves a
    # brief duplicate row rather than nothing retrievable.
    if old_node_ids:
        vector_store.delete_nodes(old_node_ids)

    return {
        "title": existing_title,
        "summary": corrected_summary,
        "markdown_path": str(markdown_path),
        **lancedb_result,
    }
