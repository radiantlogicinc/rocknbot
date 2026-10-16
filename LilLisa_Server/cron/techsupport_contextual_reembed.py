"""
techsupport_contextual_reembed.py
====================================
Periodic full re-embed of the shared TECHSUPPORT_QA_PAIRS table using Voyage's
contextual embedding (voyage-context-3, input_type="document"), so every
entry's embedding is computed with the full techsupport_qa_pairs.md file as
context -- not just whatever existed in the file when that entry was
individually added by techsupport_qa_ingest.add_verified_qa_pair() (Step 6,
scripts/techsupport_qa_ingest.py).

This is a SEPARATE, less-frequent job from Step 6's incremental add. Step 6
keeps working unchanged for immediate nightly additions (one row, cheap, no
cross-document context); this script periodically rebuilds the WHOLE
TECHSUPPORT_QA_PAIRS table from scratch so every row benefits from
full-document context, following the same get_contextualized_embeddings
pattern src/main.py's _run_rebuild_docs_task_contextual already uses for
documentation:
  - each unit of text (there: a whole doc file; here: one entry's
    "## title\n\nsummary" block) becomes a "chunk" of a single logical document
  - VoyageEmbedding.get_contextualized_embeddings() is called once per batch
    of chunks (batched to stay under Voyage's ~32K token limit, using the same
    conservative 20K-tiktoken-token budget src/main.py uses), so chunks in the
    same batch are embedded with awareness of each other
  - the new table is built under a "_new" name and, once it's fully populated
    and row-count-verified, its data is written into the live table via
    create_table(mode="overwrite") -- never drop_table() on the live table --
    so a failure partway through never leaves the live table half-written,
    AND the live table's LanceDB version history (list_versions() /
    checkout() / restore()) is preserved across every re-embed instead of
    being wiped, which is what techsupport_rollback.py relies on

Does NOT touch IDA_QA_PAIRS / IDDM_QA_PAIRS (golden QA pairs) or the
documentation tables -- only TECHSUPPORT_QA_PAIRS.

How often this actually re-embeds is controlled by
TECHSUPPORT_REEMBED_INTERVAL_DAYS (env/lillisa_server.env, default 7 days) --
run_if_due() checks techsupport_reembed_state.json's last_reembed_timestamp
and no-ops if the interval hasn't elapsed, so calling this on every
nightly_pipeline.py run is cheap on the nights it's not due, and there's no
need for a separate cron entry.

A state file that is MISSING (or exists with no last_reembed_timestamp) is
deliberately also a no-op, NOT "due now": is_reembed_due() only returns True
when a prior successful re-embed is already on record and the interval has
elapsed since then. The alternative -- treating "no record" as "due" --
means losing or resetting this one small JSON file (e.g. a state-folder path
change) silently triggers rebuild_table()'s unconditional overwrite of the
live TECHSUPPORT_QA_PAIRS table from whatever techsupport_qa_pairs.md
happens to contain at that moment, without ever having confirmed that
markdown is actually the complete corpus. Run with --force (standalone) or
run_if_due(force=True) (as a library) to run deliberately regardless of
state -- the right call right after a fresh deployment's bootstrap step
populates TECHSUPPORT_QA_PAIRS from a markdown file known to be correct (see
techsupport_bootstrap.py), or whenever an operator has verified the markdown
by hand and wants to force a re-embed now.

Unlike techsupport_qa_ingest.py, this module deliberately does NOT import
that module (or techsupport_classifier.py) -- both trigger a module-level
configure_dspy_lm(), an LLM dependency this script (which only re-embeds
existing rows; it extracts nothing new) doesn't need. The few path/table-name
constants below are duplicated rather than imported for that reason. It DOES
import github_anchor.py (see build_contextual_nodes) -- that module has no
LLM/embedding dependency of its own, so it doesn't reintroduce the thing
being avoided here.

Every full re-embed also recomputes every entry's node.metadata["github_url"]
from scratch (build_contextual_nodes, via
github_anchor.compute_github_urls_for_titles) -- since this pass already has
every title in the file in the correct order, it's the one place where
duplicate-heading slug numbering is guaranteed fully correct and consistent
across the whole table in one shot, self-healing any drift the incremental
add/replace path in techsupport_qa_ingest.py could theoretically introduce
(see its _compute_github_url_for_entry() docstring for that edge case).

Required env vars (read from ../env/lillisa_server.env, same file the main
server reads):
    LANCEDB_FOLDERPATH               - path (relative to LilLisa_Server project root) to the LanceDB folder
    VOYAGE_API_KEY_FILEPATH          - path (relative to LilLisa_Server project root) to a file
                                         containing the Voyage AI API key
    TECHSUPPORT_REEMBED_INTERVAL_DAYS - optional, defaults to 7

Usage (as a library):
    from techsupport_contextual_reembed import run_if_due
    result = run_if_due()           # interval-gated, see above
    result = run_if_due(force=True) # run now regardless of state

Usage (standalone):
    python techsupport_contextual_reembed.py
    python techsupport_contextual_reembed.py --force
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tiktoken
from dotenv import dotenv_values
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
from paths import (
    LILLISA_SERVER_ENV_PATH,
    LILLISA_SERVER_ROOT,
    PACKAGE_ROOT,
    REEMBED_STATE_PATH,
    VERIFIED_TECHSUPPORT_QA_FOLDERPATH,
    ensure_import_paths,
)

import lancedb

ensure_import_paths()
SCRIPT_DIR = PACKAGE_ROOT
PROJECT_ROOT = LILLISA_SERVER_ROOT
STATE_PATH = REEMBED_STATE_PATH

from github_anchor import compute_github_urls_for_titles  # noqa: E402

from atomic_io import atomic_write_json  # noqa: E402

from src.embedding_config import (  # noqa: E402
    VOYAGE_EMBEDDING_DIMENSION,
    VoyageEmbedding,
)
from src.llama_index_lancedb_vector_store import LanceDBVectorStore  # noqa: E402

# No basicConfig() call -- library module, see techsupport_qa_ingest.py's
# same comment on its own module-level logger.
logger = logging.getLogger(__name__)

# Same single shared store techsupport_qa_ingest.py (Step 6) writes to -- see
# module docstring for why these are duplicated here instead of imported.
# VERIFIED_TECHSUPPORT_QA_FOLDERPATH itself comes from paths.py.
TECHSUPPORT_QA_MARKDOWN_FILENAME = "techsupport_qa_pairs.md"
TECHSUPPORT_QA_TABLE_NAME = "TECHSUPPORT_QA_PAIRS"

DEFAULT_REEMBED_INTERVAL_DAYS = 7
# Conservative tiktoken-based batching budget -- same constant family
# _run_rebuild_docs_task_contextual (src/main.py) uses for its cross-document
# batching fallback, since tiktoken (GPT-3.5 tokenizer) underestimates
# Voyage's actual token counts relative to its ~32K context-embedding limit.
BATCH_TOKEN_BUDGET = 20000

# Splits on any line starting with "## " (the title heading marker) -- same
# pattern techsupport_qa_ingest.parse_summary_markdown() uses.
TITLE_BLOCK_SPLIT_PATTERN = re.compile(r"(?m)^## ")
TITLE_SUMMARY_PATTERN = re.compile(r"(.+?)\n\n(.*)", re.DOTALL)


def load_pipeline_env() -> Dict[str, str]:
    """Load LANCEDB_FOLDERPATH / VOYAGE_API_KEY_FILEPATH the same way
    techsupport_qa_ingest.load_pipeline_env() loads its own required vars."""
    env = dict(dotenv_values(str(LILLISA_SERVER_ENV_PATH)))
    env = {**env, **os.environ}
    for key in ("LANCEDB_FOLDERPATH", "VOYAGE_API_KEY_FILEPATH"):
        if not env.get(key):
            raise RuntimeError(f"{key} not found in {LILLISA_SERVER_ENV_PATH}")
    return env


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def get_reembed_interval_days() -> float:
    """Expert-adjustable knob: how many days must elapse between full
    contextual re-embeds. Overridable per-process via env var without editing
    the env file."""
    env = dict(dotenv_values(str(LILLISA_SERVER_ENV_PATH)))
    env = {**env, **os.environ}
    raw = env.get("TECHSUPPORT_REEMBED_INTERVAL_DAYS", str(DEFAULT_REEMBED_INTERVAL_DAYS))
    return float(raw)


_embedding_configured = False


def configure_embedding_model() -> None:
    """Point Settings.embed_model at the same Voyage model the golden QA
    pairs / techsupport tables use. Nodes below are pre-embedded manually
    (see build_contextual_nodes), so this only needs to exist to satisfy
    VectorStoreIndex's embed_model requirement -- it's never actually called
    to embed anything."""
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


# --- State tracking (mirrors nightly_techsupport_sync.py's state file) ---


def load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"last_reembed_timestamp": None}
    with open(STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict[str, Any]) -> None:
    atomic_write_json(STATE_PATH, state, indent=2, sort_keys=True)


def is_reembed_due(state: Dict[str, Any]) -> bool:
    """True only when a PRIOR successful re-embed is already on record (a
    "last_reembed_timestamp" written by an earlier run_if_due()/run_reembed()
    call) AND TECHSUPPORT_REEMBED_INTERVAL_DAYS have elapsed since then.

    Deliberately NOT true just because the timestamp is missing -- see the
    module docstring's "A state file that is MISSING..." section. A fresh
    deployment's bootstrap step (techsupport_bootstrap.py) or an operator who
    has verified the markdown by hand should call run_if_due(force=True) (or
    run_reembed() directly) instead of relying on a missing state file to
    trigger this automatically.
    """
    last = state.get("last_reembed_timestamp")
    if last is None:
        return False
    interval_seconds = get_reembed_interval_days() * 86400
    return (time.time() - float(last)) >= interval_seconds


# --- Markdown parsing ---


def parse_qa_pairs(markdown_text: str) -> List[Tuple[str, str]]:
    """Parse the same "## {title}" blocks
    techsupport_qa_ingest.append_summary_to_markdown() writes (and
    techsupport_qa_ingest.parse_summary_markdown() parses) -- returns an
    ordered list of (title, summary) tuples. Name kept as `parse_qa_pairs` /
    return shape kept as tuples for minimal-diff compatibility with the rest
    of this module; the tuple elements are now (title, summary), not
    (question, answer)."""
    pairs = []
    for raw_block in TITLE_BLOCK_SPLIT_PATTERN.split(markdown_text):
        block = raw_block.strip()
        if not block:
            continue
        match = TITLE_SUMMARY_PATTERN.match(block)
        if not match:
            continue
        title, summary = match[1].strip(), match[2].strip()
        if title and summary:
            pairs.append((title, summary))
    return pairs


def _batch_by_token_budget(
    qa_pairs: List[Tuple[str, str]], budget: int = BATCH_TOKEN_BUDGET
) -> List[List[Tuple[str, str]]]:
    """Group (title, summary) entries into batches whose combined
    "## title\\n\\nsummary" token count stays under `budget`. Each batch is
    sent to Voyage as a single logical document (one API call, chunks = full
    entry texts), so entries within a batch are embedded with full context of
    each other. For realistically-sized files this returns exactly one batch
    containing every entry."""
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    batches: List[List[Tuple[str, str]]] = []
    current: List[Tuple[str, str]] = []
    current_tokens = 0
    for title, summary in qa_pairs:
        token_count = len(enc.encode(f"## {title}\n\n{summary}"))
        if current and current_tokens + token_count > budget:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append((title, summary))
        current_tokens += token_count
    if current:
        batches.append(current)
    return batches


def build_contextual_nodes(qa_pairs: List[Tuple[str, str]]) -> List[TextNode]:
    """Contextually embed every entry's full "## title\\n\\nsummary" text,
    following the exact get_contextualized_embeddings pattern
    _run_rebuild_docs_task_contextual (src/main.py) uses: each batch of entry
    texts is passed as the chunks of one logical document
    (input_type="document"), so every chunk's embedding is computed with
    awareness of the others in that batch. title/source (and, when
    computable, github_url) stay as metadata, excluded from embedding --
    same schema techsupport_qa_ingest.insert_summary_into_lancedb() already
    uses, so retrieval code elsewhere needs no changes.

    github_url is computed ONCE up front from every title in `qa_pairs`, in
    order (not per-batch -- duplicate-heading slug numbering depends on the
    whole file's title sequence, not just one batch's), then indexed
    alongside each node as it's built. Best-effort: if GITHUB_REPO_URL isn't
    configured, every node just gets no github_url rather than failing the
    whole re-embed -- a missing link is not worth blocking a real vector
    rebuild over.
    """
    try:
        github_urls: List[Optional[str]] = compute_github_urls_for_titles([title for title, _summary in qa_pairs])
    except Exception:  # noqa: BLE001 -- a broken GitHub link must not break re-embedding
        logger.warning("Could not compute github_url metadata for this re-embed pass", exc_info=True)
        github_urls = [None] * len(qa_pairs)

    nodes: List[TextNode] = []
    entry_index = 0
    for batch in _batch_by_token_budget(qa_pairs):
        batch_texts = [f"## {title}\n\n{summary}" for title, summary in batch]
        embeddings = VoyageEmbedding.get_contextualized_embeddings(
            documents_chunks=[batch_texts],
            model="voyage-context-3",
            output_dimension=VOYAGE_EMBEDDING_DIMENSION,
        )[0]
        for (title, _summary), text, embedding in zip(batch, batch_texts, embeddings):
            github_url = github_urls[entry_index]
            metadata = {"title": title, "source": "techsupport"}
            excluded_keys = ["title", "source"]
            if github_url:
                metadata["github_url"] = github_url
                excluded_keys.append("github_url")

            node = TextNode(
                text=text,
                metadata=metadata,
                embedding=embedding,
            )
            # Without a SOURCE relationship, node.ref_doc_id is None for every
            # node -- LanceDBVectorStore.add() writes that straight into the
            # table's "doc_id" column (see doc_id_key in
            # src/llama_index_lancedb_vector_store.py), so a table built
            # entirely from nodes like that gets its "doc_id" column typed as
            # Arrow's `null` type (every value is None). That silently breaks
            # every future single-row append via
            # techsupport_qa_ingest.insert_summary_into_lancedb() (which DOES
            # set a real string doc_id via its Document/VectorStoreIndex path)
            # with a "cast from string to null" error. Each contextual node
            # doesn't correspond to a real parent Document the way the
            # incremental-add path's nodes do, so it points to itself --
            # giving the column a real (if not semantically meaningful)
            # string type, matching the incremental path's schema.
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=node.node_id)
            node.excluded_embed_metadata_keys = excluded_keys
            node.excluded_llm_metadata_keys = excluded_keys
            entry_index += 1
            nodes.append(node)
    return nodes


# --- Table rebuild (overwrite-in-place, ONLY TECHSUPPORT_QA_PAIRS) ---


def rebuild_table(nodes: List[TextNode], lancedb_folderpath: str) -> Dict[str, Any]:
    """Build the new contextual embeddings into a "_new" staging table, verify
    its row count, then write that data into the live TECHSUPPORT_QA_PAIRS
    table via create_table(mode="overwrite").

    Deliberately NOT drop_table() + rename on the live table name: LanceDB
    keeps a version history per table, and create_table(mode="overwrite") on
    a table that already exists appends a new version rather than wiping
    prior ones (confirmed empirically -- list_versions()/checkout()/restore()
    all still work across it, including across a schema/dimension change).
    drop_table() on the live table, by contrast, deletes its on-disk
    directory -- including _versions/ -- destroying all history. The "_new"
    staging table's own history doesn't matter (it's discarded every run), so
    it's still safe to drop_table() that one to guarantee a clean build.

    A row-count mismatch raises before the live table is ever touched, so a
    failure partway through never leaves it half-written."""
    db = lancedb.connect(lancedb_folderpath)
    table_name = TECHSUPPORT_QA_TABLE_NAME
    new_table_name = f"{table_name}_new"

    if new_table_name in db.table_names():
        db.drop_table(new_table_name)

    vector_store = LanceDBVectorStore(
        connection=db, uri=lancedb_folderpath, table_name=new_table_name, query_type="hybrid"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    _ = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

    new_table = db.open_table(new_table_name)
    new_row_count = new_table.count_rows()
    if new_row_count != len(nodes):
        raise RuntimeError(
            f"Table '{new_table_name}' row count ({new_row_count}) does not match node count ({len(nodes)})."
        )

    new_data = new_table.to_arrow()
    if table_name in db.table_names():
        db.create_table(table_name, data=new_data, mode="overwrite")
    else:
        db.create_table(table_name, data=new_data)

    db.drop_table(new_table_name)

    table = db.open_table(table_name)
    return {"table_name": table_name, "row_count": table.count_rows()}


def run_reembed() -> Dict[str, Any]:
    """Unconditionally re-embed the whole TECHSUPPORT_QA_PAIRS table from the
    current techsupport_qa_pairs.md, regardless of the configured interval.
    Use run_if_due() for the interval-gated entry point nightly_pipeline.py
    calls."""
    configure_embedding_model()
    env = load_pipeline_env()
    lancedb_folderpath = str(_resolve_path(env["LANCEDB_FOLDERPATH"]))

    markdown_path = VERIFIED_TECHSUPPORT_QA_FOLDERPATH / TECHSUPPORT_QA_MARKDOWN_FILENAME
    if not markdown_path.exists():
        raise FileNotFoundError(f"Techsupport QA markdown file not found: {markdown_path}")
    markdown_text = markdown_path.read_text(encoding="utf-8")

    qa_pairs = parse_qa_pairs(markdown_text)
    if not qa_pairs:
        raise RuntimeError(
            f"No techsupport entries found in {markdown_path}; refusing to drop the live table with nothing to replace it."
        )

    nodes = build_contextual_nodes(qa_pairs)
    result = rebuild_table(nodes, lancedb_folderpath)
    result["qa_pair_count"] = len(qa_pairs)
    return result


def run_if_due(force: bool = False) -> Dict[str, Any]:
    """Interval-gated entry point: re-embeds only if a prior re-embed is on
    record AND TECHSUPPORT_REEMBED_INTERVAL_DAYS have elapsed since then
    (see is_reembed_due()), else no-ops. Safe to call on every
    nightly_pipeline.py run -- no separate cron entry needed.

    A missing/empty state file is one of the no-op reasons, NOT a trigger --
    see is_reembed_due() and the module docstring. Pass force=True to run
    unconditionally regardless of state; the run is still recorded in state
    afterwards either way, so a forced run also starts the normal interval
    clock for future automatic calls."""
    state = load_state()
    if not force and not is_reembed_due(state):
        reason = "no_prior_reembed_on_record" if state.get("last_reembed_timestamp") is None else "interval_not_elapsed"
        return {
            "ran": False,
            "reason": reason,
            "last_reembed_timestamp": state.get("last_reembed_timestamp"),
        }

    result = run_reembed()
    state["last_reembed_timestamp"] = f"{time.time():.6f}"
    save_state(state)
    return {"ran": True, "forced": force, **result}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the periodic full contextual re-embed of TECHSUPPORT_QA_PAIRS."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Run even if techsupport_reembed_state.json has no prior re-embed on record, or "
            "TECHSUPPORT_REEMBED_INTERVAL_DAYS hasn't elapsed. Required the first time this runs "
            "against a freshly-populated table (e.g. right after a deployment bootstrap), since a "
            "missing/empty state file is deliberately NOT auto-triggered -- see is_reembed_due()."
        ),
    )
    args = parser.parse_args()
    print(run_if_due(force=args.force))
