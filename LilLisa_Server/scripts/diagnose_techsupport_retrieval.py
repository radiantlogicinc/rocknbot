"""
diagnose_techsupport_retrieval.py
====================================
Read-only diagnostic for "the bot does not answer from verified techsupport
content". Runs INSIDE the backend pod and prints one report covering every
link in the chain: server process, config, LanceDB tables and embedding
dimensions, the verified markdown, a direct vector search of
TECHSUPPORT_QA_PAIRS, the pipeline health probe, and one real /invoke/ call.

Never prints secret values (tokens, keys) -- only whether they are set.
Writes nothing except one throwaway conversation session for the /invoke/
call (session id prefixed "diag-").

Usage (from a workstation; the script is piped over stdin, nothing to copy
into the image):
    kubectl exec -i <pod> -n <namespace> -- python3 - "<question>" IDA \
        < LilLisa_Server/scripts/diagnose_techsupport_retrieval.py

Arguments: the smoke-test question, then the product (IDA, IDDM or IDO;
default IDA). Optional third argument: server port (default 8000).
"""

import json
import os
import re
import sys
import time
from pathlib import Path

SERVER_ROOT = Path(os.environ.get("LILLISA_SERVER_ROOT", "/app"))
LILLISA_ENV_PATH = SERVER_ROOT / "env" / "lillisa_server.env"
SYNC_ENV_PATH = SERVER_ROOT / "cron" / "env" / "techsupport_sync.env"
GITHUB_ENV_PATH = SERVER_ROOT / "cron" / "env" / "github_push.env"
TECHSUPPORT_TABLE = "TECHSUPPORT_QA_PAIRS"
PRODUCT_TABLES = ("IDA", "IDA_QA_PAIRS", "IDDM", "IDDM_QA_PAIRS", "IDO", "IDO_QA_PAIRS", TECHSUPPORT_TABLE)
VOYAGE_DIM = 2048
OPENAI_DIM = 3072

QUESTION = sys.argv[1] if len(sys.argv) > 1 else ""
PRODUCT = (sys.argv[2] if len(sys.argv) > 2 else "IDA").upper()
PORT = sys.argv[3] if len(sys.argv) > 3 else "8000"

findings = []


def section(title):
    print(f"\n=== {title} ===")


def finding(text):
    findings.append(text)
    print(f"  !! {text}")


def read_env(path):
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}
    return {k: v for k, v in dotenv_values(str(path)).items() if v is not None} if path.exists() else {}


def resolve(raw):
    p = Path(raw)
    return p if p.is_absolute() else (SERVER_ROOT / p).resolve()


def vector_dim(table):
    field = table.schema.field("vector")
    return getattr(field.type, "list_size", None)


# --- 1. Server process ---
section("1. Server process")
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
    except OSError:
        continue
    if "uvicorn" in cmd or "gunicorn" in cmd:
        print(f"  pid {proc.name}: {cmd}")
        if "--workers" in cmd or "gunicorn" in cmd:
            finding("Server runs multiple workers: /reload_techsupport_qa_pairs/ only reloads ONE of them.")

# --- 2. Config (presence only) ---
section("2. Config")
lillisa = {**read_env(LILLISA_ENV_PATH), **os.environ}
sync = {**read_env(SYNC_ENV_PATH), **os.environ}
github = {**read_env(GITHUB_ENV_PATH), **os.environ}
print(f"  {LILLISA_ENV_PATH} exists: {LILLISA_ENV_PATH.exists()}")
print(f"  {SYNC_ENV_PATH} exists: {SYNC_ENV_PATH.exists()}")
print(f"  {GITHUB_ENV_PATH} exists: {GITHUB_ENV_PATH.exists()}")
for key in ("SLACK_BOT_TOKEN", "TECHSUPPORT_CHANNEL_ID"):
    print(f"  {key} set: {bool(sync.get(key))}")
for key in ("GITHUB_TOKEN", "GITHUB_REPO_URL"):
    print(f"  {key} set: {bool(github.get(key))}")
print(f"  AUTHENTICATION_KEY set: {bool(lillisa.get('AUTHENTICATION_KEY'))}")
for key in ("LANCEDB_FOLDERPATH", "TECHSUPPORT_STATE_DIR", "VERIFIED_TECHSUPPORT_QA_FOLDERPATH"):
    print(f"  {key} = {lillisa.get(key, '(unset)')}")

lancedb_path = resolve(lillisa.get("LANCEDB_FOLDERPATH") or "lancedb")
state_dir = resolve(lillisa["TECHSUPPORT_STATE_DIR"]) if lillisa.get("TECHSUPPORT_STATE_DIR") else lancedb_path.parent / "techsupport_state"
verified_dir = (
    resolve(lillisa["VERIFIED_TECHSUPPORT_QA_FOLDERPATH"])
    if lillisa.get("VERIFIED_TECHSUPPORT_QA_FOLDERPATH")
    else state_dir / "verified_techsupport"
)

voyage_key_path = lillisa.get("VOYAGE_API_KEY_FILEPATH")
voyage_key = os.environ.get("VOYAGE_API_KEY")
if not voyage_key and voyage_key_path and resolve(voyage_key_path).exists():
    voyage_key = resolve(voyage_key_path).read_text(encoding="utf-8").strip()
print(f"  Voyage API key available: {bool(voyage_key)}")

# --- 3. LanceDB tables ---
section(f"3. LanceDB tables at {lancedb_path}")
dims = {}
techsupport_rows = []
try:
    import lancedb

    db = lancedb.connect(str(lancedb_path))
    names = db.table_names()
    print(f"  tables: {names}")
    for name in PRODUCT_TABLES:
        if name not in names:
            print(f"  {name}: absent")
            continue
        table = db.open_table(name)
        dims[name] = vector_dim(table)
        print(f"  {name}: rows={table.count_rows()} vector_dim={dims[name]}")
    if TECHSUPPORT_TABLE in names:
        techsupport_rows = db.open_table(TECHSUPPORT_TABLE).to_arrow().select(["text", "metadata"]).to_pylist()
    else:
        finding(f"{TECHSUPPORT_TABLE} does not exist on disk.")
except Exception as exc:  # noqa: BLE001
    finding(f"Could not read LanceDB: {type(exc).__name__}: {exc}")
    db = None

docs_table = PRODUCT if PRODUCT in dims else next((t for t in ("IDA", "IDDM", "IDO") if t in dims), None)
docs_dim = dims.get(docs_table)
ts_dim = dims.get(TECHSUPPORT_TABLE)
if docs_dim and ts_dim and docs_dim != ts_dim:
    model = "OpenAI text-embedding-3-large" if docs_dim == OPENAI_DIM else f"a {docs_dim}-dim model"
    finding(
        f"EMBEDDING MISMATCH: docs table {docs_table} is {docs_dim}-dim ({model}) but {TECHSUPPORT_TABLE} is "
        f"{ts_dim}-dim (Voyage). The server embeds each question once with the docs model and reuses it for "
        f"the techsupport search, which then fails and returns nothing."
    )

if techsupport_rows:
    metas = [r.get("metadata") or {} for r in techsupport_rows]
    with_url = sum(1 for m in metas if m.get("github_url"))
    sources = sorted({str(m.get("source")) for m in metas})
    print(f"  {TECHSUPPORT_TABLE}: {with_url}/{len(metas)} rows have github_url; metadata.source values: {sources}")
    for m in metas[:3]:
        print(f"    sample title: {m.get('title')!r}")
    if with_url == 0:
        finding("No techsupport row has github_url; answers cannot show techsupport links.")

# --- 4. Verified markdown ---
section("4. Verified markdown")
markdown_path = verified_dir / "techsupport_qa_pairs.md"
print(f"  {markdown_path} exists: {markdown_path.exists()}")
entries = []
if markdown_path.exists():
    text = markdown_path.read_text(encoding="utf-8")
    entries = [e for e in re.split(r"(?m)^## ", text)[1:] if e.strip()]
    print(f"  entries: {len(entries)}")
    if ts_dim and techsupport_rows and len(entries) != len(techsupport_rows):
        finding(f"Markdown has {len(entries)} entries but {TECHSUPPORT_TABLE} has {len(techsupport_rows)} rows.")
print(f"  state dir {state_dir} exists: {state_dir.exists()}")

if QUESTION and entries:
    words = {w for w in re.findall(r"[a-z0-9]{4,}", QUESTION.lower())}
    scored = sorted(
        ((len(words & set(re.findall(r"[a-z0-9]{4,}", e.lower()))), e.splitlines()[0]) for e in entries),
        reverse=True,
    )[:5]
    print("  entries sharing the most words with the question:")
    for score, title in scored:
        print(f"    {score:3d} word(s): {title[:120]}")
    if not scored or scored[0][0] == 0:
        finding("No markdown entry shares words with the question; the expected answer may not be in the 680 entries.")

# --- 5. Direct vector search of TECHSUPPORT_QA_PAIRS ---
section("5. Direct vector search of TECHSUPPORT_QA_PAIRS")
if not QUESTION:
    print("  skipped: no question given")
elif db is None or TECHSUPPORT_TABLE not in dims:
    print("  skipped: table unavailable")
elif not voyage_key:
    print("  skipped: no Voyage API key")
else:
    try:
        import voyageai

        client = voyageai.Client(api_key=voyage_key)
        qvec = client.contextualized_embed(
            inputs=[[QUESTION]], model="voyage-context-3", input_type="query", output_dimension=ts_dim
        ).results[0].embeddings[0]
        hits = db.open_table(TECHSUPPORT_TABLE).search(qvec, vector_column_name="vector").limit(5).to_list()
        for hit in hits:
            meta = hit.get("metadata") or {}
            print(f"    distance={hit.get('_distance'):.4f}  {meta.get('title')!r}")
        print("  If the expected entry is listed here, storage and embeddings are fine and the problem is in the server.")
    except Exception as exc:  # noqa: BLE001
        finding(f"Direct techsupport search failed: {type(exc).__name__}: {exc}")

# --- 6. Pipeline health probe ---
section("6. GET /health/techsupport_pipeline")
try:
    import requests

    resp = requests.get(f"http://127.0.0.1:{PORT}/health/techsupport_pipeline", timeout=30)
    print(f"  HTTP {resp.status_code}: {resp.text[:1500]}")
except Exception as exc:  # noqa: BLE001
    finding(f"Health probe failed: {type(exc).__name__}: {exc}")

# --- 7. End-to-end /invoke/ ---
section(f"7. POST /invoke/ (product={PRODUCT})")
if not QUESTION:
    print("  skipped: no question given")
else:
    try:
        import requests

        params = {
            "session_id": f"diag-{int(time.time())}",
            "locale": "en",
            "product": PRODUCT,
            "nl_query": QUESTION,
            "is_expert_answering": "false",
        }
        resp = requests.post(f"http://127.0.0.1:{PORT}/invoke/", params=params, timeout=300)
        print(f"  HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        body = resp.json()
        nodes = body.get("reranked_nodes") or []
        ts_nodes = [
            n for n in nodes
            if (n.get("metadata") or {}).get("source") == "techsupport"
            or ((n.get("metadata") or {}).get("github_url") and not (n.get("metadata") or {}).get("webportal_url"))
        ]
        print(f"  answer_found={body.get('answer_found')} reranked_nodes={len(nodes)} techsupport_nodes={len(ts_nodes)}")
        print(f"  primary_techsupport_match_title={body.get('primary_techsupport_match_title')!r}")
        for n in ts_nodes[:5]:
            print(f"    techsupport node: {(n.get('metadata') or {}).get('title')!r}")
        print(f"  response (first 600 chars): {str(body.get('response'))[:600]!r}")
        print(f"  links_text: {body.get('links_text')!r}")
        if not ts_nodes:
            finding("The live server returned NO techsupport nodes for this question.")
    except Exception as exc:  # noqa: BLE001
        finding(f"/invoke/ failed: {type(exc).__name__}: {exc}")

# --- Summary ---
section("SUMMARY")
if findings:
    for f in findings:
        print(f"  - {f}")
else:
    print("  No problems detected. Techsupport content reached the live answer for this question.")
print(json.dumps({"findings": findings}))
