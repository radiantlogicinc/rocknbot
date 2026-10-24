# Aggregate code review — PR submitter

**Source:** [radiantlogicinc/rocknbot#42](https://github.com/radiantlogicinc/rocknbot/pull/42)

**Beads tracking:** findings are tracked under prefix `pr42` (section epics: `pr42-blockers`, `pr42-hp`, `pr42-mp`, `pr42-lp`, `pr42-nits`). Follow-on work that is not a merge blocker lives under `pr42-enhancements`. Each heading and item below is annotated with its beads ID.

**Verified against:** the aggregate tree diff `17de46f` → `66733fe` only (`git diff 17de46f 66733fe`). Intermediate commits were not used to keep, drop, or map findings.

This pass removed items that only existed in an intermediate commit (or were already fixed by `66733fe`), merged duplicates that described the same final-tree behavior, and dropped nits that were about commit splitting/messages rather than the final code.

**Overall:** Request changes. Makefile/gitignore/env-example work can land; server APIs, Slack escalation, and the nightly auto-publish path need the blockers below before merge.

---

## 1. Blockers (`pr42-blockers`)

Must fix before merge.

### FEATURE (`pr42-blockers.1`) — **closed** (3/3 children closed)

**Unvalidated classifier writes straight into live retrieval (and GitHub)** (`pr42-blockers.1.1`)  
`techsupport_classifier.py` documents that `is_useful` / `is_conclusive` are zero-shot, unoptimized, with no labeled validation. Ingest has no review gate: entries are queryable the same night and pushed to GitHub. A “yes” on an unresolved or social thread (or a bad summary) becomes an “expert verified” chunk in IDA/IDDM/IDO answers. Hold auto-ingest behind a queue, human approve, or a validated classifier.

- **Done:** Demoted from merge blocker. Ops mitigations accepted: thumbs-down on bot answers, and a reviewer can edit the techsupport QA markdown on GitHub when an error surfaces. No ingest-pipeline code change.
- **Beads:** `pr42-blockers.1.1` **closed**. Follow-on epic `pr42-enhancements.1` (**open**, P3, `discovered-from` this task): labeled useful/conclusive dataset, stronger model, F1 on both axes. Cheaper first step recorded on that epic: Slack resolution reaction, or stop treating auto-ingest as expert-verified until human sign-off.

`**[[NO_ANSWER]]` retry can turn a correct “I don’t know” into a wrong answer** (`pr42-blockers.1.2`)  
If the model emits `[[NO_ANSWER]]` and the top rerank score is `> 3.0`, the identical call is retried and the retry **replaces** the first response even when the first was honest. High cross-encoder score means “this chunk looks similar,” not “the answer is in context.” A nearby but insufficient techsupport/doc hit will retry, often produce a fluent answer without the marker, set `answer_found=True`, and hide the prominent escalate button. Accept a retry only with stronger evidence (or require the marker to stay absent **and** another check).

- **Done:** Not treated as a merge blocker. PR retry behavior is **kept and served**: same-prompt retry still replaces try-1 (including when try-2 is also a no-answer). Added an INFO `NO_ANSWER_RETRY | <json>` record per retry (query, both raw completions, top score, top chunk text+metadata, `changed_outcome`) so we can measure how often try-1 no-answer flips to a served answer. Marker detection on both tries uses the 1.3 startswith helper, not `find()`.
- **Beads:** `pr42-blockers.1.2` **closed**. Follow-on task `pr42-enhancements.2` (**open**, P3, `discovered-from` this task): extract those log lines, judge whether served try-2 is grounded in the high-scoring chunk, report flip rate and grounded-among-flips, then decide keep / drop / change the retry. Parent epic `pr42-enhancements` is **open**.

`**[[NO_ANSWER]]` detected with `find()`, not “starts the response”** (`pr42-blockers.1.3`)  
The prompt says the marker must be first. The code treats **any** occurrence as no-answer, then slices from there. A mention in context, a quoted example, or a mid-response “I would use `[[NO_ANSWER]]` if…” would flip `answer_found` and drop everything before the match. Use `llm_response.lstrip().startswith(marker)` (and ignore the marker inside fenced code).

- **Done:** Implemented. `parse_leading_no_answer_marker` in `LilLisa_Server/src/utils.py` treats no-answer only if the marker starts the response after `lstrip()`. Mid-body / fenced mentions are left in the answer. Used for try-1 and try-2. Unit tests: `LilLisa_Server/tests/test_no_answer_marker.py`. Version-sentence-before-marker is `pr42-mp.1.2` (prompt rule 11 + rule 7 exemption; not a reason to keep `find()`).
- **Beads:** `pr42-blockers.1.3` **closed**. Version-sentence-before-marker (`pr42-mp.1.2`) is also **closed** (rule 11 + rule 7 exemption).

### SECURITY (`pr42-blockers.2`) — **open** epic (2/3 children closed)

**Unauthenticated write: `/tag_techsupport_thread/`** (`pr42-blockers.2.1`)  
Anyone who can reach the server can POST `thread_ts` + `related_entry_title` and rewrite `scripts/techsupport_thread_tags.json`. That file drives merge/enrich in the nightly pipeline, so an attacker can attach an escalation to the wrong verified entry (or spam the file). `reload_techsupport_qa_pairs` requires JWT; this endpoint should too (or the same `encrypted_key`).

- **Done:** Same `encrypted_key` / `AUTHENTICATION_KEY` JWT as `/reload_techsupport_qa_pairs/`. `POST /tag_techsupport_thread/` calls `_require_jwt`; bad signature → 401. Slack `tag_techsupport_thread` sends `ENCRYPTED_AUTHENTICATION_KEY` in query params (already minted at bot startup for admin routes).
- **Beads:** `pr42-blockers.2.1` **closed**.

**Unauthenticated read: `/get_conversation_history/` and unauthenticated LLM: `/refine_escalation_query/`** (`pr42-blockers.2.2`)  
Full User/Assistant transcript by `session_id` (Slack uses `conv_id` as `session_id`). If those IDs are guessable or leak in logs/URLs, this is a conversation dump. Refine runs an LLM on arbitrary body text (cost/DoS). `/invoke/` is already unauthenticated (existing bot-network assumption); these new endpoints are more sensitive. Gate with JWT or a shared internal token; cap refine body size.

- **Done:** Same JWT on `GET /get_conversation_history/` and `POST /refine_escalation_query/`. Slack bot passes `encrypted_key` on both. Refine rejects `conversation_history` longer than `REFINE_ESCALATION_MAX_CHARS` (32768) with 413 before the LLM call. No second internal token.
- **Beads:** `pr42-blockers.2.2` **closed**.

**PII / customer data in the bot and in GitHub** (`pr42-blockers.2.3`)  
Ingest comments claim no PII stripping and that content “stays Slack-only.” Summaries are embedded for product Slack Q&A **and** pushed to a GitHub repo. Threads are formatted with display names. The summarize prompt asks not to mention usernames; it does not strip emails, hostnames, ticket IDs, or customer tenant details. Add a redaction pass (or refuse to ingest messages that look like secrets/PII) before markdown/LanceDB/GitHub.

- **Partial:** Summarize / merge / title prompts now tell the model to omit emails, hostnames, IPs, ticket IDs, tenant/customer names, and secrets. `redact_obvious_pii()` in `LilLisa_Server/scripts/techsupport_pii.py` runs on generated title/summary before markdown/LanceDB/GitHub (add, replace, enrich). Covers emails, Slack mentions, AKIA keys, PEM private keys, `password=`/`secret=`/`token=`/`api_key=` assignments, `INC`/`SR`/`HD`/`TICKET` ids, `*.local`/`*.internal`/`*.lan`/`*.corp`/`*.intranet` FQDNs, and RFC1918/loopback/link-local IPv4. Product versions like `7.3.1.0` are not treated as IPs. Tests: `LilLisa_Server/tests/test_techsupport_pii_redact.py`.
- **Still open:** Tune remaining patterns against a real `techsupport_qa_pairs.md` (customer/tenant names, bare hostnames, public IPv4/IPv6, org-specific Jira keys, and anything the current filters miss). No sample doc in this tree.
- **Beads:** `pr42-blockers.2.3` **open** (P0). Parent epic `pr42-blockers.2` remains **open** until this child closes.

### PERFORMANCE (`pr42-blockers.3`) — **closed** (1/1 children closed)

**Nightly sync is O(all known threads)** (`pr42-blockers.3.1`)  
`sync()` calls `conversations.replies` for **every previously seen thread**, throttled at 0.3s. A multi-year channel is thousands of API calls per run, plus rate limits and `channel_not_found` retries. The ~4-year bootstrap makes this worse. Need a cheaper “updated threads” strategy (e.g. store `latest_reply` from history where possible, or cap/age out dead threads). Do not ship “check every historical thread every night” as the long-term design.

- **Done:** `sync()` no longer calls `conversations.replies`. New threads still take `latest_reply` from `conversations.history`. Known threads get a parent-only `history` lookup (`inclusive`, `limit=1`). Nightly hot window: 30 days (`TECHSUPPORT_SYNC_HOT_DAYS`). Periodic catch-up every `TECHSUPPORT_SYNC_CATCHUP_INTERVAL_DAYS` (default 7) covers threads within 90 days (`TECHSUPPORT_SYNC_CATCHUP_AGE_DAYS`) that were not in the hot set. Each set is capped at `TECHSUPPORT_SYNC_MAX_PARENT_LOOKUPS` (200), hottest first. Threads quieter than 90 days stay in `techsupport_sync_state.json` (including `added_to_verified_db`) but are not polled. Tests: `LilLisa_Server/tests/test_nightly_techsupport_sync.py`.
- **Beads:** task `pr42-blockers.3.1` **closed**. Sub-epic `pr42-blockers.3` (PERFORMANCE) **closed** (1/1 children). Parent epic `pr42-blockers` remains **open**

### RELIABILITY (`pr42-blockers.4`) — **closed** (5/5 children closed)

**Escalation tracker is set before the tech-support post and not cleared on failure** (`pr42-blockers.4.1`)  
`check_and_update_endorsement(..., "escalated")` runs before `chat_postMessage`. If the post fails (missing invite, `not_in_channel`, rate limit), the original thread is silent forever and nothing lands in tech support. Clicking again hits `[ESCALATE DUPLICATE]` and does nothing. Set the flag only after a successful post, or roll it back in `except`.

- **Done:** Lock is still claimed before the post (see 4.2). `clear_escalation_claim` runs in the `chat_postMessage` `except`, plus an ephemeral/in-thread error so the user can click again.
- **Beads:** `pr42-blockers.4.1` **closed**.

**Duplicate-click race: lock is taken after slow I/O** (`pr42-blockers.4.2`)  
`fetch_conversation_history` and `get_refined_escalation_query` run **before** `check_and_update_endorsement`. Two clicks (two bot replies in the same thread, or a double-click) can both finish refine and both post to tech support. Claim the lock immediately after `ack()`, then do network work.

- **Done:** After payload parse and the configured-channel check, the lock is claimed, then history/refine run. Combined with 4.1 rollback on post failure.
- **Beads:** `pr42-blockers.4.2` **closed**.

**Comment vs code: escalate button is not disabled when env is unset** (`pr42-blockers.4.3`)  
Comment says the button will be disabled if the channel env is unset. `process_msg` always attaches escalate blocks whenever it is not an expert answer. Click then logs `[ESCALATE] No techsupport channel configured` and returns after `ack()` — Slack shows a spinner, then nothing. Skip `build_escalation_blocks` when `get_techsupport_channel(product)` is None; on click, post an ephemeral/in-thread error.

- **Done:** `process_msg` only attaches escalate blocks when `get_techsupport_channel(product)` is set. Click with no channel posts ephemeral (then in-thread fallback). Comment/warning text matches.
- **Later:** `get_techsupport_channel` was removed. The button is attached when the single `TECHSUPPORT_CHANNEL_ID` is set. Every product escalates to that channel.
- **Beads:** `pr42-blockers.4.3` **closed**.

`**json.loads(raw_result)` can leave “Processing…” forever** (`pr42-blockers.4.4`)  
`get_ans` already returns plain strings on timeout/exception. `process_msg` still does `parsed = json.loads(raw_result)` with no try/except. That exception never hits `_post_message_with_fallback`. Wrap parse failure and post the error text.

- **Done:** `parse_get_ans_result` in `lil-lisa/src/utils.py` treats non-object JSON as the reply `response`. Tests: `lil-lisa/tests/test_parse_get_ans_result.py`.
- **Beads:** `pr42-blockers.4.4` **closed**.

**Crash windows lose or duplicate verified knowledge** (`pr42-blockers.4.5`)  

- `add_verified_qa_pair`: markdown append, then LanceDB, then state → orphan markdown, or markdown+DB without `added_to_verified_db` → duplicate on retry.  
- `replace_verified_qa_pair`: `delete_nodes` **before** rewrite/insert → gap with no row until retry.  
- `save_state`: truncating write, not temp+`replace` → kill mid-write corrupts `techsupport_sync_state.json`.  
Write markdown/state atomically; on replace, insert new nodes then delete old; `os.replace` for JSON.

- **Done:** `atomic_io.atomic_write_text` / `atomic_write_json` (temp + `os.replace`) used by ingest markdown/review state, `nightly_techsupport_sync.save_state`, and reembed `save_state`. Replace and enrich insert new LanceDB nodes, save review state, then `delete_nodes`. `add_verified_qa_pair` with a known `thread_ts` delegates to replace. Tests: `LilLisa_Server/tests/test_atomic_io.py`.
- **Beads:** `pr42-blockers.4.5` **closed**. Sub-epic `pr42-blockers.4` **closed**. Parent epic `pr42-blockers` remains **open** while `pr42-blockers.2.3` is open.

### INSTRUMENTATION (`pr42-blockers.5`)

None identified at blocker severity (logging leftovers are High Pri).

### MISC (`pr42-blockers.6`)

None identified at blocker severity.

---

## 2. High Pri (`pr42-hp`)

Should fix before merge.

### FEATURE (`pr42-hp.1`) — **closed** (4/4 children closed)

**Stream path ignores the new Q&A contract** (`pr42-hp.1.1`)  
`/invoke/` returns `answer_found`, `needs_escalation`, `links_text`, `primary_techsupport_match_title`. `/invoke_stream_with_nodes/` still only parses `response` + `reranked_nodes`. Fine if Slack only uses `/invoke/`; if the web UI streams, escalation and source links never appear there. State this in the PR, or wire the same fields into the stream.

- **Done (won’t wire stream):** Web UI is intentionally behind; Slack-only for v1. By design: the web app has no auth, and we do not want unauthenticated users escalating to tech support or seeing tech-support answers that may contain PII. `/invoke_stream_with_nodes/` left unchanged. Same note posted as a PR #42 comment (reviewer account cannot edit the fork PR body; submitter should fold it into the description).
- **Beads:** `pr42-hp.1.1` **closed**.

**Other escalate buttons in the thread stay live** (`pr42-hp.1.2`)  
`chat_update` only strips `escalation_note` / `escalation_actions` on the **clicked** message. Earlier/later bot replies still show the button. After a successful escalate those clicks no-op (if the tracker held). Strip them, or replace with “already posted to tech support.”

- **Done:** After a successful tech-support post (and on the duplicate-lock path), walk the whole thread via `conversations_replies`, strip escalate blocks from every bot message that still has them, and replace with an “Already posted to tech support.” context block. Helper: `strip_escalation_blocks_from_thread` in `lil-lisa/src/slack.py`.
- **Beads:** `pr42-hp.1.2` **closed**.

**Enrichment bar is too low** (`pr42-hp.1.3`)  
Tagged threads only need `is_useful` (conclusiveness skipped). A tagged escalation that is still an open argument can `MergeTechsupportSummaries` into a good verified entry. Require conclusive (or a dedicated “new insight vs noise” classifier) before enrich.

- **Reviewer conclusion:** Enrich is **not** a full regeneration. `enrich_verified_entry` summarizes the new thread, then `MergeTechsupportSummaries` does a merge/append (preserve existing content, add new insight). Title is reused verbatim. LanceDB: insert new nodes, then delete that entry’s old node ids — so duplicate **table rows** for the entry are removed, but duplicate **facts in the prose** are not guaranteed. Unresolved tagged threads can still merge unfinished debate into a good article. Differs from `replace_verified_qa_pair`, which regenerates from the same thread’s full history.
- **Author decision:** Keep current behavior (`skip_conclusive=True` when tagged). No code change.
- **Beads:** `pr42-hp.1.3` **closed** (won’t-fix / accepted as designed).

`**test_techsupport_*.py` are not tests** (`pr42-hp.1.4`)  
Live Slack + live LanceDB eyeball scripts, with a hardcoded production-ish `DEFAULT_THREAD_TS`. They can mutate the verified table. Rename to `tools/` or guard with `--write`, and add pytest for `parse_summary_markdown`, slugger, and state transitions (no network).

- **Done:** Live scripts moved to `LilLisa_Server/smoke/` (dropped `test_` prefix so pytest won’t collect them): `smoke/techsupport_qa_ingest.py`, `smoke/techsupport_classifier.py`. Existing unit tests moved to `LilLisa_Server/tests/` and `lil-lisa/tests/`. No new tests added (no parse_summary_markdown / slugger / state-transition coverage in this pass). No `lil-lisa-web` smoke/tests folders (nothing to move).
- **Beads:** `pr42-hp.1.4` **closed**. Parent FEATURE epic `pr42-hp.1` **closed**.

### SECURITY (`pr42-hp.2`) — **closed** (2/2 children closed)

**GitHub token in clone URL** (`pr42-hp.2.1`)  
`_authenticated_url` puts the PAT in `https://{token}@github.com/...`, which Git writes into `.git/config` in the temp clone. `rmtree` in `finally` helps; a crash or `git` error log can still leak the token. Prefer `GIT_ASKPASS` / `x-access-token` header, and never log the authenticated URL.

- **Done:** Clone/push use the plain `GITHUB_REPO_URL` plus `GIT_ASKPASS` (helper prints `$GITHUB_TOKEN`; token is not in the URL or `.git/config`). `credential.helper=` for that clone so a global helper cannot persist the PAT. Same `github_push.env` vars. Tests: `LilLisa_Server/tests/test_github_sync.py`.
- **Beads:** `pr42-hp.2.1` **closed**.

**Full Q&A still logged at INFO (privacy)** (`pr42-hp.2.2`)  
Customer/employee questions and model output land in production logs. `DEBUG_NO_ANSWER` sites are DEBUG in the final tree, but these INFO dumps remain: retry `retry_response: %r`, `logger.info(str(conv_dict))` in `get_ans`, `[ESCALATE CLICK]` query, `[ESCALATE HISTORY]` conversation_history, `[ESCALATE REFINE]` / `[ESCALATE FINAL]` queries. Default to DEBUG; never log full answers at INFO. Retry metadata (`changed_outcome`, top score) can stay INFO.

- **Done:** Full query/history/answer/refine/retry JSON moved to DEBUG (not dropped). INFO heartbeats: Slack `[GET_ANS] conv_id=… http_status=… bytes=… elapsed_ms=…`; server `invoke session_id=… product=… query_chars=… answer_found=… elapsed_ms=… outcome=…`; escalate lines keep ids/counts/char lengths. `NO_ANSWER_RETRY` INFO is metadata only; full payload is `NO_ANSWER_RETRY_DETAIL` at DEBUG.
- **Beads:** `pr42-hp.2.2` **closed**. Parent SECURITY epic `pr42-hp.2` **closed**.

### PERFORMANCE (`pr42-hp.3`) — **closed** (2/2 children closed)

`**dspy>=2.6.27` added to the server runtime** (`pr42-hp.3.1`)  
Nothing under `src/` imports `dspy`. It is used by pipeline scripts. Putting it in default `dependencies` (and large `uv.lock` churn) bloats the API image. Prefer an optional extra (`[project.optional-dependencies] pipeline`) or a separate scripts package.

- **Done:** Moved pipeline jobs to top-level `lil-lisa-cron-scripts/` with its own `pyproject.toml` (`dspy>=2.6.27` + path dep on `LilLisa_Server`). Removed `dspy` from `LilLisa_Server` default dependencies and regenerated `uv.lock` (DSPy and related packages dropped from the API lock). Cron: `make setup-env` then `python nightly_pipeline.py`. Thread tags stay at `LilLisa_Server/scripts/techsupport_thread_tags.json` (API writes them). Tests: `lil-lisa-cron-scripts/tests/`.
- **Superseded (deployment rework):** The split was reversed. The jobs now live at `LilLisa_Server/cron/` with no `pyproject.toml` of their own, `dspy>=2.6.27` is back in the server's default dependencies, and the API image ships and runs them (`src/techsupport_cron.py`). The original image-bloat concern did not hold up once measured: DSPy adds ~21 mostly pure-Python packages to an image that already ships `torch` with CUDA wheels. In exchange, the deployment needs no host access, no second image, and no shared volume between containers, and there is only one virtualenv to build. See Section 5 of the deploy notes.
- **Beads:** `pr42-hp.3.1` **closed**.

**Blocking `requests` on the Bolt async loop** (`pr42-hp.3.2`)  
`get_ans` already did this. Escalation adds `tag_techsupport_thread`, `fetch_conversation_history`, and `get_refined_escalation_query` as sync `requests` inside `async def`. A slow refine (60s timeout) stalls every other Slack handler. Use `httpx.AsyncClient` / `aiohttp`, or `asyncio.to_thread`.

- **Done:** `await asyncio.to_thread(...)` via `_requests_call` in `lil-lisa/src/slack.py` for every `requests.post` / `requests.get` in that module (including `get_ans` and the three escalation helpers).
- **Beads:** `pr42-hp.3.2` **closed**. Parent PERFORMANCE epic `pr42-hp.3` **closed**.

### RELIABILITY (`pr42-hp.4`) — **closed** (4/4 children closed)

**Race on `techsupport_thread_tags.json`** (`pr42-hp.4.1`)  
Read whole file → mutate dict → write. Two concurrent escalations can drop a tag. Write to a temp file and `os.replace`, and serialize with a lock (or SQLite). Lost tags mean duplicate verified entries instead of merge.

- **Done:** `upsert_thread_tag` in `LilLisa_Server/src/techsupport_thread_tags.py` loads, sets the key, and `atomic_write_json` (temp + `os.replace`). `POST /tag_techsupport_thread/` holds `THREAD_TAGS_LOCK` around the upsert. Tests: `LilLisa_Server/tests/test_techsupport_thread_tags.py`.
- **Beads:** `pr42-hp.4.1` **closed**.

`**ENDORSEMENT_TRACKER` is process memory** (`pr42-hp.4.2`)  
Restart clears `escalated`. Bot talks again in the original thread; users can escalate twice. Same as old endorsement tracking, but silence-after-escalate is a product promise. Persist (Speedict/sqlite) or document the restart behavior.

- **Done:** Persist **only** successful escalations in `lil-lisa/escalation_tracker.json` (gitignored; override `ESCALATION_TRACKER_PATH`). Max age 90 days (`ESCALATION_MAX_AGE_DAYS`). Thumbs stay RAM-only. `claim_escalation` before the tech-support post; `clear_escalation` on post failure. `process_msg` silence checks the persisted store (expired rows no longer silence). Tests: `lil-lisa/tests/test_escalation_tracker.py`.
- **Beads:** `pr42-hp.4.2` **closed**.

`**configure_dspy_lm()` at import time** (`pr42-hp.4.3`)  
`techsupport_classifier.py` and `techsupport_qa_ingest.py` call it at module load. Importing `nightly_pipeline` requires LLM key files even for `--help` or unit-test parsers. Configure lazily inside `classify_thread` / summarize.

- **Done:** Idempotent lazy `configure_dspy_lm()`; called from `classify_thread`, `generate_verified_title_and_summary`, `enrich_verified_entry`, `historical_import.convert_entry`, and `historical_revert_to_prose._generate_title_with_retry`. Module-level calls removed. Tests: `LilLisa_Server/cron/tests/test_lazy_dspy_configure.py`.
- **Beads:** `pr42-hp.4.3` **closed**. Parent RELIABILITY epic `pr42-hp.4` **closed**.

`**github_sync.load_env` ignores `os.environ**` (`pr42-hp.4.4`)  
Every other script does dotenv then overlay env. Cron/`GITHUB_TOKEN` in the environment will be ignored if the file is empty. Inconsistent and surprising in deploy.

- **Done:** Same `{**dotenv_values(...), **os.environ}` overlay as `nightly_techsupport_sync.load_env`. Empty file placeholders no longer shadow a real process-env token. Tests: `LilLisa_Server/cron/tests/test_github_sync.py`.
- **Beads:** `pr42-hp.4.4` **closed**.

### INSTRUMENTATION (`pr42-hp.5`)

None beyond the INFO Q&A dumps listed under Security.

### MISC (`pr42-hp.6`)

None identified at high severity beyond the test-script issue (listed under Feature).

---

## 3. Medium Pri (`pr42-mp`)

Fix in this PR if cheap; otherwise track immediately after.

### FEATURE (`pr42-mp.1`) — **closed** (11/11 remaining children closed; `pr42-mp.1.1` reparented)

**Fragile “is this a techsupport node?” test** (`pr42-mp.1.1`)  
`webportal_url is None and github_url is not None` is used for useful links and for `primary_techsupport_match_title`. A doc node missing `webportal_url` would be treated as techsupport and could enrich into the wrong entry. Prefer an explicit metadata flag (`source: techsupport`) set at ingest.

- **Done:** Not treated as a this-PR code change. Heuristic kept for v1. Ingest already stamps `source: techsupport` on new LanceDB rows; retrieval still uses the URL heuristic. Long-term fix is an explicit flag at retrieve time.
- **Beads:** `pr42-mp.1.1` **open**, P3. Reparented to `pr42-enhancements` (not closed). ID stays `pr42-mp.1.1`.

**Prompt rule clash (`[[NO_ANSWER]]` vs “no tags”)** (`pr42-mp.1.2`)  
Rule 7: no extraneous symbols/tags/prefixes. Rule 9: start with `[[NO_ANSWER]]`. The version branch then appends another **“10.”** while tables are already rule 10, and tells the model to mention versions even on no-answer (with a carve-out that the marker must stay first). Easy for the model to put the version sentence before the marker. Number the injected rule 11, and exempt the marker from rule 7.

- **Done:** `qa_system_prompt.txt` rule 7 exempts the leading `[[NO_ANSWER]]` marker; rule 9 names that exemption. Injected version text is rule 11 via `append_product_version_rule` in `LilLisa_Server/src/utils.py`. Tests: `LilLisa_Server/tests/test_qa_version_rule.py`.
- **Beads:** `pr42-mp.1.2` **closed**.

**Init-error paths set `answer_found: True`** (`pr42-mp.1.3`)  
If IDA/IDDM/IDO indices are missing, the user sees “contact an administrator” **without** escalation. If that is intentional (don’t page tech support for a down bot), comment it. Otherwise `False` is more consistent.

- **Done:** Behavior kept (`answer_found: True`). Comments on the IDA/IDDM/IDO init-error returns in `answer_from_document_retrieval` state that hiding escalate is intentional so we do not page tech support for a down bot.
- **Beads:** `pr42-mp.1.3` **closed**.

`**match_context_text` is always `""**` (`pr42-mp.1.4`)  
`invoke()` reads it from the retrieval JSON; `answer_from_document_retrieval` never sets it. Dead field unless the Slack client fills it.

- **Done:** Field removed from `/invoke/` JSON (`LilLisa_Server/src/main.py`). Nothing read it.
- **Beads:** `pr42-mp.1.4` **closed**.

**Anyone in the channel can escalate anyone else’s thread** (`pr42-mp.1.5`)  
Payload `user_id` is the original asker, not the clicker. A troll can page tech support and `@` the asker. Restrict to the asker (and maybe the product expert), or include the clicker in the TS note (`clicked by <@U…>`). This is also an abuse/security concern.

- **Done:** Escalate remains open to anyone in the channel. Tech-support context note now includes `Clicked by <@U…>` from Slack `body["user"]["id"]`. Asker-only restriction not in this PR.
- **Beads:** `pr42-mp.1.5` **closed**.

`**@` mention in a tech-support channel still runs `process_msg**` (`pr42-mp.1.6`)  
Techsupport channels special-case mentions into `process_msg`, but `determine_product_and_expert` does not map TS channel IDs, so the user gets “I am unable to provide answers in this channel.” Either map TS channels to the product (no escalate button there) or don’t call `process_msg` for mentions in TS.

- **Done:** Mentions in TS no longer call `process_msg`. Bot replies in-thread: “Ask in the relevant product channel, not here.”
- **Beads:** `pr42-mp.1.6` **closed**.

**Slack `value` 2000-char cap can still overrun** (`pr42-mp.1.7`)  
Query is capped at 1500; `primary_techsupport_match_title` is not. Long titles + JSON envelope can make Slack drop the action. Cap or hash the title.

- **Done:** `build_escalation_button_value` in `lil-lisa/src/utils.py` shrinks the title until the JSON is ≤ 2000 (drops the title field if it still cannot fit). Tests: `lil-lisa/tests/test_escalation_button_value.py`.
- **Beads:** `pr42-mp.1.7` **closed**.

**Mixed embedding spaces** (`pr42-mp.1.8`)  
Nightly insert uses Voyage `input_type="query"` (same as `VoyageEmbedding._get_text_embedding`). Weekly reembed uses contextual `input_type="document"` over the whole file. New rows live in a different space until the next reembed. Document that in ops, or embed new rows with a single-chunk contextual call so the space matches. (`_get_text_embedding` using `"query"` for document-shaped calls is the same helper, moved into `embedding_config.py`.)

- **Done (docs only):** Ops note in `PR42 Release and Deployment Notes.md` §4b and ingest module comments. Nightly insert still uses query-space until weekly reembed.
- **Beads:** `pr42-mp.1.8` **closed**. Follow-on task `pr42-enhancements.3` (**open**, P3, `discovered-from` this task): embed new rows with a single-chunk contextual / `document` call so they match `techsupport_contextual_reembed.py`.

`**##` in generated titles/summaries breaks the parser** (`pr42-mp.1.9`)  
`parse_summary_markdown` splits on `^##`. Prompt says not to use `#` in titles; models still do. Sanitize (strip heading markers; indent `##` in body).

- **Done:** `techsupport_markdown.py` strips `#` from titles and indents body lines that start with `##`. Called on generate / add / replace / enrich. Tests: `LilLisa_Server/cron/tests/test_techsupport_markdown.py`.
- **Beads:** `pr42-mp.1.9` **closed**.

**Enrich matches title exactly; first hit wins** (`pr42-mp.1.10`)  
Duplicate auto-titles merge into the wrong entry. Prefer node id / markdown index from the tag payload, not title string.

- **Done (uniqueness at ingest):** New titles that collide get ` - 2`, ` - 3`, … (`uniquify_techsupport_title`). Enrich still looks up by exact title; uniqueness makes first-hit safer on new data.
- **Beads:** `pr42-mp.1.10` **closed**. Follow-on task `pr42-enhancements.4` (**open**, P3, `discovered-from` this task): store a stable id in the tag payload and enrich by id (title fallback for old tags).

`**historical_import.py` is retired but runnable** (`pr42-mp.1.11`)  
It still extracts Q&A and passes `(question, answer)` into `append_summary_to_markdown`. A future “1-year production import” would corrupt the prose file. Guard `main()` with a hard error (“retired; use …”) or delete/move it.

- **Done:** `main()` prints a retired message and `sys.exit(2)` before DSPy/LanceDB import when run as a script. File kept for reference. Use `historical_import_production.py` or `nightly_pipeline.py` instead.
- **Beads:** `pr42-mp.1.11` **closed**.

**Single `TECHSUPPORT_CHANNEL_ID` vs three Slack env vars** (`pr42-mp.1.12`)  
Comments say IDA/IDDM/IDO channels are the same. If they ever differ, the pipeline only watches one. Assert they match at bot+pipeline startup, or sync all three.

- **Done:** Bot `assert_shared_techsupport_channel_ids` raises if configured IDA/IDDM/IDO IDs disagree (unset products ignored). Also warns when IDO’s channel is unset. Pipeline `assert_pipeline_matches_product_channel_ids` raises if optional product-specific IDs in `techsupport_sync.env` differ from `TECHSUPPORT_CHANNEL_ID`. Documented in env example and Release/Deployment Notes.
- **Later:** Those three bot variables and both assert helpers are gone. The bot and the backend each have one `TECHSUPPORT_CHANNEL_ID`, set to the same channel.
- **Beads:** `pr42-mp.1.12` **closed**. Parent FEATURE epic `pr42-mp.1` **closed**.

### SECURITY (`pr42-mp.2`)

None beyond the escalate-anyone-else’s-thread item under Feature.

### PERFORMANCE (`pr42-mp.3`) — **closed** (1/1 children closed)

`**_no_answer_streak` never expires** (`pr42-mp.3.1`)  
In-process `dict` keyed by `session_id`. Comment says “flag after one rephrase,” but `needs_escalation = not answer_found` (streak unused). Unbounded growth on a long-lived server. Drop the dict or bound/TTL it. Comment and code disagree.

- **Done:** Removed `_no_answer_streak` and the unused increment/reset. `needs_escalation` is still `not answer_found`. Debug log no longer includes the streak.
- **Beads:** `pr42-mp.3.1` **closed**. Parent PERFORMANCE epic `pr42-mp.3` **closed**.

### RELIABILITY (`pr42-mp.4`) — **closed** (2/2 children closed)

`**clamp_to_slack_block_limit` is a hard slice** (`pr42-mp.4.1`)  
Unlike server `_truncate_match_answer`, this can cut inside fenced code and produce `invalid_blocks` anyway (fallback then strips formatting). Prefer the fence-aware truncate.

- **Done:** `truncate_preserving_code_fences` in `lil-lisa/src/utils.py` (copy of server fence logic; no shared package). `clamp_to_slack_block_limit` uses it. Tests: `lil-lisa/tests/test_truncate_preserving_code_fences.py`.
- **Beads:** `pr42-mp.4.1` **closed**.

`**tag_techsupport_thread` ignores HTTP status** (`pr42-mp.4.2`)  
No `raise_for_status()`. Failed tags look successful; nightly ingest will create a duplicate instead of merging.

- **Done:** Slack `tag_techsupport_thread` calls `raise_for_status()`. HTTP failures log status and a truncated body and are still swallowed so escalate UX is unchanged.
- **Beads:** `pr42-mp.4.2` **closed**. Parent RELIABILITY epic `pr42-mp.4` **closed**.

### INSTRUMENTATION (`pr42-mp.5`) — **closed** (1/1 children closed)

`**enriched` omitted from the summary log line** (`pr42-mp.5.1`)  
`counts` tracks `enriched` (and GitHub push / reload use it). The `logger.info` pipeline summary and the admin-alert string include `replaced` but not `enriched`. An enrich-only night looks like `added=0 replaced=0`.

- **Done:** `format_pipeline_counts()` in `LilLisa_Server/cron/pipeline_summary.py` renders all counts (`enriched` after `added`). Shared by the INFO summary and the admin-alert parenthetical in `nightly_pipeline.py`. Unknown keys are appended sorted. Tests: `LilLisa_Server/cron/tests/test_format_pipeline_counts.py`.
- **Beads:** `pr42-mp.5.1` **closed**. Parent INSTRUMENTATION epic `pr42-mp.5` **closed**.

### MISC (`pr42-mp.6`)

None identified at medium severity.

---

## 4. Low Pri (`pr42-lp`) — **closed** (all children closed)

Worth doing; not merge-blocking.

### FEATURE (`pr42-lp.1`) — **closed** (3/3 children closed)

**IDO techsupport channel is easy to omit** (`pr42-lp.1.1`)  
`lil-lisa.env.example` has `TECHSUPPORT_CHANNEL_ID_IDA` / `_IDDM` but omits `_IDO` even in the IDO optional block. The bot also only warns when IDA/IDDM are unset, not IDO. Easy to deploy IDA/IDDM escalation and forget IDO.

- **Done:** Commented `TECHSUPPORT_CHANNEL_ID_IDO` in the IDO optional block of `lil-lisa/app_envfiles/lil-lisa.env.example`; comment notes it must equal the IDA/IDDM channel IDs. IDO stays optional.
- **Later:** `TECHSUPPORT_CHANNEL_ID_IDA` / `_IDDM` / `_IDO` were replaced by one `TECHSUPPORT_CHANNEL_ID` in `lil-lisa.env.example`. IDO does not have its own tech-support channel.
- **Beads:** `pr42-lp.1.1` **closed**.

`**lillisa_server.env.example` omits `LIL_LISA_SERVER_URL**` (`pr42-lp.1.2`)  
`nightly_pipeline.reload_techsupport_index()` reads it (default `http://127.0.0.1:8000`). Cron on another host will silently hit localhost.

- **Done:** Added `LIL_LISA_SERVER_URL=http://127.0.0.1:8000` to `LilLisa_Server/env/lillisa_server.env.example` with a comment that cron should override if it does not share a host with the API.
- **Beads:** `pr42-lp.1.2` **closed**.

`**tools/env/` has no references in this PR’s code** (`pr42-lp.1.3`)  
Harmless if you have local harvest tools; otherwise it is a dead ignore. Short comment or drop it.

- **Done:** Removed unused `tools/env/` ignore from `lil-lisa/.gitignore`.
- **Beads:** `pr42-lp.1.3` **closed**. Parent FEATURE epic `pr42-lp.1` **closed**.

### SECURITY (`pr42-lp.2`) — **closed** (3/3 children closed)

**Harvest dump gitignore is name-specific** (`pr42-lp.2.1`)  
`production_test_pull*.txt` / `test_harvest_pull*.txt` only. A differently named dump under `data/historical_import/` would be committable Slack data. Ignoring `data/historical_import/` (and tracking examples elsewhere) is safer if that directory is only for local harvest output.

- **Done:** `LilLisa_Server/.gitignore` ignores `data/historical_import/*` and un-ignores `.gitkeep` so the empty directory stays in git.
- **Beads:** `pr42-lp.2.1` **closed**.

`**GITHUB_TOKEN=ghp_your-personal-access-token-here` looks like a real PAT prefix** (`pr42-lp.2.2`)  
Some scanners flag `ghp_`. Use `GITHUB_TOKEN=` plus a comment.

- **Done:** `LilLisa_Server/cron/env/github_push.env.example` uses empty `GITHUB_TOKEN=` plus a header comment. No `ghp_` prefix.
- **Beads:** `pr42-lp.2.2` **closed**.

`**AUTHENTICATION_KEY` / `JWT_SECRET_KEY` in the server example** (`pr42-lp.2.3`)  
Expected placeholders; confirm this file is never used as a real env (gitignore of `env/*` with `!*.example` is correct).

- **Done:** Header comment on `LilLisa_Server/env/lillisa_server.env.example`: copy to gitignored `lillisa_server.env`; never put real secrets in `*.example`.
- **Beads:** `pr42-lp.2.3` **closed**. Parent SECURITY epic `pr42-lp.2` **closed**.

### PERFORMANCE (`pr42-lp.3`) — **closed** (no children)

None identified at low severity.

### RELIABILITY (`pr42-lp.4`) — **closed** (3/3 children closed)

**Silent miss on optional Makefile env include** (`pr42-lp.4.1`)  
`-include` will not warn if someone expected `./env/lillisa_server.env` (or lil-lisa’s `app_envfiles/${IMAGE}.env`) and forgot to copy from `.env.example`. A one-line comment above the include would help.

- **Done:** Comments above `-include` in `LilLisa_Server/makefile` and `lil-lisa/makefile`: optional so make works without env; copy from `*.example` before deploy/run; examples stay unquoted `KEY=value` (Make include ≠ python-dotenv).
- **Beads:** `pr42-lp.4.1` **closed**.

**Rollback does not restore markdown or `review_state`** (`pr42-lp.4.2`)  
Rolling LanceDB back without the file will drift. Document “restore md + state + table together.”

- **Done:** Module docs plus `_print_coordinated_restore_reminder()` after `rollback_to_version()` in `techsupport_rollback.py`. Script does not auto-restore markdown or `review_state`; it prints the files to restore with LanceDB.
- **Beads:** `pr42-lp.4.2` **closed**.

**Classifier `Literal["yes", "no"]` equality is brittle** (`pr42-lp.4.3`)  
If the model returns `Yes` / `yes.`, comparison fails. Normalize.

- **Done:** `is_yes_answer()` in `techsupport_classifier.py` treats yes/no case-insensitively (punctuation stripped); unknown stays `False`. Tests: `LilLisa_Server/cron/tests/test_techsupport_classifier_yes_no.py`.
- **Beads:** `pr42-lp.4.3` **closed**. Parent RELIABILITY epic `pr42-lp.4` **closed**.

### INSTRUMENTATION (`pr42-lp.5`) — **closed** (no children)

None beyond the IDO-missing-warn aspect of the Feature item above.

### MISC (`pr42-lp.6`) — **closed** (3/3 children closed)

**One gitignore line per state filename will keep growing** (`pr42-lp.6.1`)  
A pattern such as `scripts/*_state.json` plus an explicit `techsupport_thread_tags.json` would absorb later state files automatically.

- **Done:** `LilLisa_Server/cron/.gitignore` uses `*_state.json`. Thread tags stay name-specific on the server (`LilLisa_Server/scripts/techsupport_thread_tags.json`).
- **Beads:** `pr42-lp.6.1` **closed**.

**Inconsistent `KEY = value` vs `KEY=value` in examples** (`pr42-lp.6.2`)  
Harmless for python-dotenv; slightly annoying to copy.

- **Done:** Tracked env examples normalized to `KEY=value`. Live env files not touched.
- **Beads:** `pr42-lp.6.2` **closed**.

**One-shot scripts in the default mental model** (`pr42-lp.6.3`)  
`backfill_github_urls.py` / `historical_import_production.py` should stay out of the default cron (README/makefile target that is *not* `nightly_pipeline`).

- **Done:** Deploy notes §4 and `lil-lisa-cron-scripts/makefile`: `run-nightly` is the only cron target; historical import and GitHub URL backfill are optional one-shot make targets, never cron.
- **Superseded (deployment rework):** There is no crontab at all now — the API process schedules `nightly_pipeline.py` itself on a tick, and `POST /run_nightly_pipeline/` forces a run. The underlying point still holds and is still enforced: `nightly_pipeline.py` is the only thing that ever runs on a schedule, and `historical_import_production.py` / `backfill_github_urls.py` remain manual make targets in `LilLisa_Server/cron/`.
- **Beads:** `pr42-lp.6.3` **closed**. Parent MISC epic `pr42-lp.6` **closed**. Parent epic `pr42-lp` **closed**.

---

## 5. Nits (`pr42-nits`) — **closed** (all children closed)

Non-blocking polish.

### FEATURE (`pr42-nits.1`) — **closed** (2/2 children closed)

`**ESCALATE_NOTE_TEXT` is a bit legalistic** (`pr42-nits.1.1`)  
Fine for v1.

- **Done:** Plainer copy: “If this didn't help and your question is complete, post it to tech support.” Completeness caution kept.
- **Beads:** `pr42-nits.1.1` **closed**.

`**qa_system_prompt.txt` still has no trailing newline** (`pr42-nits.1.2`)

- **Done:** Trailing newline on `LilLisa_Server/data/prompts/qa_system_prompt.txt`.
- **Beads:** `pr42-nits.1.2` **closed**. Parent FEATURE epic `pr42-nits.1` **closed**.

### SECURITY (`pr42-nits.2`) — **closed** (1/1 children closed)

**Make `include` ≠ dotenv** (`pr42-nits.2.1`)  
`KEY=value` works; quoted values, `export`, and inline comments may not match how the Python app loads the same file. Already true of these `-include`s; no Makefile change required for that alone.

- **Done:** Documented on the Makefile `-include` comments (same as `pr42-lp.4.1`): keep examples as unquoted `KEY=value`, no `export`, no inline comments.
- **Beads:** `pr42-nits.2.1` **closed**. Parent SECURITY epic `pr42-nits.2` **closed**.

### PERFORMANCE (`pr42-nits.3`) — **closed** (no children)

None.

### RELIABILITY (`pr42-nits.4`) — **closed** (3/3 children closed)

`**handle_escalate_to_techsupport` channel id sources may diverge** (`pr42-nits.4.1`)  
Uses `body["channel"]["id"]` for `chat_update` but `orig_channel_id` from the payload for posts — usually the same; if a message is forwarded, maybe not.

- **Done:** Still uses payload `channel_id` for chat_update/posts. `warn_if_escalate_body_channel_mismatch` logs when `body.channel.id` differs. Tests in `lil-lisa/tests/test_escalation_button_value.py`.
- **Beads:** `pr42-nits.4.1` **closed**.

**Duplicate retriever globals** (`pr42-nits.4.2`)  
`TECHSUPPORT_QA_PAIRS_RETRIEVER = None` declared twice (same pre-existing pattern for other retrievers).

- **Done:** Removed the duplicate `TECHSUPPORT_QA_PAIRS_RETRIEVER = None` in `LilLisa_Server/src/agent_and_tools.py`. Other retrievers still use the pre-existing double-declare pattern.
- **Beads:** `pr42-nits.4.2` **closed**.

`**except Warning` around retrieve is odd** (`pr42-nits.4.3`)  
`Warning` is an `Exception`. Pre-existing for docs; copied for techsupport.

- **Done:** Techsupport retrieve catches `Exception` and fail-closes (empty results + `logger.exception`). Docs retrieve still has the pre-existing `except Warning`.
- **Beads:** `pr42-nits.4.3` **closed**. Parent RELIABILITY epic `pr42-nits.4` **closed**.

### INSTRUMENTATION (`pr42-nits.5`) — **closed** (no children)

None beyond items already listed at Low/High.

### MISC (`pr42-nits.6`) — **closed** (1/1 children closed)

`**GithubAnchorSlugger` increment-from-zero** (`pr42-nits.6.1`)  
Stores `result` with count `0` then increments `base`. Worth a unit test against github-slugger fixtures (some live README checks were done; encode those as tests).

- **Done:** Algorithm unchanged. Unit tests in `LilLisa_Server/cron/tests/test_github_anchor.py`.
- **Beads:** `pr42-nits.6.1` **closed**. Parent MISC epic `pr42-nits.6` **closed**. Parent epic `pr42-nits` **closed**.

---

## Follow-on enhancements (`pr42-enhancements`) — **open** epic (0/5 children complete)

Created for work demoted from blockers (and later from hp/mp). Not required before merge.

**Validate techsupport classifier (labeled dataset + F1)** (`pr42-enhancements.1`) — **open** epic, P3, `discovered-from` `pr42-blockers.1.1`. Collect labeled useful/conclusive threads, switch to a stronger model, report F1. Cheaper first step on the epic: Slack resolution reaction or don’t treat auto-ingest as expert-verified until human sign-off.

**Eval same-prompt NO_ANSWER retry relevance from logs** (`pr42-enhancements.2`) — **open** task, P3, `discovered-from` `pr42-blockers.1.2`. Extract `NO_ANSWER_RETRY |` INFO JSON (retry is **served**, so `changed_outcome=true` is live user impact), judge whether try-2 is grounded in the top chunk, report flip rate and grounded-among-flips, then decide keep / drop / change the retry.

**Explicit `source: techsupport` metadata at retrieve** (`pr42-mp.1.1`) — **open** task, P3, reparented from `pr42-mp.1`. Heuristic (`webportal_url is None and github_url is not None`) kept for this PR. Ingest already sets `source: techsupport` on new rows; retrieval should use that flag (backfill or treat missing as not-techsupport) so a doc chunk missing `webportal_url` cannot become `primary_techsupport_match_title` / enrich the wrong entry.

**Embed nightly techsupport inserts in the same space as weekly contextual reembed** (`pr42-enhancements.3`) — **open** task, P3, `discovered-from` `pr42-mp.1.8`. In-PR work was ops docs only. This is the real fix: embed new verified Q&A rows with a single-chunk contextual / `input_type="document"` call matching `techsupport_contextual_reembed.py`, not `VoyageEmbedding._get_text_embedding` query-space. Acceptance: nightly add/replace/enrich vectors are comparable to post-reembed vectors; document extra Voyage cost.

**Enrich techsupport by stable id, not title string** (`pr42-enhancements.4`) — **open** task, P3, `discovered-from` `pr42-mp.1.10`. In-PR work uniquifies auto-titles at ingest (` - 2`). This enhancement puts a stable id in the tag payload (LanceDB node id and/or markdown index). Nightly enrich looks up by id; title is fallback for old tags. Duplicate titles must not merge into the wrong entry if uniqueness-at-ingest is bypassed.

---

## Appendix — verification notes

Checked each original finding against `git diff 17de46f 66733fe` and the `66733fe` tree.

**Dropped (not present in the aggregate, or not a code issue):**

- Per-commit SHA mapping and commit-level Approve/Request-changes table (this review is the final tree, not a commit series).
- Duplicate “anyone can escalate another user’s thread” (was listed under both Feature and Security).
- Duplicate “log-level fix is incomplete” (same remaining INFO dumps as the High Pri Security item; `DEBUG_NO_ANSWER` *is* DEBUG in the final tree).
- “`replaced` omitted from the pipeline summary log” — false in the final tree; `replaced` is in the format string. Retained as: `**enriched` omitted**.
- Nits about the Makefile commit subject (“remove inclusion”) and putting lil-lisa `-include` in a later commit — those are history/splitting, not the final diff. The final makefiles still use `-include` (covered under Low Pri reliability).

**Must-fix summary for the submitter:**

1. ~~JWT (or equivalent) on `/tag_techsupport_thread/`, `/get_conversation_history/`, `/refine_escalation_query/`.~~ **Done this session:** same `encrypted_key` JWT on all three; refine body capped at 32768 chars (`pr42-blockers.2.1` and `pr42-blockers.2.2` closed).
2. ~~Do not treat high rerank + retry as proof the answer is in context; detect `[[NO_ANSWER]]` at start of reply, not `find()`.~~ **Partially done this session:** startswith detection shipped (`pr42-blockers.1.3` closed). Same-prompt retry is still served; whether try-2 is grounded is deferred to `pr42-enhancements.2` (open) using the new `NO_ANSWER_RETRY` logs (`pr42-blockers.1.2` closed as non-blocker).
3. ~~Escalation tracker: lock before slow I/O; rollback if tech-support post fails; hide the button when the channel env is unset; don’t `json.loads` a timeout string.~~ **Done this session** (`pr42-blockers.4` and all five children closed).
4. ~~Don’t auto-publish unvalidated classifier output into live retrieval + GitHub;~~ strip PII; don’t `conversations.replies` every historical thread every night. **Classifier auto-publish demoted:** thumbs-down + GitHub markdown review (`pr42-blockers.1.1` closed). Measured classifier quality is `pr42-enhancements.1` (open). **PII partially done:** obvious-format regex + prompt line shipped; remaining regex needs a real QA doc (`pr42-blockers.2.3` still open). **Nightly-sync volume done:** `pr42-blockers.3.1` closed.
5. Finish the log downgrade (retry + escalate + `conv_dict`). Note: retry now logs a structured INFO `NO_ANSWER_RETRY | <json>` payload on purpose for `pr42-enhancements.2`; that line is not a leftover dump.

**Worked this session (beads):** Closed `pr42-blockers.4` and children `.4.1`–`.4.5` (escalate lock+rollback, omit button when TS env unset, parse timeout strings, atomic state/markdown + insert-then-delete). Parent `pr42-blockers` remains **open** while `pr42-blockers.2.3` is open. `pr42-blockers.1` and `pr42-blockers.3` were already closed.

**Also this session (`pr42-mp.1`):** Closed FEATURE epic `pr42-mp.1` and children `.1.2`–`.1.12` (prompt rule 11, init-error comments, drop `match_context_text`, clicker in TS note, TS mention one-liner, Slack value cap, embedding-space ops docs, `##` sanitize, unique titles, retired `historical_import.py`, shared-channel asserts). `pr42-mp.1.1` left **open** and reparented to `pr42-enhancements`. New follow-ons: `pr42-enhancements.3` (insert embeddings = weekly reembed), `pr42-enhancements.4` (enrich by stable id).