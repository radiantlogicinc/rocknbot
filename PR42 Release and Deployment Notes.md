# Deployment Documentation: Tech Support Integration

## 1 Overview

This adds two things to Rocknbot. First, a real-time escalation flow that lets Lil Lisa post unanswered questions to a tech support channel with one click. Second, a nightly pipeline that automatically turns resolved tech support conversations into verified answers the bots can use going forward, including merging new insight into an existing answer instead of always creating a duplicate. Nothing here requires a new service, a scheduler, or host access. It's an extension of `LilLisa_Server` and `lil-lisa`: the pipeline ships inside the API image and the API process runs it on its own schedule (Section 5).

The same nightly run also watches the product channels themselves. When Lil Lisa answers a question in the IDA / IDDM / IDO channel and a member of that product's expert user group replies in the thread to fix or extend the answer, that reply is picked up and either rewrites the verified entry the answer came from or becomes a new one. There is no command to learn and no prefix to type: being in the product's expert user group is the whole signal. See Section 4c.

## 2 New Environment Variables



### `LilLisa_Server/env/lillisa_server.env`


| Variable                                 | Default                      | Purpose                                                                                                                                                                                                                                                                                                                                                                                        |
| ---------------------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TECHSUPPORT_SYNC_INTERVAL_HOURS`        | `24`                         | How often the nightly pipeline checks the tech support channel for new or updated threads. Expert-adjustable.                                                                                                                                                                                                                                                                                  |
| `TECHSUPPORT_SYNC_HOT_DAYS`              | `30`                         | Nightly, `nightly_techsupport_sync.py` only refreshes known threads whose last reply activity is within this many days. Uses a cheap parent `latest_reply` lookup (`conversations.history`, not a full `conversations.replies` download).                                                                                                                                                      |
| `TECHSUPPORT_SYNC_CATCHUP_AGE_DAYS`      | `90`                         | Age cap for the periodic catch-up: known threads quieter than the hot window but still within this many days are checked on catch-up runs. Threads older than this stay in state but are not polled.                                                                                                                                                                                           |
| `TECHSUPPORT_SYNC_CATCHUP_INTERVAL_DAYS` | `7`                          | How often the catch-up pass runs (stamped in `techsupport_sync_state.json` as `last_catchup_timestamp`). Independent of `TECHSUPPORT_SYNC_INTERVAL_HOURS`.                                                                                                                                                                                                                                     |
| `TECHSUPPORT_SYNC_MAX_PARENT_LOOKUPS`    | `200`                        | Cap on each of the nightly hot set and the catch-up set (hottest first). Excess waits for a later run.                                                                                                                                                                                                                                                                                         |
| `TECHSUPPORT_SCAN_PRODUCT_CHANNELS`      | `true`                       | Master switch for the nightly product-channel scan for expert corrections (Section 4c). Set it to `false` to turn the whole pass off without unsetting the channel IDs.                                                                                                                                                                                                                        |
| `PRODUCT_SCAN_MAX_THREADS_PER_RUN`       | `50`                         | Per-product cap on how many changed threads one run classifies. The excess is remembered in that channel's state (`pending_thread_ids`) and picked up next run, hottest first, so a busy day cannot become an unbounded LLM bill.                                                                                                                                                              |
| `PRODUCT_SCAN_INITIAL_LOOKBACK_DAYS`     | `30`                         | How far back the *first* scan of a product channel looks. A channel with no state would otherwise be synced from timestamp zero, i.e. its entire history, and every old thread carrying an expert reply would be ingested. Later runs simply resume where the previous one stopped, so this only ever applies once per channel. Product channels only; the tech support channel is unaffected. |
| `TECHSUPPORT_REEMBED_INTERVAL_DAYS`      | `7`                          | How often the verified techsupport table gets a full contextual re-embed (late-chunking refresh). Expert-adjustable.                                                                                                                                                                                                                                                                           |
| `TECHSUPPORT_STATE_DIR`                  | `techsupport_state/` beside `LANCEDB_FOLDERPATH` (`/data/techsupport_state/` when that is `/data/lancedb`) | Folder for everything the pipeline writes: the state JSON files, the two tag files, and by default the verified markdown. Optional: the default already puts it on the LanceDB volume. Set it only if that volume's root is the LanceDB folder itself, or you want state elsewhere on persistent storage. |
| `VERIFIED_TECHSUPPORT_QA_FOLDERPATH`     | `<state dir>/verified_techsupport/` | Where the verified techsupport markdown file lives locally, before it gets pushed to the GitHub repo (see Section 7). Optional.                                                                                                                                                                                                                                                                          |
| `TECHSUPPORT_PIPELINE_TICK_MINUTES`      | `60`                         | How often the API process checks whether the nightly pipeline is due (Section 5). Not the nightly schedule itself — real work is gated by `TECHSUPPORT_SYNC_INTERVAL_HOURS`, so this only needs to be more frequent than it.                                                                                                                                                                   |
| `TECHSUPPORT_PREFLIGHT_STRICT`           | `true`                       | Whether the API **refuses to start** when the startup preflight finds a BLOCKING techsupport pipeline problem (Section 5e). Read from this file, overlaid by the process environment. Accepts `true`/`false`/`1`/`0`/`yes`/`no`/`on`/`off`, case-insensitively; blank or unrecognised stays strict, so a typo cannot switch the guard off. Set `false` only on a dev box with no Slack/GitHub credentials. Never set it in staging or production to get a pod to start: the pipeline stays broken. |

**Where the pipeline writes.** Everything the pipeline writes goes to one folder, `techsupport_state/`, beside the LanceDB folder. With `LANCEDB_FOLDERPATH=/data/lancedb` that is `/data/techsupport_state/`, on the same persistent volume, so no extra mount and no extra setting is needed:

- State JSON files (`techsupport_sync_state.json`, `techsupport_reembed_state.json`, `techsupport_review_state.json`).
- The two tag files (`techsupport_thread_tags.json`, `techsupport_answer_tags.json`).
- The verified markdown, in `verified_techsupport/techsupport_qa_pairs.md`.

Nothing is written under `/app`, which is rebuilt from the image on every rollout. `TECHSUPPORT_STATE_DIR` and `VERIFIED_TECHSUPPORT_QA_FOLDERPATH` move the folder or just the markdown. Both are optional and read from `lillisa_server.env` with the process environment on top, like `LANCEDB_FOLDERPATH`.




### `LilLisa_Server/cron/env/techsupport_sync.env` (dedicated file)


| Variable                                    | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SLACK_BOT_TOKEN`                           | Bot token the nightly pipeline uses to read the tech support channel. Same bot as `lil-lisa`'s.                                                                                                                                                                                                                                                                                                                                   |
| `TECHSUPPORT_CHANNEL_ID`                    | The real tech support channel ID. **Must equal** `lil-lisa`'s `TECHSUPPORT_CHANNEL_ID_IDA` / `_IDDM` / `_IDO` (those three must all be the same ID; the bot raises at startup if they disagree). If you also copy the product-specific names into this file, the pipeline refuses to start when they differ from `TECHSUPPORT_CHANNEL_ID`.                                                                                        |
| `ADMIN_CHANNEL_ID`                          | Where pipeline error notifications get posted.                                                                                                                                                                                                                                                                                                                                                                                    |
| `PRODUCT_CHANNEL_ID_IDA` / `_IDDM` / `_IDO` | The product channels Lil Lisa answers questions in, scanned nightly for expert corrections (Section 4c). Same IDs as `lil-lisa`'s `CHANNEL_ID_*`. All three are optional: a missing one simply disables that product's scan. The bot has to be a member of each one (Section 6).                                                                                                                                                  |
| `LIL_LISA_SLACK_USERID`                     | The bot's own Slack user ID, used to tag its turns as `(bot)` when a thread is formatted for the classifier and correction prompts. Mandatory for Slack bot. Optional for nightly pipeline: bot messages are still recognised by their `bot_id` without it.                                                                                                                                                                       |
| `EXPERT_GROUP_ID_IDA` / `_IDDM` / `_IDO`    | Slack **user group** IDs (e.g. `S0123ABCD`, from `usergroups.list` — not the `@handle`) whose members count as experts for that product. Same values as `lil-lisa`'s. `_IDA` and `_IDDM` are **required**: the resolver raises `ValueError` naming the missing variable, so the product-channel pass reports an error for that product instead of scanning it. `_IDO` is optional. Needs the `usergroups:read` scope (Section 6). |
| `EXPERT_GROUP_CACHE_SECONDS`                | How long expert group membership is cached before Slack is asked again. Default `300`.                                                                                                                                                                                                                                                                                                                                            |


Expert group membership is resolved by `LilLisa_Server/cron/expert_group.py`, the cron-side twin of `lil-lisa/src/expert_group.py` (the two packages deploy independently and cannot import each other, so the contract is duplicated — keep them aligned). Both cache membership for `EXPERT_GROUP_CACHE_SECONDS` and retry once on Slack `ratelimited`. On a failed lookup they keep serving the last cached membership for that product, with a warning; if nothing was ever cached they raise `ExpertLookupError` so the failure is loud. A Slack user group is now the only source of expert identity: there is no single-ID setting and no silent degradation to "nobody is an expert".

This is a separate env file from `lil-lisa`'s on purpose, so the pipeline never depends on `lil-lisa`'s directory or config existing — `LilLisa_Server` and `lil-lisa` deploy independently.

### `LilLisa_Server/cron/env/github_push.env` (dedicated file, new)


| Variable                | Purpose                                                                                                                                                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `GITHUB_TOKEN`          | Personal access token used to push both GitHub repos: the verified techsupport markdown (`GITHUB_REPO_URL`) and the golden QA pairs (`QA_PAIRS_GITHUB_REPO_URL`). See Section 7a for PAT scopes and how auth is passed (GIT_ASKPASS; token must not be in the URL). |
| `GITHUB_REPO_URL`       | HTTPS URL of that repo with **no** embedded credentials.                                                                                                          |




### `lil-lisa/app_envfiles/lil-lisa.env` (additions to the existing file)


| Variable                                 | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TECHSUPPORT_CHANNEL_ID_IDA`             | Tech support channel for IDA.                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `TECHSUPPORT_CHANNEL_ID_IDDM`            | Tech support channel for IDDM.                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `TECHSUPPORT_CHANNEL_ID_IDO`             | Tech support channel for IDO (optional product).                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `EXPERT_GROUP_ID_IDA` / `_IDDM` / `_IDO` | Slack **user group** IDs (e.g. `S0123ABCD`, from `usergroups.list` — not the `@handle`) whose members count as experts for that product. Every member gets the expert `👍` → golden QA pair and `#answer` behaviour, not just one person. `_IDA` and `_IDDM` are **required**: the bot refuses to start without them, the same as a missing `SLACK_BOT_TOKEN`. `_IDO` is optional; with no group nobody is an IDO expert. Needs the `usergroups:read` scope (Section 6). |
| `EXPERT_GROUP_CACHE_SECONDS`             | How long expert group membership is cached before Slack is asked again. Default `300`.                                                                                                                                                                                                                                                                                                                                                                                   |


**Removed variables:** `EXPERT_USER_ID_IDA` / `_IDDM` / `_IDO` are gone. Expert identity comes only from the Slack user group, so set `EXPERT_GROUP_ID_IDA` and `_IDDM` before deploying: the bot raises at startup without them. The expert who gets DMed after a golden QA pair is added is now the expert who actually reacted, not a configured ID.

In production these must all point to the **same** real channel (one shared tech support channel, not one per product). The bot asserts that at startup. The pipeline watches only `TECHSUPPORT_CHANNEL_ID` in `techsupport_sync.env`; set that to the same ID.

**Important existing variable:** `MAX_LENGTH`: must be 3000 or less, since that's Slack's hard limit on message length. A backup check also exists in case this is ever misconfigured.

### 2a. How these files reach the container

The tables above describe files in the repo. Whether those files are *inside* the image depends on which dockerfile built it, and the three differ on purpose:


| Image                                  | `env/` + `passwords/`                                              | Built by                                                                       |
| -------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| `dockerfile_local`, `dockerfile_cloud` | `COPY env /app/env` and `COPY passwords /app/passwords` — baked in | `make build-local` / `make build-cloud`, from your working copy                |
| `dockerfile_prod`                      | **Neither is copied** — supply at runtime                          | `.github/workflows/ci.yaml`, published as `radiantone/rocknbot-server:staging` |


`dockerfile_prod` omits them deliberately: CI builds it from a clean checkout where both are gitignored, and it gets pushed to a registry, so it must contain no secrets. That means a prod deployment has to supply the configuration itself, either way round:

- **Mount the directories** at `/app/env` and `/app/passwords`. This is the closest match to the tables above and to how local/cloud behave.
- **Inject environment variables.** Every config loader in `src/` and `cron/` reads its `.env` file and then overlays `os.environ`, so a real env var beats a missing file or an empty placeholder in it. This covers `lillisa_server.env`, `cron/env/techsupport_sync.env`, and `cron/env/github_push.env` alike.

The one thing injection alone cannot replace is the key files. `LLM_API_KEY_FILEPATH`, `OPENAI_API_KEY_FILEPATH`, and `VOYAGE_API_KEY_FILEPATH` name *paths*, and `main.lifespan` reads each file at startup — a missing one raises `FileNotFoundError` and the container never serves. So even under injection, the key files must exist at whatever paths those variables point to.

This fails fast and loudly at boot rather than degrading, so a misconfigured prod container is obvious immediately.

**A serving container now implies a configured pipeline — with three exceptions.** Startup validates the pipeline's own configuration (Section 5e) and, in the default strict mode, refuses to start when anything BLOCKING is missing. A container that is serving traffic therefore has what the pipeline needs to ingest, push the verified markdown, and answer, *unless*: strict mode was explicitly turned off, only DEGRADED problems are present (for example, a product-channel scan left unconfigured), or the configuration changed after boot (Section 5e, "After boot").

Do not assume this held before this release. It did not, and that assumption is exactly what hid the staging incident: startup never read `cron/env/techsupport_sync.env` or `cron/env/github_push.env`, so the API served traffic normally while every hourly tick raised `Missing required env var(s) ['SLACK_BOT_TOKEN', 'TECHSUPPORT_CHANNEL_ID']`, `TECHSUPPORT_QA_PAIRS` was never created, and the rollout looked healthy throughout.

**Do not hand-build the prod image on a dev box.** `COPY cron /app/cron` picks up whatever is in `cron/env/` in the build context, and `.dockerignore` does not exclude it (local and cloud rely on those files being present). Building `dockerfile_prod` from a working copy that has real credentials there would bake `SLACK_BOT_TOKEN` and `GITHUB_TOKEN` into an image that is otherwise secret-free. CI is unaffected — it clones fresh, and both files are gitignored.

## 3 New Files and Directories

- `techsupport_state/verified_techsupport/techsupport_qa_pairs.md`, beside the LanceDB folder (`/data/techsupport_state/verified_techsupport/` in the pod, overridable with `VERIFIED_TECHSUPPORT_QA_FOLDERPATH`). The verified techsupport QA source file (markdown). Auto-created and appended by the pipeline, and auto-pushed to GitHub (Section 7 after every update.
- `LilLisa_Server/cron/`. Nightly pipeline Python jobs (see Section 4), shipped in the API image and run by the API process itself — see Section 5. It has no `pyproject.toml` of its own: DSPy is in the server's dependencies and everything shares `LilLisa_Server/.venv`.  
- `LilLisa_Server/src/techsupport_cron.py`. Adapter that lets the API process run `nightly_pipeline.run_pipeline()`: resolves the cron package, serialises overlapping runs, and provides the periodic tick that `main.lifespan` starts.  
- `LilLisa_Server/.dockerignore`. Keeps virtualenvs and generated state out of the build context (it was several GB before this existed).  
- `LilLisa_Server/cron/env/techsupport_sync.env` and `LilLisa_Server/cron/env/github_push.env`. See Section 2 Both need to be gitignored (already confirmed).  
- State files, auto-created and gitignored, safe to delete if you want a clean slate since the pipeline will just re-detect everything as new on the next run. These three live in the techsupport state folder (`/data/techsupport_state/` in the pod):
  - `techsupport_sync_state.json`  
  - `techsupport_reembed_state.json`  
  - `techsupport_review_state.json`  
- Tag files, in the same state folder:
  - `techsupport_thread_tags.json`, written by the API request handler (`POST /tag_techsupport_thread/`). The pipeline only reads it; see Section 4a.  
  - `techsupport_answer_tags.json`, written by the API at answer time (`POST /invoke/`): `{Slack thread ts: title of the verified techsupport entry that answer cited}`. The pipeline only reads it, to tell an expert correcting a cited entry from a brand-new question; see Section 4c.

`techsupport_sync_state.json` is now keyed by channel, since the pipeline watches the tech support channel and the product channels:

```
{"version": 2,
 "channels": {"<channel id>": {"last_run_timestamp": ...,
                               "last_catchup_timestamp": ...,
                               "threads": {...},
                               "pending_thread_ids": [...]}}}
```

This is the only shape the pipeline accepts. A missing file is created fresh on the next run. A file whose top-level `version` is not `2`, including an early build's flat file with `last_run_timestamp` / `last_catchup_timestamp` / `threads` at the top level, is rejected: the run stops with a `RuntimeError` naming the path and telling you to delete the file. Delete it and re-run; the next run rebuilds it, and for the tech support channel that means the first sync starts from the beginning of the channel again. Nothing in production has the old shape, so this should never fire outside a dev box. `pending_thread_ids` holds the product-channel threads deferred by `PRODUCT_SCAN_MAX_THREADS_PER_RUN` (Section 4c).

## 4 Scripts


| Script                              | When to run                                                                                                    | Purpose                                                                                                                                                                                                                                                                   |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nightly_pipeline.py`               | **Nightly, automatic.** Run by the API process (Section 5); the **only** script that should ever be scheduled. | The main entry point. Orchestrates everything below, including the GitHub push and the live server's index reload.                                                                                                                                                        |
| `nightly_techsupport_sync.py`       | Nightly, via `nightly_pipeline.py`                                                                             | Detects new or updated threads in the tech support channel. New parents come from `conversations.history` since last run. Known threads are not all polled: nightly hot window + periodic 90-day catch-up, both capped (see the `TECHSUPPORT_SYNC_*` knobs in Section 2). |
| `techsupport_classifier.py`         | Nightly, via `nightly_pipeline.py`                                                                             | Classifies whether a thread is useful and conclusive (DSPy-based).                                                                                                                                                                                                        |
| `techsupport_qa_ingest.py`          | Nightly, via `nightly_pipeline.py`                                                                             | Extracts a summary from a resolved thread and adds it to the markdown file plus LanceDB, or merges it into an existing entry (see Section 4a).                                                                                                                            |
| `techsupport_contextual_reembed.py` | Nightly, via `nightly_pipeline.py` (on its own interval)                                                       | Periodically re-embeds the whole verified table together (late-chunking), on the interval from Section 2                                                                                                                                                                  |
| `techsupport_review_sync.py`        | Nightly, via `nightly_pipeline.py`                                                                             | Picks up manual edits an expert might make directly to the markdown file and syncs them into LanceDB. Fully optional, not a gate.                                                                                                                                         |
| `techsupport_rollback.py`           | Manual (ops)                                                                                                   | `list_available_versions()` and `rollback_to_version(n)`, see Section 8                                                                                                                                                                                                   |
| `github_sync.py`                    | Nightly, via `nightly_pipeline.py` (or standalone retry)                                                       | Pushes the current markdown file to the dedicated GitHub repo. Skips the push if the file hasn't actually changed. Called automatically by `nightly_pipeline.py`, but can also be run standalone if a push needs to be retried manually.                                  |
| `expert_group.py`                   | Library / helper                                                                                               | Resolves which Slack users count as experts for a product from `EXPERT_GROUP_ID_*` (Section 2), raising rather than guessing when a required group is unset or unreadable. Used by the product-channel scan. Cron-side twin of `lil-lisa/src/expert_group.py`.            |
| `github_anchor.py`                  | Library / helper                                                                                               | Generates GitHub-accurate anchor slugs from entry titles, so answers can link directly to the right section of the file on GitHub.                                                                                                                                        |
| `historical_import_production.py`   | **One-shot, manual.** Never scheduled. Optional: `make run-historical-import`.                                 | One-time bulk import of `data/historical_import/production_1year.txt` into the same verified-techsupport store the nightly pipeline maintains. Resumable; not part of the default schedule.                                                                               |
| `backfill_github_urls.py`           | **One-shot, manual.** Never scheduled. Optional: `make run-backfill-github-urls`.                              | One-time metadata patch: adds `github_url` on existing LanceDB rows that predate the GitHub-anchor feature. New nightly inserts already get this; do not schedule this script.                                                                                            |


`nightly_pipeline.py` is the only script that runs on a schedule. Leave `historical_import_production.py` and `backfill_github_urls.py` in `LilLisa_Server/cron/` (do not move them into a one-shot folder); they are invoked by hand only.

### 4a. Merge/Enrich: avoiding duplicate entries

If Lil Lisa answers a question by citing an existing verified techsupport entry, and the user escalates anyway, any genuinely new insight added in that escalation gets merged into the existing entry instead of creating a near-duplicate one. Here's how that works:

1. When an answer's top source is an existing techsupport entry, that connection is noted in the response.
2. If the user escalates, this gets recorded server-side through a new endpoint, `/tag_techsupport_thread/` on `LilLisa_Server`, following the same pattern as `/record_endorsement/`, and tracked in `techsupport_thread_tags.json`.
3. When the nightly pipeline later processes that escalation thread, if it's tagged this way, it uses a lighter classification bar. It just needs to be useful, not necessarily fully conclusive on its own, since the point isn't to independently resolve a new question but to add supplementary insight to something already resolved.
4. If it passes, the existing entry's content gets updated through an LLM merge call, but its title never changes. That's what keeps the entry's GitHub link stable even as its content grows over time.



### 4b. Embedding space until weekly reembed (ops)

Nightly `add` / `replace` / `enrich` embed each new row with Voyage `input_type="query"` (the same helper retrieval uses). `techsupport_contextual_reembed.py` (default every `TECHSUPPORT_REEMBED_INTERVAL_DAYS` days) re-embeds the **whole** `techsupport_qa_pairs.md` with contextual `input_type="document"`.

Until that weekly job runs, **new rows live in a different vector space** than rows last written by reembed. They are still searchable against user queries (query-to-query), but they may rank oddly next to older document-space rows. If a brand-new verified answer seems missing or weak in Slack, wait for the next reembed (or run `techsupport_contextual_reembed.py` once). Matching insert embeddings to the weekly job is follow-on work (`pr42-enhancements.3`).

### 4c. Product-channel scan for expert corrections

`nightly_pipeline.run_product_channel_pass()` runs after the tech support loop and before the GitHub push, once per configured product channel (`PRODUCT_CHANNEL_ID_IDA` / `_IDDM` / `_IDO`). It uses the same thread-detection machinery as the tech support channel, with its own slice of `techsupport_sync_state.json` and its own `TECHSUPPORT_SYNC_INTERVAL_HOURS` gate per channel, so the channels are independent of one another.

**Additive, not a change to the tech support loop.** This pass is a separate scan of separate channels. The tech support channel's own detection, classification and ingest behave exactly as they did before, and turning the pass off with `TECHSUPPORT_SCAN_PRODUCT_CHANNELS=false` leaves the nightly run as it was.

**What the first scan covers.** A product channel with no state yet is seeded to `now - PRODUCT_SCAN_INITIAL_LOOKBACK_DAYS` (default 30) before its first `sync()` call, so that run only sees threads created inside that window rather than the channel's whole history. The seed is written to that channel's slice of `techsupport_sync_state.json` and logged as an INFO line naming the lookback. It happens once: from the second run on, the channel resumes from where the previous run stopped. Widen `PRODUCT_SCAN_INITIAL_LOOKBACK_DAYS` before the first run if you want more backlog picked up, and remember the per-run cap still applies, so a wide window drains over several runs rather than in one.

**Hot and catch-up windows apply per channel.** `TECHSUPPORT_SYNC_HOT_DAYS` (30), `TECHSUPPORT_SYNC_CATCHUP_AGE_DAYS` (90) and `TECHSUPPORT_SYNC_CATCHUP_INTERVAL_DAYS` (7) govern each product channel exactly as they govern the tech support channel: known threads quieter than the hot window are only re-checked on catch-up runs, and threads quieter than the catch-up cap stay in state but are no longer polled. These are per channel, so one busy channel does not shorten another's windows.

**The expert-reply gate comes first.** A thread is only looked at further if some message *other than the thread parent* was posted by a member of that product's expert user group. An expert opening a thread with a question of their own is not a correction. No expert reply means no LLM call at all, counted as `skipped_no_expert_reply`, which is what keeps ordinary product-channel traffic from costing anything.

**The expert-insight gate comes second.** An expert having replied is not by itself a reason to rewrite the knowledge base. One `HasExpertInsight` call reads the role-tagged thread and answers whether any `(expert)` message corrects, confirms, or adds technical insight to the `(bot)` answer or to the topic: a correction, a confirmation that the fix worked, an extra fix, a caveat. An expert's own questions and follow-up questions never count, and neither does small talk. A no is counted as `skipped_no_expert_insight` and nothing else happens: no classification, no ingest, no state flag. Because nothing is recorded, the thread comes back for reconsideration the moment `sync()` sees new reply activity on it, so the correction an expert posts after their question is picked up on the following run.

**Role tags.** A thread that passes the gate is rendered for the prompts with the speaker's role attached: `Lil Lisa (bot)`, `Jane (expert)`, and `Lil Lisa (bot, relaying a user's question)` for the repost the escalate button makes. The bot's contentless `Processing...` placeholders are dropped, and the classifier, summarize and correction prompts all state that `(bot)` content is AI-generated and unverified, and that a later `(expert)` message supersedes it.

**Routing.** Whether the answer in the thread cited a verified entry is what decides. That is recorded at answer time in `techsupport_answer_tags.json` (Section 3), keyed by the Slack thread ts:

1. **Cited entry, expert replied: supersede.** `correct_verified_entry()` rewrites that entry so anything the expert contradicts is removed or replaced, and everything else is kept. Only the "useful" bar applies here, not conclusiveness, the same lighter bar the escalation merge in 4a uses: the topic is already resolved, this thread only has to be worth folding in.
2. **Cited entry no longer in the markdown.** Falls back to a normal add, re-classified against the full useful *and* conclusive bar rather than inheriting the lighter one.
3. **No cited entry.** A plain new QA: full useful and conclusive bar, then `add_verified_qa_pair()`.
4. **Thread already ingested, new expert activity since.** The same operation is repeated: corrected again if last time was a correction (a corrected entry keeps the thread ts of the thread that originally created it, so a replace could not find it), otherwise replaced.

**Supersede versus append merge.** These are deliberately different operations. The escalation path in 4a preserves the existing content and appends the new insight. A correction is the opposite: the point is that the contradicted content must *not* survive. Both keep the entry's title verbatim, which is what keeps its GitHub anchor link stable.

**Provenance.** A supersede appends a record to that entry in `techsupport_review_state.json` under `corrections`: `source_channel_id`, `source_thread_ts`, and `superseded_at`. The escalation enrich path appends a matching record under `enrichments` (same fields, stamped `enriched_at`), so each entry carries the history of every thread that changed it. An entry that was never corrected keeps exactly the shape it had before, so old state files are read unchanged.

**Volume and reporting.** Each product classifies at most `PRODUCT_SCAN_MAX_THREADS_PER_RUN` threads per run, hottest first; the rest are written to that channel's `pending_thread_ids` *before* any work starts, so a deferred thread is still picked up next run even though the sync already moved `last_seen_reply_ts` forward. Results appear under `product_channels` in the run summary, with three new counts, `corrected`, `skipped_no_expert_reply` and `skipped_no_expert_insight`. Per-thread errors ride along in the same admin alert as the tech support loop, prefixed with the product. A failure of the whole pass is alerted and does not fail the rest of the run.

**Prerequisites:** `PRODUCT_CHANNEL_ID_`* set (Section 2), the bot a member of those channels, and the `usergroups:read` scope (Section 6). If no experts can be resolved for a product, that product is skipped with a warning rather than scanned.

## 5 Nightly Pipeline Scheduling

**There is no crontab.** The API process schedules and runs the nightly pipeline itself. Nothing to install on a host, no second container, no external scheduler — deploying the API image is the whole setup.

**The API now refuses to start if the pipeline cannot run.** `main.lifespan` calls `techsupport_cron.run_startup_preflight()` immediately before it starts the tick. If the pipeline's configuration has a BLOCKING problem, the API logs one prescriptive CRITICAL report and aborts startup, so the pod exits non-zero and `kubectl rollout status` fails. Previously the pod looked healthy while every tick failed. This is controlled by `TECHSUPPORT_PREFLIGHT_STRICT` (default `true`). See Section 5e.

The three API dockerfiles `COPY cron /app/cron`, and `LilLisa_Server/pyproject.toml` includes DSPy. `src/techsupport_cron.py` puts that package on `sys.path` and runs the same `run_pipeline()` in-process, and `main.lifespan` starts the tick at boot.

Because `cron/` lives inside the server tree, `paths.py` resolves the server root as its own parent — `/app/cron` → `/app` in the image, and the checkout path on a dev box. No env var is needed for either.

**Build context.** Unchanged (`LilLisa_Server/`), but a new `LilLisa_Server/.dockerignore` is now required: it keeps `.venv`, `speedict/`, and generated state out of the build, which took the context from ~7.3 GB down to ~179 MB (almost all of the remainder is `lancedb/`, deliberately kept because `dockerfile_lancedb` copies it). What each image does and does not carry out of that context differs — see Section 2a before building or deploying `dockerfile_prod`.

### 5a. How the schedule works

The tick fires every `TECHSUPPORT_PIPELINE_TICK_MINUTES` (default 60), starting one interval after boot so it never competes with startup index rebuilds.

That tick is **not** the nightly schedule. Each run calls `is_channel_check_due()`, which does the real gating against `TECHSUPPORT_SYNC_INTERVAL_HOURS` (default 24) using `last_run_timestamp` in `techsupport_sync_state.json`. So an hourly tick still produces one real run per day, ticks that are not due no-op cheaply, and a container restart self-heals on the next tick instead of missing the window. To change the actual cadence, adjust `TECHSUPPORT_SYNC_INTERVAL_HOURS`, not the tick.

If the cron package is somehow missing from the image, the server logs an error at startup and keeps serving; only the nightly work is skipped.

That case deliberately does **not** fail startup, even in strict mode. A missing or unimportable cron package means a broken image, and the fix is to rebuild it, not to mount a Secret. It still shows up: `GET /health/techsupport_pipeline` returns 503 with `"cron_available": false` (Section 5e), so a startup/readiness probe on that path still holds the rollout.

### 5b. Forcing a run by hand

`POST /run_nightly_pipeline/` runs the pipeline immediately, for ops work and post-deploy verification. It uses the same `encrypted_key` JWT as the other admin routes:

```
curl -X POST "https://your-api/run_nightly_pipeline/?encrypted_key=$ENCRYPTED_KEY"
```

`ENCRYPTED_KEY` is the same JWT `lil-lisa` sends: `jwt.encode({"some": "payload"}, AUTHENTICATION_KEY, algorithm="HS256")`.

It returns immediately and runs as a FastAPI background task; progress and the run summary go to the server log. Runs are serialised — if the tick is already running the pipeline, the call is ignored rather than queued, and vice versa. Returns 503 if the image was built without the cron package.

For a one-off run outside the API (debugging, historical imports), the scripts are still directly invocable:

```
cd LilLisa_Server/cron
../.venv/bin/python nightly_pipeline.py
```

Avoid doing that while the API is up — the tick and a manual process are separate OS processes, and the in-process lock cannot see across them.

### 5c. Required volume

Only one volume is needed: the one behind `LANCEDB_FOLDERPATH`. The pipeline's state files (`techsupport_sync_state.json`, `techsupport_reembed_state.json`, `techsupport_review_state.json`), the two tag files the API writes and the pipeline reads (`techsupport_thread_tags.json`, `techsupport_answer_tags.json`), and the verified markdown all live in `techsupport_state/` beside the LanceDB folder, so they persist with it. With `LANCEDB_FOLDERPATH=/data/lancedb` that is `/data/techsupport_state/`. Nothing under `/app` needs a mount.

That default assumes the LanceDB folder's **parent** is on the volume, as with `/data/lancedb` on a volume mounted at `/data`. If the volume is mounted at the LanceDB folder itself, set `TECHSUPPORT_STATE_DIR` to a folder on persistent storage. Without persistent state, every container restart loses the dedup state and the next run re-processes every thread as new. Losing the answer-tags file is not data loss, but a correction posted in a thread whose answer tag is gone is filed as a new entry instead of rewriting the entry the answer actually came from (Section 4c).

**Unwritable directories are now caught at boot.** The startup preflight (Section 5e) checks that `LANCEDB_FOLDERPATH`, the state folder (`TECHSUPPORT_STATE_DIR`, default `techsupport_state/` beside it), and `VERIFIED_TECHSUPPORT_QA_FOLDERPATH` (default inside the state folder) are writable by the server process. It resolves them with the same code the pipeline uses, so it checks the folders that are actually written. A directory that does not exist yet passes if its nearest existing parent is writable, because the pipeline creates it on first use. Previously an unwritable directory only failed the run part-way, after Slack and the LLM had already been paid for. Now it is a BLOCKING problem that names the path, so a read-only root filesystem with no volume mounted fails the rollout instead. The preflight checks **writability only, not persistence**.

**Secrets.** `env/techsupport_sync.env` and `env/github_push.env` live under the cron package (`/app/cron/env/`) and are gitignored, so supply them the same way `passwords/` is supplied in your deployment — mounted, or injected as environment variables (see Section 2a).

Without `techsupport_sync.env`, a missing `SLACK_BOT_TOKEN` or `TECHSUPPORT_CHANNEL_ID` is BLOCKING, so in the default strict mode the API **does not start at all**: you get one `TECHSUPPORT_CONFIG_MISSING` CRITICAL report naming every missing key and the exact mount path (Section 5e). The pre-preflight behaviour — an hourly `Missing required env var(s)` error in a pod that otherwise looked healthy — now only happens with `TECHSUPPORT_PREFLIGHT_STRICT=false`. Without `github_push.env`, a missing `GITHUB_TOKEN` or `GITHUB_REPO_URL` is the same kind of failure: the pod does not start, because the verified markdown would never be pushed and answers would have no clickable source links.

Since the pipeline runs in the same process as the API, `LIL_LISA_SERVER_URL` keeps its `http://127.0.0.1:8000` default — the index reload call is to itself.

**Fixed in this release:** a `LIL_LISA_SERVER_URL` that is present but blank (`LIL_LISA_SERVER_URL=` in `lillisa_server.env`, or an empty value injected by a manifest) now falls back to that default, the same as an absent key. Before, a blank value gave the post-ingest index reload an empty base URL. The reload then failed silently, and freshly ingested entries stayed invisible to live queries until the next restart. No configuration change is needed.

### 5d. Forcing the product-channel scan by hand

`POST /run_product_channel_scan/` runs only the product-channel expert-correction pass (Section 4c): the IDA/IDDM/IDO channels are scanned, and whatever changed is pushed to GitHub and reloaded into the running index. The tech support loop, the review sync and the contextual re-embed are not run. Same `encrypted_key` JWT as the other admin routes:

```
curl -X POST "https://your-api/run_product_channel_scan/?encrypted_key=$ENCRYPTED_KEY"
```

`force` defaults to `true`, which is what an operator triggering a scan by hand normally wants: it bypasses the per-channel `TECHSUPPORT_SYNC_INTERVAL_HOURS` gate, so every configured channel is scanned now instead of being skipped because it was already checked today. Pass `force=false` to respect that gate, i.e. to behave exactly like the nightly run:

```
curl -X POST "https://your-api/run_product_channel_scan/?encrypted_key=$ENCRYPTED_KEY&force=false"
```

Nothing else is bypassed. `TECHSUPPORT_SCAN_PRODUCT_CHANNELS=false` still turns the pass off, `PRODUCT_SCAN_MAX_THREADS_PER_RUN` still caps each product, and threads already ingested are still deduped.

Like `/run_nightly_pipeline/`, it returns immediately and runs as a FastAPI background task, with progress and the run summary in the server log, and returns 503 if the image was built without the cron package. Both endpoints and the tick share one lock: a scan arriving during a nightly run is ignored rather than queued, and so is a nightly run arriving during a scan.

### 5e. Startup preflight and the health probe

`cron/techsupport_preflight.py` validates the techsupport pipeline's configuration once at boot, before the tick starts. The same check runs again on every health-probe call and after every failed tick. Four things about it matter for deployment:

- **It checks the resolved configuration, not whether files exist.** Each env file is read and then overlaid by the process environment, the same way the pipeline's own loaders work (Section 2a). A key injected only as a container env var is valid even when no `.env` file is mounted.
- **It reports every problem at once.** It does not stop at the first one, so one rollout fixes everything instead of one problem per rollout.
- **It is secret-free.** Reports name config keys and filesystem paths only, never a value. A malformed `GITHUB_REPO_URL` is reported without showing the URL, because a URL that fails the credentials check is carrying a token.
- **Every remediation line names the exact Kubernetes fix.** For a missing key in `techsupport_sync.env` it reads: mount Secret key `techsupport_sync.env` at `/app/cron/env/techsupport_sync.env` with `subPath: techsupport_sync.env`, or inject the key into the container environment.

**Severity tiers.** Every problem is one of two tiers. The split is deliberate. Anything that stops ingestion, the GitHub mirror, or the clickable source links fails the rollout. A product-channel scan left unconfigured, or a setting that already has a working default, warns and still boots.


| Tier       | Problems                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Effect                                                                                                    |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `BLOCKING` | Missing `SLACK_BOT_TOKEN` or `TECHSUPPORT_CHANNEL_ID`. Any of `TECHSUPPORT_CHANNEL_ID_IDA` / `_IDDM` / `_IDO` set in `techsupport_sync.env` but not equal to `TECHSUPPORT_CHANNEL_ID`. Missing `AUTHENTICATION_KEY` (the post-ingest index reload cannot authenticate, so new entries never reach live queries). Missing `LANCEDB_FOLDERPATH`. Missing `GITHUB_TOKEN` or `GITHUB_REPO_URL`, or a `GITHUB_REPO_URL` that is **present but malformed** (not `https://`, or with embedded credentials). An unwritable `LANCEDB_FOLDERPATH`, `TECHSUPPORT_STATE_DIR` (default `techsupport_state/` beside the LanceDB folder, Section 5c), or `VERIFIED_TECHSUPPORT_QA_FOLDERPATH`. | Refuses startup when `TECHSUPPORT_PREFLIGHT_STRICT` is on (the default). Health probe returns 503.        |
| `DEGRADED` | Absent or blank `LIL_LISA_SERVER_URL`: falls back to the `http://127.0.0.1:8000` loopback default (Section 5c). Missing `PRODUCT_CHANNEL_ID_IDA` / `_IDDM` or `EXPERT_GROUP_ID_IDA` / `_IDDM` while the product-channel scan is on (the default): that product's expert corrections are not picked up, but the techsupport channel ingest that the bot's answers depend on runs normally. `run_pipeline()` completes the ingest loop before the scan and catches the scan's per-product failures, which is why none of these four keys can block a boot — and they must not, since the scan being on by default would otherwise make four extra keys mandatory everywhere. | Always a warning, in either mode. Never blocks startup. Health probe returns 200. |


`_IDO` keys are never required. `TECHSUPPORT_SCAN_PRODUCT_CHANNELS=false` removes both product-scan checks. The `TECHSUPPORT_QA_PAIRS` table, `techsupport_qa_pairs.md`, and the state JSON files are **not** required: the first successful run creates them, so a correctly configured new deployment reports zero problems.

**What happens at boot.**


| Preflight result                 | `TECHSUPPORT_PREFLIGHT_STRICT=true` (default)                                                                                                | `TECHSUPPORT_PREFLIGHT_STRICT=false`                                                                             |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Any BLOCKING problem             | Full report at CRITICAL, then `RuntimeError: TECHSUPPORT_CONFIG_MISSING: refusing to start ...`. Uvicorn aborts startup, the pod exits non-zero, and the rollout fails. | Full report at ERROR, ending `Starting anyway because TECHSUPPORT_PREFLIGHT_STRICT=false`. Serves traffic; the pipeline stays broken. |
| DEGRADED problems only           | Full report at WARNING. Serves traffic normally.                                                                                             | Same.                                                                                                            |
| No problems                      | One INFO line: `techsupport pipeline preflight passed: no configuration problems found.`                                                     | Same.                                                                                                            |
| Cron package failed to import    | Not a boot failure (broken image, see Section 5a). The scheduler logs it at ERROR; the probe returns 503.                                     | Same.                                                                                                            |


**Log codes for alerting.** These two strings are stable on purpose. Match log alerts on them, not on the surrounding wording.


| Code                          | Emitted when                                                                                                                                                                                                                                         | Suggested routing                                    |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `TECHSUPPORT_CONFIG_MISSING`  | At least one BLOCKING problem. It starts the boot report (CRITICAL, or ERROR when strict is off) and the `RuntimeError`. It also appears in the CRITICAL record when a tick fails and the preflight finds a BLOCKING problem at that moment (see "After boot"). | Page. The pipeline cannot ingest, push the verified markdown, or attach source links. |
| `TECHSUPPORT_CONFIG_DEGRADED` | Every problem is DEGRADED. It starts the boot WARNING report.                                                                                                                                                                                        | Ticket, not a page. A product-channel scan is unconfigured, or a setting is using its working default. |


Every report starts `<code>: techsupport pipeline preflight failed (N problems: B blocking, D degraded).`, or `found degraded configuration` instead of `failed` when nothing is BLOCKING. Numbered `[BLOCKING]` / `[DEGRADED]` lines follow, each with a `Fix:` line, BLOCKING first.

**Health probe: `GET /health/techsupport_pipeline`.** This endpoint is unauthenticated, unlike the other techsupport routes: Kubernetes probes cannot practically present the `encrypted_key` JWT. That is safe only because the body is secret-free by construction. Every field is a code, a severity, a key name, a path, a count, or a boolean. It re-runs the preflight on every call, with no caching and no network, so it reflects the configuration the pod can see right now.

- **503:** any BLOCKING problem, or the cron package failed to import (`"cron_available": false`).
- **200:** DEGRADED-only or clean configuration, **and also** any BLOCKING problem when `TECHSUPPORT_PREFLIGHT_STRICT=false`.

That last exception is what keeps the escape hatch usable. Under strict mode, a BLOCKING problem present at boot means the process already refused to start, so this route was never reachable to answer at all — the only way to be serving traffic *with* a known BLOCKING problem is to have explicitly opted out. Returning 503 there would let the startupProbe below restart-loop the very pod the operator chose to bring up degraded. The body still reports `"ok": false` either way, so **alert on the body, not the status code** if you want to catch that case.

What the 503 is genuinely for is the case strict mode cannot cover: because the preflight re-runs per request, configuration that regresses **after** boot (a Secret rotated out from under a running pod) is caught here and correctly fails readiness, where nothing else would have noticed.

Body keys: `ok`, `strict`, `code` (`TECHSUPPORT_CONFIG_MISSING`, `TECHSUPPORT_CONFIG_DEGRADED`, or `null`), `blocking_count`, `degraded_count`, `problems` (each with `severity`, `keys`, `problem`, `remediation`, `path`), `sources` (the three env-file paths consulted), `cron_available`, `running`, `last_run_failed`, `last_run_failure_code`. `last_run_failed` is `null` until the first tick or `/run_nightly_pipeline/` call in this process finishes. `last_run_failure_code` is `TECHSUPPORT_CONFIG_MISSING` or `null`, never exception text. When the cron package is unavailable, the body also has `"ok_reason": "cron package unavailable"`, with `strict` and `code` set to `null`.

Use it as a **startupProbe and/or readinessProbe**, never a livenessProbe. A misconfigured pod is alive: it keeps answering unrelated requests while the pipeline is dead. A liveness failure would just restart a pod that is alive but permanently misconfigured, and the restarted pod comes back with exactly the same configuration. That achieves nothing except a restart loop and noisier logs. Keep any existing livenessProbe on whatever path it uses today.

Ready-to-paste block for the API container (uvicorn listens on 8000 in these images; adjust `port` if yours differs):

```yaml
startupProbe:
  httpGet:
    path: /health/techsupport_pipeline
    port: 8000
  periodSeconds: 10
  timeoutSeconds: 5
  # The server only starts listening after lifespan finishes, which on a cold
  # volume includes rebuilding the LanceDB indices. Size this budget
  # (periodSeconds x failureThreshold = 15 minutes here) to your slowest boot.
  failureThreshold: 90
readinessProbe:
  httpGet:
    path: /health/techsupport_pipeline
    port: 8000
  periodSeconds: 30
  timeoutSeconds: 5
  failureThreshold: 3
```

With strict mode on, a BLOCKING pod exits before it ever listens, so the startupProbe adds nothing in that case. Its value is in the cases strict mode cannot see: a broken image (`cron_available: false`), and configuration that regresses after boot. In both, the pod boots but the probe returns 503, and once `failureThreshold` is used up the kubelet restarts the container, so the rollout does not complete. It deliberately does **not** catch `TECHSUPPORT_PREFLIGHT_STRICT=false`, which returns 200 by design — see the 503-vs-200 rule above.

Understand the readinessProbe trade-off before applying it. If a BLOCKING problem appears **after** boot, for example a volume going read-only or an env file disappearing from a non-`subPath` mount, the pod is taken out of its Service endpoints. Lil Lisa then stops being served by that pod, even though only the pipeline is broken. On a single-replica StatefulSet that means no backend at all. If you would rather keep serving and only alert, keep the `startupProbe` (it gates the rollout), point the `readinessProbe` at your existing readiness path, and alert on `TECHSUPPORT_CONFIG_MISSING` in the logs instead.

**After boot.** A clean boot can still regress later. The main way this happens is a rotated Secret: a Secret mounted with `subPath` is **not** re-delivered into a running pod, so the pod keeps the old file until it is restarted. **After rotating any of these Secrets, restart the pod** (`kubectl rollout restart`). `run_once()` is the runtime safety net. When a tick fails, it re-runs the preflight. If that finds a BLOCKING problem, it logs the same prescriptive report at CRITICAL, prefixed with a note that the problem may have been introduced after startup, and it carries `TECHSUPPORT_CONFIG_MISSING`. To avoid hourly spam, the full report is logged only when a given set of BLOCKING problems first appears or changes. Identical repeats get a one-line CRITICAL reminder that still starts with the code, so a log alert keeps firing. Any successful run resets this. Note that the preflight checks that keys are **present**, not that they are **valid**. A stale or revoked token that is still in the old mounted file passes the preflight. That tick failure is then logged as a plain `Nightly techsupport pipeline failed: ...` CRITICAL with a traceback and **no** code, and the probe stays 200 with `"last_run_failed": true` and `"last_run_failure_code": null`. Alert on that message too.

## 6 Slack App Configuration

One new scope is needed: `usergroups:read`. Everything else in the escalation flow and the nightly scripts is covered by the existing bot token and scopes (`app_mentions:read`, `chat:write`, `im:write`, `channels:history`, `channels:read`, `im:history`).

Event Subscriptions also need to be turned on alongside the bot token scopes above. Go to the Slack app's Event Subscriptions page, enable it, and subscribe to `message.channels` and `app_mention` under bot events. If you skip this, the bot will connect fine and look healthy, but it won't receive any messages at all. This applies any time you're setting up a brand new Slack app.

`usergroups:read` lets the bot call `usergroups.users.list` to resolve the per-product expert user groups (`EXPERT_GROUP_ID_IDA` / `_IDDM` / `_IDO`, Section 2). Add the scope and reinstall the app **before** deploying: without it every expert lookup fails loudly (`ExpertLookupError` out of the Slack handlers, an errored product in the nightly pass) instead of quietly treating everyone as a non-expert.

### 6a. Expert user groups: what to set up


| Question                                               | Answer                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Does the bot need a new permission?                    | **Yes, one.** Add the bot token scope `usergroups:read` under OAuth & Permissions in the Slack app settings, then reinstall the app to the workspace so the bot token picks it up. It is a read-only scope. No other scope changes are needed.                             |
| Does the bot need to be a member of the expert groups? | **No.** User groups are workspace-level objects, and a bot token with `usergroups:read` can read any group's member list without belonging to it. User groups only hold human members, so there is nothing to add the bot to.                                              |
| What does the bot do with the group?                   | Exactly one read-only call: `usergroups.users.list` for the configured group ID, cached for `EXPERT_GROUP_CACHE_SECONDS`. It never posts to the group, never mentions it, and never changes its membership. The group is only a lookup table for "is this user an expert". |
| What value goes in `EXPERT_GROUP_ID_*`?                | The group's **ID**, which starts with `S` (for example `S0123ABCD`), **not** its `@handle`. Find it on the group's page under People & user groups in Slack, or from the `usergroups.list` API method. A handle in this variable fails the lookup.                         |
| Which Slack plan?                                      | User groups are a paid-plan feature. They are not available on the free tier.                                                                                                                                                                                              |


The bot doesn't have `channels:join`. If it's ever removed from the admin channel, or the admin channel changes someone will need to manually invite it back.

For the real production tech support channel, the bot needs to actually be a member of that channel for the nightly scripts to read its history, or `conversations.history` and `conversations.replies` will fail with `not_in_channel`. The same holds for the product channels in `PRODUCT_CHANNEL_ID_*`, which the expert-correction scan reads with those same two calls. The bot answers questions in them already, so it is already a member; there is nothing to do unless it gets removed.

## 7 Private GitHub Repo

This is now live and actively syncing. The verified techsupport markdown file gets pushed automatically after every nightly update to a dedicated private repo. Every answer that draws on this content includes a working link directly to the relevant section of the file, using GitHub's heading-anchor links, so users can click through and read the full context.

This repo currently lives under my personal GitHub account, matching the existing golden QA pairs repo (`drawal1/rocknbot_qa_pairs`), not a company-owned org account. This can be moved if needed.

### 7a. `github_push.env` setup (devops)

`github_sync.py` (called from `nightly_pipeline.py`) clones that private repo over HTTPS and pushes `techsupport_qa_pairs.md` if it changed. **Same two variables as before** — there is no new env var and no SSH deploy key.

1. Copy the example and fill it in (this file is gitignored):

```
cd LilLisa_Server/cron/env
cp github_push.env.example github_push.env
```

1. Set:


| Variable          | What to put                                                                                                                                                     |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GITHUB_TOKEN`    | A GitHub PAT that can push to the repo in `GITHUB_REPO_URL`. Fine-grained: **Contents: Read and write** on that repo. Classic: `repo` scope for a private repo. |
| `GITHUB_REPO_URL` | Plain HTTPS clone URL only, e.g. `https://github.com/org/rocknbot_techsupport_qa_pairs.git`.                                                                    |


1. **Do not put the token in the URL.** These are wrong and will be rejected (or would leak the PAT into `.git/config` / git error logs):

```
# BAD
GITHUB_REPO_URL=https://ghp_....@github.com/org/repo.git
GITHUB_REPO_URL=git@github.com:org/repo.git
```

The script injects credentials via `GIT_ASKPASS` (a short helper that prints `$GITHUB_TOKEN` when Git asks for a username/password). GitHub accepts the PAT as that password. The clone URL logged at INFO is the clean `https://github.com/...` URL with no secret.

1. **Either the file or the process environment works.** `github_sync.load_env()` reads `github_push.env` and then overlays `os.environ`, so a container/k8s secret wins over an empty placeholder in the file. Supplying both is fine; supplying neither raises `Missing required env var(s) [...] - expected in <path> or the process environment`.
2. No extra Git config is needed. The image needs `/bin/sh`, which the `python:3.11-slim` base provides. A user-level `credential.helper` will not persist this PAT: that clone passes `credential.helper=` so the token is not written to `~/.git-credentials`.
3. Smoke-check after deploy. Inside the running container (no virtualenv there — dependencies are installed system-wide with `uv pip install --system`):

```
cd /app/cron
python github_sync.py
```

From a dev checkout instead:

```
cd LilLisa_Server/cron
../.venv/bin/python github_sync.py
```

Unchanged markdown prints `{'pushed': False, 'reason': 'unchanged'}`. A real change prints `pushed: True` and a commit message. Logs should show `Cloning https://github.com/...` **without** a token in the URL.

**Fixed in this release:** that clone now also passes `allow_unsafe_options=True`. GitPython classes `--config` as an unsafe clone option and raised `UnsafeOptionError` without it, which failed every nightly push. No configuration change is involved.

**If push fails:** GitHub 401 / `Authentication failed` almost always means the PAT is expired, lacks Contents write, or `GITHUB_REPO_URL` points at the wrong repo. Rotate the token in `github_push.env` (or the process environment), not in a clone URL. You should not need to change Git's global config.

### 7b. Expert thumbs-up pairs now push to the golden QA repo

Previously, a `POST /add_expert_qa_pair/` (the expert thumbs-up in a product channel) only wrote the pair into the `{PRODUCT}_QA_PAIRS` LanceDB table. `/update_golden_qa_pairs/` **drops that table** and rebuilds it from the markdown in `QA_PAIRS_GITHUB_REPO_URL`, so every expert pair added since the last rebuild was silently lost. `src/golden_qa_sync.py` now appends each verified pair to `{product}_qa_pairs.md` in that repo and pushes it, so the rebuild stays a pure function of the repo.


| Item           | Detail                                                                                                                                                                                                                                                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Repo           | `QA_PAIRS_GITHUB_REPO_URL` from `env/lillisa_server.env` (the golden QA pairs repo). Plain HTTPS clone URL only — a URL with an embedded token, or an SSH/scp-style URL, is rejected.                                                                                                                                          |
| Token          | The same `GITHUB_TOKEN` from `cron/env/github_push.env` that pushes the verified techsupport markdown. It must also be allowed to push to this repo. Read from that file, overlaid by the process environment. |
| Auth mechanism | Same as section 7a: `GIT_ASKPASS` prints the PAT, `credential.helper=` on the clone so nothing is persisted to disk, and the clone is shallow (`depth 1`).                                                                                                                                                                     |
| Commit         | `Add expert-verified QA pair (IDDM) - <date>`, authored as `LilLisa Expert QA <noreply@radiantlogic.com>`.                                                                                                                                                                                                                     |


**The push cannot fail the request.** The LanceDB insert happens first and always stands; the push runs after it in a worker thread and never raises. The endpoint's JSON response now carries a `"pushed": true|false` field, and a failed push is logged as `Expert QA pair stored in LanceDB but NOT pushed to the golden QA repo ...` with the underlying error. `pushed: false` means the pair answers queries now but will disappear at the next golden QA rebuild — fix the token/URL and re-add it, or use the backfill script below.

**Backfill (pairs verified before this change).** Those pairs are recoverable from the server log, which records each one as `Expert QA Verification: {json}`:

```
cd LilLisa_Server
PYTHONPATH=. .venv/bin/python scripts/backfill_expert_qa_pairs.py --log-file /path/to/server.log --dry-run
PYTHONPATH=. .venv/bin/python scripts/backfill_expert_qa_pairs.py --log-file /path/to/server.log
```

It dedupes repeated log lines, skips pairs already present in the repo file (so re-running after a partial failure is safe), and prints a JSON summary (`found` / `skipped_existing` / `pushed` / `failed`). `--dry-run` changes nothing; `--product IDDM` (repeatable) limits it to one product. In the container use `python` instead of `.venv/bin/python`.

## 8 Rollback Procedure

The verified techsupport LanceDB table (`TECHSUPPORT_QA_PAIRS`) keeps full version history. Every re-embed or bulk write creates a new version instead of overwriting in place. If something goes wrong, like a bad re-embed or corrupted data, you can check what's available.

Run this in the deployed container, since that is where the LanceDB volume is mounted (`python`, not `../.venv/bin/python` — the image installs dependencies system-wide):

cd /app/cron

python c "from techsupport_rollback import list_available_versions; list_available_versions()"

Then roll back to a specific version:

python c "from techsupport_rollback import rollback_to_version; rollback_to_version(N)"

On a dev checkout the same commands work with `cd LilLisa_Server/cron` and `../.venv/bin/python`.

This is non-destructive. Rolling back creates a new version rather than deleting anything, so you can always move forward or backward again afterward.

Now that the GitHub repo in Section 7 is live, the markdown file itself also has real version history through normal git commits, so both the database content and the source file can be inspected or reverted independently if needed.

### 8a. Restore markdown + review_state with LanceDB

`techsupport_rollback.py` only restores the LanceDB table (`TECHSUPPORT_QA_PAIRS`). It does **not** auto-restore files from Lance versions.

The verified techsupport markdown, `techsupport_review_state.json`, and LanceDB must be restored **together** from the same point in time. Rolling the table back without the matching files will drift: review_state `node_ids` will not match table rows, and the markdown source will not match what retrieval serves.

After `rollback_to_version(N)`, the script prints a loud reminder listing the other files. Restore them yourself:

1. LanceDB `TECHSUPPORT_QA_PAIRS` — this script (`rollback_to_version(N)`).
2. `techsupport_qa_pairs.md` — under `VERIFIED_TECHSUPPORT_QA_FOLDERPATH` (by default `/data/techsupport_state/verified_techsupport/techsupport_qa_pairs.md`). Use git history in the Section 7 repo (or a backup) from the same moment as LanceDB version N.
3. `techsupport_review_state.json` in the techsupport state folder (`/data/techsupport_state/` by default) — gitignored; restore from a backup/copy taken at that same moment. There is no Lance version of this file.

