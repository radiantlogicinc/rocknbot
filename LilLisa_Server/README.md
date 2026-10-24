# LilLisa Server
## Description

Built using FastAPI, LilLisa Server is an AI question-and-answer program that enables users to ask natural language questions and receive insightful answers based on their personalized data.

LilLisa Server uses LanceDB as the vector database, providing blazingly fast retrieval times and LlamaIndex to handle complex queries.

LilLisa Server is able to grow its knowledge base and get better at answering user questions as more documentation is added and conversations are stored.

## Visuals

[Conversation within Slack](./visuals/conversation_slack.png)

[Conversation within Browser](./visuals/conversation_web.png)

## Installation

- Clone this project using this command:
  - git clone https://oauth2:&lt;YOUR_GITLAB_ACCESS_TOKEN&gt;@gitlab.com/radiant-logic-engineering/rl-datascience/lil-lisa.git
- Navigate to LilLisa_Server folder
- In the terminal, run `make setup-env`. This creates a uv virtual environment (`.venv`) and installs all dependencies
- Select `.venv` (LilLisa_Server/.venv/bin/python) as the Python interpreter in your IDE

## Usage

<ins>IMPORTANT<ins>: If using VS Code, start main.py by using "Python Debugger: Debug using launch.json"
​
Integrate with Slack, FastHTML, or another application that handles user input. Run one of the above along with LilLisa_Server conccurently.

Slash commands are encrypoted and can only be used by admins specified in Slack.

Methods free to use are:

**/invoke**

Uses a session ID to retrieve past conversation history. Based on a query, it searches relevant documents in the knowledge base and retrieves multiple fewshot examples from the QA pairs database to help synthesis of a formatted answer. Queries are handled differently and depend on whether an expert is answering or not. ReACT agent handles the use of information given to intellignetly craft an answer. This endpoint is primarily used for the LilLisa Slack bot integration.

**/invoke_stream_with_nodes/**

Streams the Chain of Thought (CoT) and Answer (ANS) in HTML format, along with the top relevant nodes from the knowledge base. This endpoint is designed for the web interface, allowing for a streaming response that shows the reasoning process and final answer incrementally. Like the invoke endpoint, it uses session history and ReACT agent to craft responses, but returns the data as a streaming response with HTML formatting for better web display.

**/record_endorsement**

Records an endorsemewnt, usually given when an answer is correct, by either a "user" or "expert". This is helpful when admins call the 'get_conversations' method and use it to create more golden QA pairs.

Administrative methods requiring JWT encryption key:

**/get_golden_qa_pairs/**

Retrieves golden QA pairs for a specified product from a GitHub repository. This endpoint requires authentication via JWT token to access the QA pairs data. The response is a Markdown file containing the QA pairs for the requested product.

**/update_golden_qa_pairs/**

Initiates the update of golden QA pairs in LanceDB for a specified product in the background. This endpoint triggers a process that clones the latest QA pairs from the repository and updates the vector database. Changes take approximately 2 minutes to become effective.

**/rebuild_docs_traditional/**

Initiates a complete rebuild of the documentation database using traditional OpenAI chunking with text-embedding-3-large for both documents and QA pairs. This is an administrative function that can take up to an hour to complete as it rebuilds all document and QA pair indices.

**/rebuild_docs_contextual/**

Initiates a complete rebuild using contextual Voyage chunking with voyage-context-3 for both documents and QA pairs. This endpoint provides an alternative embedding model for potentially improved semantic search capability on contextual queries.

**/cleanup_sessions/**

Deletes session folders under SPEEDICT_FOLDERPATH that are older than the configured retention period. This endpoint requires the environment variable `SESSION_LIFETIME_DAYS` (e.g., `SESSION_LIFETIME_DAYS = 30`) to determine which sessions to remove based on their age. This helps manage disk space by removing outdated conversation histories.

**/add_expert_qa_pair/**

Adds an expert-verified question and answer pair to the golden QA database. Called by the Slack bot when an expert reacts with a thumbs up to one of Lil Lisa's answers. The pair is inserted into the product's LanceDB QA pairs table and is also appended and pushed to the golden QA pairs repository, so it survives the next `/update_golden_qa_pairs/` rebuild (which drops the table and rebuilds it from that repo). The JSON response carries a `pushed` field: `false` means the pair answers queries now but did not reach the repo, so it would be lost at the next rebuild.

**/tag_techsupport_thread/**

Records that a newly created tech support escalation thread relates to an existing verified tech support entry, so the nightly pipeline merges the new insight into that entry instead of adding a near duplicate. Stored in `techsupport_thread_tags.json` in the tech support state folder, which the pipeline only reads.

**/reload_techsupport_qa_pairs/**

Rebuilds the in-memory QA pairs retrievers and indices (including `TECHSUPPORT_QA_PAIRS`) from the current on-disk LanceDB tables. The nightly pipeline writes new verified entries straight to LanceDB, so without this call a long-running server keeps serving from the retrievers it built at startup.

**/run_nightly_pipeline/**

Runs the nightly tech support pipeline immediately, as a background task, for ops work and post-deploy verification. It returns right away and the run summary goes to the server log. Overlapping runs are dropped rather than queued, and it returns 503 if the image was built without the `cron` package.

**/run_product_channel_scan/**

Runs only the product-channel expert-correction pass immediately, as a background task: the IDA/IDDM/IDO channels are scanned for expert replies, and whatever changed is pushed to the verified entries repo and reloaded into the running index. The tech support loop, the review sync and the contextual re-embed are not run. The optional `force` parameter defaults to true, which bypasses the per-channel check interval so every configured channel is scanned now; pass `force=false` to respect it. Shares one lock with the nightly pipeline, so overlapping runs are dropped rather than queued.

## Nightly techsupport pipeline

`cron/nightly_pipeline.py` ships inside the server image and the API process runs it on a timer, so there is no crontab and no second container. Each run first scans the one tech support channel (`TECHSUPPORT_CHANNEL_ID`, the same variable the Slack bot uses) for threads that reached a resolution and turns them into verified entries in `techsupport_qa_pairs.md` plus LanceDB. It then scans the IDA/IDDM/IDO product channels (`CHANNEL_ID_IDA` / `_IDDM` / `_IDO` from `lil-lisa.env`, injected into this process's environment in the API container) for threads where a member of that product's expert user group replied to one of Lil Lisa's answers. A thread with no expert reply never reaches an LLM call. If the answer in the thread cited an existing verified entry, the expert's reply rewrites that entry, keeping its title so its GitHub link stays valid; otherwise the thread is added as a new entry.

Everything the pipeline writes lives in one state folder, `techsupport_state/`, beside the LanceDB folder: `techsupport_sync_state.json` (per channel), `techsupport_review_state.json`, `techsupport_reembed_state.json`, the two tag files the API writes and the pipeline reads (`techsupport_thread_tags.json` and `techsupport_answer_tags.json`), and `verified_techsupport/techsupport_qa_pairs.md`. With `LANCEDB_FOLDERPATH=/data/lancedb` that is `/data/techsupport_state/`, on the same persistent volume as the table, with no extra config. `TECHSUPPORT_STATE_DIR` and `VERIFIED_TECHSUPPORT_QA_FOLDERPATH` override the folder and the markdown location; neither is required. Full ops detail, including the environment variables involved, is in "PR42 Release and Deployment Notes.md" at the root of this repo.

## Contributing

The project is not currently open for contributions.

### Requirements
- Docker container
- Python 3.11.9 
- RAM: 1.0 GB
- Size of Docker container: 23.6 GB

### Docker GPU Access
For GPU-accelerated reranking: use `docker run --gpus all ...` (or equivalent in your orchestration) so the LilLisa_Server container can use the GPU.

### Pushing to Cloud

For assistance with deploying to AWS Lambda, refer to this blog:
  - https://fanchenbao.medium.com/api-service-with-fastapi-aws-lambda-api-gateway-and-make-it-work-c20edcf77bff

## Support

Reach out to us if you have questions:
- Dhar Rawal (Slack: @Dhar Rawal, Email: drawal@radiantlogic.com)

## Authors and acknowledgment

- Carlos Escobar
- Dhar Rawal
- Unsh Rawal
- Nico Guyot
- Priyanshu Jani

## License

This project is currently closed source

## Project status

Under active development

## Socials
- [Link to Medium.com blog](https://medium.com/@carlos-a-escobar/deep-dive-into-the-best-chunking-indexing-method-for-rag-5921d29f138f)