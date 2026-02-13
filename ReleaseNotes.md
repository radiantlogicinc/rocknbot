# Release Notes

## Important Notes for This Release

**Current Production Versions:**
- lil-lisa: 2.2.4
- lillisa_server: 2.5.2
- lil-lisa-web: 2.3.4

**Changed Production Version:**
- lil-lisa: 2.3.2 (includes changes from undeployed 2.3.0, 2.3.1 + new features)
- lillisa_server: 2.6.2 (includes changes from undeployed 2.6.0, 2.6.1 + new features)
- lil-lisa-web: 2.4.1 (includes changes from undeployed 2.4.0 + new features)

**Note:** This release includes all changes from the previously submitted but undeployed releases.

- LDB_TAG Changed from 2.5.3 to 2.6.0 (Is this necessary?)

## Changes in lillisa_server 2.6.2

### GPU-Accelerated Reranking
- Added PyTorch with CUDA support as a new dependency (`pytorch=2.6.0`, `pytorch-cuda=12.6` via conda; `torch` via pip)
- The reranker (`cross-encoder/ms-marco-MiniLM-L-12-v2`) now automatically detects and uses GPU if available, falling back to CPU

### Replaced ReAct Agent with Deterministic Retrieval Pipeline
- Removed the ReAct agent from both `/invoke/` and `/invoke_stream_with_nodes/` endpoints
- Replaced with a deterministic pipeline: conditionally improve query → retrieve documents → synthesize answer
- Eliminates non-deterministic agent behavior and the "Reached max iterations" fallback path
- LLM answer generation switched from streaming to a single synchronous completion call

### Parallel Retrieval & Cached Embeddings
- Document retrieval and QA pairs retrieval now run concurrently via `ThreadPoolExecutor`
- New `CachedQueryEmbedding` wrapper pre-computes and caches query embeddings, avoiding duplicate API calls when multiple retrievers process the same query
- `similarity_top_k` reduced from 50 to 20 for document retrieval; reranker `top_n` reduced from 50 to 10

### Streaming Thinking Messages
- While the pipeline processes in the background, humorous placeholder messages (e.g., "Crunching bits...", "Consulting oracles...") are streamed to the client as `COT:` events

### IDO Product Support Made Optional
- IDO LanceDB tables, documentation versions, and product versions are now loaded with graceful fallback
- Server starts cleanly without IDO configuration, logging warnings instead of raising fatal errors
- The `/invoke/` and QA pair endpoints dynamically include IDO only when configured
- Background tasks (golden QA pair updates, doc rebuilds) skip IDO when not configured

### Performance Logging
- Added comprehensive `PERF |` timing markers throughout the pipeline (embedding, retrieval, reranking, LLM calls, context save, etc.)

## Changes in lil-lisa 2.3.2

### IDO Slack Channel Configuration Made Optional
- IDO channel IDs (`CHANNEL_ID_IDO`, `ADMIN_CHANNEL_ID_IDO`, `EXPERT_USER_ID_IDO`) are now loaded via `.get()` with graceful fallback to `None`
- A warning is logged if `CHANNEL_ID_IDO` is not configured
- All admin authorization checks and product determination now guard against missing IDO configuration

## Golden QA Pairs Repository Update
Create a new `file ido_qa_pairs.md` in the golden QA pairs repository with the following sample data:
``` 
# Question/Answer Pair 1

Question: What is IDO?

Answer: IDO (Identity Data Orchestration) is RadiantLogic's solution for managing identity data across multiple systems.
```


## Pre-Deployment Action
Delete the existing LanceDB datastore before deploying this release. A fresh datastore with all documents will be automatically created and populated upon deployment.

**Note**: This process takes approximately 2 hours to complete. There is no notification when the rebuild is complete.

## New environment variables
### Server Configuration (lillisa_server.env)
1. Add the following environment variables:
    - `DOCUMENTATION_IDO_VERSIONS="v1.0"` *(optional — IDO functionality is disabled if not set)*
    - `IDO_PRODUCT_VERSIONS="dev/v1.0, v1.0"` *(optional — IDO functionality is disabled if not set)*

#### Slack Configuration (lil-lisa.env)
1. Create two new Slack channels:
    - `lil-ido`
    - `lil-ido-admin`
2. Add the following environment variables *(all optional — IDO Slack support is disabled if not set)*:
    - `CHANNEL_ID_IDO` - Set this to the channel ID of `lil-ido`
    - `ADMIN_CHANNEL_ID_IDO` - Set this to the channel ID of `lil-ido-admin`
    - `EXPERT_USER_ID_IDO` - Set this to the appropriate expert user ID

## New Dependencies (lillisa_server)
- **PyTorch 2.6.0 with CUDA 12.6** — Required for GPU-accelerated reranking. Installed via conda channels `pytorch` and `nvidia`, or via pip with the CUDA wheel index (`pip install torch --index-url https://download.pytorch.org/whl/cu126`).

### Docker GPU Access
For Docker: use `docker run --gpus all ...` (or equivalent in your orchestration) so the LilLisa_Server container can use the GPU for reranking.

## Standard Deployment Process

### Deployment

Deploy only those applications where the version has changed:

- lil-lisa
- lillisa_server
- lil-lisa-web
 
### Testing (Wait approximately 2 hours after deployment)

1. Run a query and verify that citations are sourced from the developer portal, not GitHub
2. Test the new IDO product by running a query in the `lil-ido` channel (if IDO is configured)
    - Example query: "In Graph Pipelines Configuration can you explain me Structure of a configuration."
    - Expected output should include: Source Objects, Vertices, Edges, Functions
3. Verify performance logging — check server logs for `PERF |` entries to confirm timing is being captured
