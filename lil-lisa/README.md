# Lil Lisa
## Description

Lil Lisa is an application that is responsible for integrating with Slack and handling incoming user events. It acts as a bridge between Slack's platform and the conversational AI experience provided by API calls to a concurrently-running FastAPI application.  

Lil Lisa provides several slash commands for admin use to help improve the knowledge base.

## Visuals

[Simple conversation](./visuals/simple_conversation.png)

[Architecture/Diagram](./visuals/diagram.png)

## Installation

Simply add Lil Lisa to your Slack workspace

1. Open Slack
2. In the bottom left, under Apps, click 'Add apps'
3. Search for "rocknbot" and select it.

[Click here if you need further instruction on adding an app to Slack](https://slack.com/help/articles/202035138-Add-apps-to-your-Slack-workspace)

Setup Lil Lisa Server and have both applications running at the same time.
​
## Usage

For use in a channel, simply call **@rocknbot** in the channel, and it will return with *'Processing...'* and subsequently the answer to the question.

For IDDM queries, refer to the "lil-elvis" channel.
For IDA queries, refer to the "lil-lisa" channel.

Experts are everyone in the product's Slack user group (`EXPERT_GROUP_ID_*`), so expert handling applies to every member rather than to one configured person: a thumbs up on one of Lil Lisa's answers turns that question and answer into a golden QA pair, which is now also pushed to the QA pairs repo so it survives the next rebuild, and the confirmation DM goes to whoever reacted.

An expert who wants to correct an answer just replies in that thread, with no command or prefix. LilLisa Server's nightly pipeline scans the product channels for those replies and either rewrites the knowledge base entry the answer came from or files the thread as a new one.

The slash commands available for admin use include:

**/get_golden_qa_pairs**

Retrieves QA pairs stored in the appropriate github repository

**/update_golden_qa_pairs**

Ingests changes made to the QA pairs github repository by deleting the previous LanceDB tables and recreating with up-to-date data

**/run_nightly_pipeline**

Starts the backend's techsupport pipeline in the background. It collects and classifies new techsupport threads, syncs expert corrections, reloads changed techsupport entries, and pushes changed verified markdown to GitHub. It is admin-only, returns immediately, and reports progress and the final summary in the backend logs.

**/get_conversations [endorsed_by]**

Retrieves conversations from a local folder that were endorsed by [endorsed_by]

**/rebuild_docs_traditional**

Ingests changes made to the knowledge base (github repository) by deleting the previous LanceDB tables and recreating with up-to-date data using traditional chunking with OpenAI text-embedding-large-3

**/rebuild_docs_contextual**

Ingests changes made to the knowledge base (github repository) by deleting the previous LanceDB tables and recreating with up-to-date data using contextual chunking with Voyage voyage-context-3

**/cleanup_sessions**

Removes old session data based on the configured retention period to free up storage space and maintain system performance

## Contributing

The project is not currently open for contributions.

### Requirements

- Docker container
- Python 3.11.9
- RAM: 0.4 GB
- Size of Docker container: 1.4 GB

### Setup dev environment

- Clone this project using this command:
  - git clone https://oauth2:&lt;YOUR_GITLAB_ACCESS_TOKEN&gt;@gitlab.com/radiant-logic-engineering/rl-datascience/lil-lisa.git
- Navigate to lil-lisa folder
- In the terminal, run `make setup-env`. This creates a uv virtual environment (`.venv`) and installs all dependencies
- Select `.venv` (lil-lisa/.venv/bin/python) as the Python interpreter in your IDE

### Environment Configuration

The application uses several environment variables configured in `app_envfiles/lil-lisa.env`:

- **MAX_LENGTH**: Controls the maximum length of messages sent to Slack (e.g., `MAX_LENGTH = 4000`). Messages exceeding this limit will be truncated to ensure proper formatting and delivery.
- **EXPERT_GROUP_ID_IDA / _IDDM**: Required. Slack **user group** IDs (e.g. `S0123ABCD`, from `usergroups.list`, not the `@handle`) whose members count as experts for that product. This is the only source of expert identity, so the bot refuses to start if either is missing. The Slack app needs the `usergroups:read` scope; without it expert lookups fail loudly rather than treating everyone as a non-expert.
- **EXPERT_GROUP_ID_IDO**: Optional, for the IDO product. With no group, nobody is an IDO expert.
- **EXPERT_GROUP_CACHE_SECONDS**: How long expert group membership is cached before Slack is asked again, in seconds (default `300`). A failed refresh keeps serving the last cached membership; with nothing cached it raises.
- Other Slack API tokens and channel configurations as required

### Deployment Instructions

**Important for Deployment Teams:**

The application now supports two different document chunking strategies:

1. **Traditional Chunking** (Default): Uses OpenAI text-embedding-3-large
2. **Contextual Chunking**: Uses Voyage voyage-context-3

**Initial Deployment Behavior:**
- When the application is freshly deployed without an existing document store, it will automatically create the documentation database using **traditional chunking** as the default method.

**Switching Chunking Strategies:**
After deployment, you can switch between chunking strategies using slash commands:
- Execute `/rebuild_docs_traditional` to rebuild the knowledge base using traditional chunking with OpenAI text-embedding-3-large
- Execute `/rebuild_docs_contextual` to rebuild the knowledge base using contextual chunking with Voyage voyage-context-3

Register `/run_nightly_pipeline` in the Slack app's Slash Commands settings so Slack delivers it to this Socket Mode bot. The command calls the backend's `/run_nightly_pipeline/` endpoint and therefore requires the backend's techsupport pipeline configuration to be present.

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

## Future Enhancements

- Response streaming
- Return screenshots/images with answers
- Allow users to provide screenshots with their questions
- Let users Direct Message the bot for Question-Answering

## Socials
- [Link to Medium.com blog](https://medium.com/@carlos-a-escobar/deep-dive-into-the-best-chunking-indexing-method-for-rag-5921d29f138f)