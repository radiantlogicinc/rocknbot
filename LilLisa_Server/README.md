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
- Navigate to lil-lisa folder
- In the terminal, run "make condaenv". This will create a conda environment and install all necessary packages
- Select 'lillisa-server' as python interpreter

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

## Contributing

The project is not currently open for contributions.

### Requirements
- Docker container
- Python 3.11.9 
- RAM: 1.0 GB
- Size of Docker container: 23.6 GB

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