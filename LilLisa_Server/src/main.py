import asyncio
import io
import logging
import os
import re
import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Generator, Optional, Sequence

import git
import jwt
import litellm
import tiktoken
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    StreamingResponse,
)
from litellm import completion
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import LLM, ChatMessage, ChatResponse, LLMMetadata
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as OpenAI_Llama
from pydantic import Extra
from speedict import Rdict

import lancedb

from src import utils
from src.agent_and_tools import (
    IDA_PRODUCT_VERSIONS,
    IDDM_PRODUCT_VERSIONS,
    PRODUCT,
    answer_from_document_retrieval,
    get_matching_versions,
    handle_user_answer,
    improve_query,
    update_retriever,
)
from src.lillisa_server_context import LOCALE, LilLisaServerContext
from src.llama_index_lancedb_vector_store import LanceDBVectorStore
from src.llama_index_markdown_reader import MarkdownReader

logging.getLogger("LiteLLM").setLevel(logging.INFO)

# Global configuration variables
REACT_AGENT_PROMPT = None  # Path to the React agent prompt file
LANCEDB_FOLDERPATH = None  # Path to the LanceDB folder
AUTHENTICATION_KEY = None  # Authentication key for JWT
DOCUMENTATION_FOLDERPATH = None  # Path to the documentation folder
QA_PAIRS_GITHUB_REPO_URL = None  # URL of the GitHub repository for QA pairs
QA_PAIRS_FOLDERPATH = None  # Path to the QA pairs folder
DOCUMENTATION_NEW_VERSIONS = None  # List of new documentation versions
DOCUMENTATION_EOC_VERSIONS = None  # List of EOC documentation versions
DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS = None  # List of Identity Analytics documentation versions
DOCUMENTATION_IA_PRODUCT_VERSIONS = None  # List of IA product documentation versions
DOCUMENTATION_IA_SELFMANAGED_VERSIONS = None  # List of IA self-managed documentation versions
MAX_ITERATIONS = None  # Maximum number of iterations for the ReAct agent
LLM_MODEL = None  # Model name


# -----------------------------------------------------------------------------
# Custom LLM Implementation
# -----------------------------------------------------------------------------
class LiteLLM(LLM):
    """Custom LLM implementation using LiteLLM's completion API."""

    class Config:
        extra = Extra.allow

    def __init__(self, model=LLM_MODEL, callback_manager=None, system_prompt=None, **kwargs):
        super().__init__(callback_manager=callback_manager, system_prompt=system_prompt, **kwargs)
        self.model = model
        self.last_thought = ""  # Stores the last generated thought

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=32000, num_output=8191, is_chat_model=True)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        try:
            response = completion(model=self.model, messages=formatted_messages)
            self.last_thought = response["choices"][0]["message"]["content"]
            return ChatResponse(message=ChatMessage(role="assistant", content=self.last_thought))
        except Exception as e:
            utils.logger.error("Error in LiteLLM completion: %s", str(e))
            raise ValueError(f"Failed to generate response: {str(e)}")

    def complete(self, prompt: str, **kwargs: Any) -> str:
        messages = [ChatMessage(role="user", content=prompt)]
        return self.chat(messages, **kwargs).message.content

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Generator[ChatResponse, None, None]:
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        response = completion(model=self.model, messages=formatted_messages, stream=True, **kwargs)
        for chunk in response:
            yield ChatResponse(
                message=ChatMessage(role="assistant", content=chunk["choices"][0]["delta"].get("content", ""))
            )

    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        messages = [ChatMessage(role="user", content=prompt)]
        for resp in self.stream_chat(messages, **kwargs):
            yield resp.message.content

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.complete, prompt, **kwargs)

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        response = completion(model=self.model, messages=formatted_messages, stream=True, **kwargs)
        for chunk in response:
            yield ChatResponse(
                message=ChatMessage(role="assistant", content=chunk["choices"][0]["delta"].get("content", ""))
            )

    async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        messages = [ChatMessage(role="user", content=prompt)]
        async for resp in self.astream_chat(messages, **kwargs):
            yield resp.message.content


# -----------------------------------------------------------------------------
# Application Lifecycle Management
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Manages startup and shutdown tasks for the FastAPI application."""
    global REACT_AGENT_PROMPT, LANCEDB_FOLDERPATH, AUTHENTICATION_KEY, DOCUMENTATION_FOLDERPATH, QA_PAIRS_GITHUB_REPO_URL, QA_PAIRS_FOLDERPATH, DOCUMENTATION_NEW_VERSIONS, DOCUMENTATION_EOC_VERSIONS, DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS, DOCUMENTATION_IA_PRODUCT_VERSIONS, DOCUMENTATION_IA_SELFMANAGED_VERSIONS, MAX_ITERATIONS

    lillisa_server_env = utils.LILLISA_SERVER_ENV_DICT

    # Load configuration from environment variables
    for key, var in [
        ("REACT_AGENT_PROMPT_FILEPATH", "REACT_AGENT_PROMPT"),
        ("LANCEDB_FOLDERPATH", "LANCEDB_FOLDERPATH"),
        ("AUTHENTICATION_KEY", "AUTHENTICATION_KEY"),
        ("DOCUMENTATION_FOLDERPATH", "DOCUMENTATION_FOLDERPATH"),
        ("QA_PAIRS_GITHUB_REPO_URL", "QA_PAIRS_GITHUB_REPO_URL"),
        ("QA_PAIRS_FOLDERPATH", "QA_PAIRS_FOLDERPATH"),
    ]:
        if value := lillisa_server_env.get(key):
            globals()[var] = str(value)
        else:
            utils.logger.critical("%s not found in lillisa_server.env", key)
            raise ValueError(f"{key} not found in lillisa_server.env")

    # Load and validate React agent prompt file
    if not os.path.exists(REACT_AGENT_PROMPT):
        utils.logger.critical("%s not found", REACT_AGENT_PROMPT)
        raise NotImplementedError(f"{REACT_AGENT_PROMPT} not found")
    with open(REACT_AGENT_PROMPT, "r", encoding="utf-8") as file:
        globals()["REACT_AGENT_PROMPT"] = file.read()

    # Validate LanceDB folder path
    if not os.path.exists(LANCEDB_FOLDERPATH):
        utils.logger.critical("%s not found", LANCEDB_FOLDERPATH)
        raise NotImplementedError(f"{LANCEDB_FOLDERPATH} not found")

    # Load max iterations
    if iterations := lillisa_server_env.get("MAX_ITERATIONS"):
        globals()["MAX_ITERATIONS"] = int(iterations)
    else:
        utils.logger.critical("MAX_ITERATIONS not found in lillisa_server.env")
        raise ValueError("MAX_ITERATIONS not found in lillisa_server.env")
    # Load model name
    if model := lillisa_server_env.get("LLM_MODEL"):
        globals()["LLM_MODEL"] = str(model)
    else:
        utils.logger.critical("LLM_MODEL not found in lillisa_server.env")
        raise ValueError("LLM_MODEL not found in lillisa_server.env")
    # Load documentation versions
    for key, var in [
        ("DOCUMENTATION_NEW_VERSIONS", "DOCUMENTATION_NEW_VERSIONS"),
        ("DOCUMENTATION_EOC_VERSIONS", "DOCUMENTATION_EOC_VERSIONS"),
        ("DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS", "DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS"),
        ("DOCUMENTATION_IA_PRODUCT_VERSIONS", "DOCUMENTATION_IA_PRODUCT_VERSIONS"),
        ("DOCUMENTATION_IA_SELFMANAGED_VERSIONS", "DOCUMENTATION_IA_SELFMANAGED_VERSIONS"),
    ]:
        if value := lillisa_server_env.get(key):
            globals()[var] = str(value).split(", ")
        else:
            utils.logger.critical("%s not found in lillisa_server.env", key)
            raise ValueError(f"{key} not found in lillisa_server.env")

    if LLM_API_KEY_FILEPATH := lillisa_server_env.get("LLM_API_KEY_FILEPATH"):
        if not os.path.exists(LLM_API_KEY_FILEPATH):
            utils.logger.critical("%s not found", LLM_API_KEY_FILEPATH)
            raise FileNotFoundError(f"LLM API key file not found: {LLM_API_KEY_FILEPATH}")
        with open(LLM_API_KEY_FILEPATH, "r", encoding="utf-8") as file:
            api_key = file.read().strip()
            # Instead of setting an environment variable, assign directly:
            litellm.api_key = api_key
    else:
        utils.logger.critical("LLM_API_KEY_FILEPATH not found in lillisa_server.env")
        raise ValueError("LLM_API_KEY_FILEPATH not found in lillisa_server.env")
    yield
    os.unsetenv("OPENAI_API_KEY")
    litellm.api_key = None


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def get_llsc(
    session_id: str, locale: Optional[LOCALE] = None, product: Optional[PRODUCT] = None
) -> LilLisaServerContext:
    """
    Retrieves or creates a LilLisaServerContext for a given session ID.

    Args:
        session_id (str): Unique identifier for the session.
        locale (Optional[LOCALE]): Locale for the conversation, required for new sessions.
        product (Optional[PRODUCT]): Product for the conversation, required for new sessions.

    Returns:
        LilLisaServerContext: The session context.

    Raises:
        ValueError: If locale or product is missing for a new session.
    """
    db_folderpath = LilLisaServerContext.get_db_folderpath(session_id)
    try:
        keyvalue_db = Rdict(db_folderpath)
        llsc = keyvalue_db[session_id] if session_id in keyvalue_db else None
    finally:
        keyvalue_db.close()

    if not llsc:
        if not (locale and product):
            raise ValueError("Locale and Product are required to initiate a new conversation.")
        llsc = LilLisaServerContext(session_id, locale, product)
    return llsc


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.post("/invoke_stream/", response_class=StreamingResponse)
async def invoke_stream(session_id: str, locale: str, product: str, nl_query: str, is_expert_answering: bool):
    """
    Waits for the full response and streams it in chunks as plain text.
    """
    try:
        response_text = invoke(session_id, locale, product, nl_query, is_expert_answering)
        chunk_size = 100
        async def stream_generator():
            for i in range(0, len(response_text), chunk_size):
                yield response_text[i:i+chunk_size]
                await asyncio.sleep(0.05)  # slight delay to simulate streaming
        return StreamingResponse(stream_generator(), media_type="text/plain")
    except Exception as e:
        utils.logger.critical("Error in invoke_stream: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/invoke/", response_model=str, response_class=PlainTextResponse)
def invoke(session_id: str, locale: str, product: str, nl_query: str, is_expert_answering: bool) -> str:
    """
    Processes a natural language query or expert answer without streaming.

    Args:
        session_id (str): Unique identifier for the session.
        locale (str): Locale of the conversation.
        product (str): Product ("IDA" or "IDDM").
        nl_query (str): Natural language query or expert answer.
        is_expert_answering (bool): Indicates if an expert is providing the answer.

    Returns:
        str: Response from AI or expert answer.

    Raises:
        HTTPException: On internal errors or invalid input.
    """
    try:
        utils.logger.info("session_id: %s, locale: %s, product: %s, nl_query: %s", session_id, locale, product, nl_query)
        llsc = get_llsc(session_id, LOCALE.get_locale(locale), PRODUCT.get_product(product))

        if is_expert_answering:
            llsc.add_to_conversation_history("Expert", nl_query)
            return nl_query

        conversation_history = "\n".join(f"{poster}: {message}" for poster, message in llsc.conversation_history)
        tools = [
            FunctionTool.from_defaults(fn=improve_query),
            FunctionTool.from_defaults(fn=answer_from_document_retrieval, return_direct=True),
            FunctionTool.from_defaults(fn=handle_user_answer, return_direct=True),
        ]
        llm = LiteLLM(model=LLM_MODEL)
        react_agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=(utils.LOG_LEVEL == utils.logging.DEBUG),
            max_iterations=MAX_ITERATIONS
        )
        react_agent_prompt = (
            REACT_AGENT_PROMPT.replace("<PRODUCT>", product)
            .replace("<CONVERSATION_HISTORY>", conversation_history)
            .replace("<QUERY>", nl_query)
        )
        response = react_agent.chat(react_agent_prompt).response
        llsc.add_to_conversation_history("User", nl_query)
        llsc.add_to_conversation_history("Assistant", response)
        return response

    except HTTPException as exc:
        raise exc
    except Exception as exc:
        if isinstance(exc, ValueError) and "Reached max iterations." in str(exc):
            final_response = llm.last_thought or ""
            utils.logger.info("Returning last thought for session %s: %s", session_id, final_response)
            llsc.add_to_conversation_history("User", nl_query)
            llsc.add_to_conversation_history("Assistant", final_response)
            return final_response
        utils.logger.critical("Internal error in invoke() for session_id: %s, nl_query: %s. Error: %s", session_id, nl_query, exc)
        raise HTTPException(status_code=500, detail=f"Internal error in invoke() for session_id: {session_id}") from exc


@app.post("/record_endorsement/", response_model=str, response_class=PlainTextResponse)
async def record_endorsement(session_id: str, is_expert: bool, thumbs_up: bool) -> str:
    """
    Records an endorsement for a conversation.

    Args:
        session_id (str): Unique identifier for the session.
        is_expert (bool): True if the endorsement is from an expert.
        thumbs_up (bool): True if the endorsement is positive.

    Returns:
        str: "ok" on success.

    Raises:
        HTTPException: On internal errors.
    """
    try:
        utils.logger.info("session_id: %s, is_expert: %s, thumbs_up: %s", session_id, is_expert, thumbs_up)
        llsc = get_llsc(session_id)
        llsc.record_endorsement(is_expert, thumbs_up)
        return "ok"
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        utils.logger.critical("Internal error in record_endorsement() for session_id: %s. Error: %s", session_id, exc)
        raise HTTPException(
            status_code=500, detail=f"Internal error in record_endorsement() for session_id: {session_id}"
        ) from exc


@app.post("/get_golden_qa_pairs/")
async def get_golden_qa_pairs(product: str, encrypted_key: str) -> FileResponse:
    """
    Retrieves golden QA pairs for a specified product from a GitHub repository.

    Args:
        product (str): Product name ("IDA" or "IDDM").
        encrypted_key (str): JWT key for authentication.

    Returns:
        FileResponse: Markdown file containing QA pairs, or None if not found.

    Raises:
        HTTPException: On authentication failure or internal errors.
    """
    try:
        jwt.decode(encrypted_key, AUTHENTICATION_KEY, algorithms="HS256")
        if os.path.exists(QA_PAIRS_FOLDERPATH):
            shutil.rmtree(QA_PAIRS_FOLDERPATH)
        git.Repo.clone_from(QA_PAIRS_GITHUB_REPO_URL, QA_PAIRS_FOLDERPATH)
        filepath = f"{QA_PAIRS_FOLDERPATH}/{product.lower()}_qa_pairs.md"
        if os.path.isfile(filepath) and os.path.getsize(filepath) > 0:
            return FileResponse(filepath)
        return None
    except jwt.exceptions.InvalidSignatureError as e:
        raise HTTPException(
            status_code=401,
            detail="Failed signature verification. Unauthorized.",
        ) from e
    except Exception as exc:
        utils.logger.critical("Internal error in get_golden_qa_pairs(): %s", exc)
        raise HTTPException(status_code=500, detail="Internal error in get_golden_qa_pairs()") from exc


@app.post("/update_golden_qa_pairs/", response_model=str, response_class=PlainTextResponse)
async def update_golden_qa_pairs(product: str, encrypted_key: str) -> str:
    """
    Updates golden QA pairs in LanceDB for a specified product.

    Args:
        product (str): Product name ("IDA" or "IDDM").
        encrypted_key (str): JWT key for authentication.

    Returns:
        str: Confirmation message on success.

    Raises:
        HTTPException: On authentication failure or internal errors.
    """
    try:
        jwt.decode(encrypted_key, AUTHENTICATION_KEY, algorithms="HS256")
        if os.path.exists(QA_PAIRS_FOLDERPATH):
            shutil.rmtree(QA_PAIRS_FOLDERPATH)
        git.Repo.clone_from(QA_PAIRS_GITHUB_REPO_URL, QA_PAIRS_FOLDERPATH)
        filepath = f"{QA_PAIRS_FOLDERPATH}/{product.lower()}_qa_pairs.md"
        with open(filepath, "r", encoding="utf-8") as file:
            file_content = file.read()

        db = lancedb.connect(LANCEDB_FOLDERPATH)
        table_name = f"{product}_QA_PAIRS"
        try:
            db.drop_table(table_name)
        except Exception:
            utils.logger.exception("Table %s seems to have been deleted.", product)

        qa_pairs = [pair.strip() for pair in file_content.split("# Question/Answer Pair") if pair.strip()]
        documents = []
        qa_pattern = re.compile(r"Question:\s*(.*?)\nAnswer:\s*(.*)", re.DOTALL)
        product_versions = IDDM_PRODUCT_VERSIONS if product == "IDDM" else IDA_PRODUCT_VERSIONS
        version_pattern = (
            re.compile(r"v?\d+\.\d+", re.IGNORECASE)
            if product == "IDDM"
            else re.compile(r"\b(?:IAP[- ]\d+\.\d+|version[- ]\d+\.\d+|descartes(?:-dev)?)\b", re.IGNORECASE)
        )

        for pair in qa_pairs:
            if match := qa_pattern.search(pair):
                question, answer = match[1].strip(), match[2].strip()
                doc = Document(text=question)
                doc.metadata["answer"] = answer
                matched_versions = get_matching_versions(question, product_versions, version_pattern)
                doc.metadata["version"] = matched_versions[0] if matched_versions else "none"
                doc.excluded_embed_metadata_keys.extend(["version", "answer"])
                documents.append(doc)

        splitter = SentenceSplitter(chunk_size=10000)
        nodes = splitter.get_nodes_from_documents(documents=documents, show_progress=True)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        vector_store = LanceDBVectorStore(uri="lancedb", table_name=table_name, query_type="hybrid")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
        retriever = index.as_retriever(similarity_top_k=8)
        update_retriever(table_name, retriever)
        return "Successfully inserted QA pairs into DB."
    except jwt.exceptions.InvalidSignatureError as e:
        raise HTTPException(
            status_code=401,
            detail="Failed signature verification. Unauthorized.",
        ) from e
    except Exception as exc:
        utils.logger.critical("Internal error in update_golden_qa_pairs(): %s", exc)
        raise HTTPException(status_code=500, detail="Internal error in update_golden_qa_pairs()") from exc


@app.post("/get_conversations/")
async def get_conversations(product: str, endorsed_by: str, encrypted_key: str) -> StreamingResponse:
    """
    Retrieves conversations based on product and endorsement type.

    Args:
        product (str): Product name ("IDA" or "IDDM").
        endorsed_by (str): "user" or "expert".
        encrypted_key (str): JWT key for authentication.

    Returns:
        StreamingResponse: ZIP file of conversations, or None if none found.

    Raises:
        HTTPException: On authentication failure or internal errors.
    """
    try:
        jwt.decode(encrypted_key, AUTHENTICATION_KEY, algorithms="HS256")
        session_ids = [
            entry
            for entry in os.listdir(LilLisaServerContext.SPEEDICT_FOLDERPATH)
            if os.path.isdir(os.path.join(LilLisaServerContext.SPEEDICT_FOLDERPATH, entry))
        ]
        useful_conversations = []
        product_enum = PRODUCT.get_product(product)

        for session_id in session_ids:
            llsc = get_llsc(session_id)
            if product_enum == llsc.product:
                endorsements = (
                    llsc.user_endorsements
                    if endorsed_by == "user"
                    else llsc.expert_endorsements
                    if endorsed_by == "expert"
                    else None
                )
                if endorsements:
                    useful_conversations.append(llsc.conversation_history)

        if useful_conversations:
            zip_stream = io.BytesIO()
            with zipfile.ZipFile(zip_stream, "w") as zipf:
                for i, conversation in enumerate(useful_conversations, start=1):
                    conversation_history = "\n".join(f"{poster}: {message}" for poster, message in conversation)
                    zipf.writestr(f"conversation_{i}.md", conversation_history.encode("utf-8"))
            zip_stream.seek(0)
            return StreamingResponse(
                zip_stream,
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=conversations.zip"},
            )
        return None
    except jwt.exceptions.InvalidSignatureError as e:
        raise HTTPException(
            status_code=401,
            detail="Failed signature verification. Unauthorized.",
        ) from e
    except Exception as exc:
        utils.logger.critical("Internal error in get_conversations(): %s", exc)
        raise HTTPException(status_code=500, detail="Internal error in get_conversations()") from exc


@app.post("/rebuild_docs/", response_model=str, response_class=PlainTextResponse)
async def rebuild_docs(encrypted_key: str) -> str:
    """
    Rebuilds the documentation database from GitHub repositories.

    Args:
        encrypted_key (str): JWT key for authentication.

    Returns:
        str: Success message with any cloning failure details.

    Raises:
        HTTPException: On authentication failure or internal errors.
    """
    try:
        jwt.decode(encrypted_key, AUTHENTICATION_KEY, algorithms="HS256")
        db = lancedb.connect(LANCEDB_FOLDERPATH)
        failed_clone_messages = ""
        product_repos_dict = {
            "IDDM": [
                ("https://github.com/radiantlogic-v8/documentation-new.git", DOCUMENTATION_NEW_VERSIONS),
                ("https://github.com/radiantlogic-v8/documentation-eoc.git", DOCUMENTATION_EOC_VERSIONS),
            ],
            "IDA": [
                (
                    "https://github.com/radiantlogic-v8/documentation-identity-analytics.git",
                    DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS,
                ),
                ("https://github.com/radiantlogic-v8/documentation-ia-product.git", DOCUMENTATION_IA_PRODUCT_VERSIONS),
                (
                    "https://github.com/radiantlogic-v8/documentation-ia-selfmanaged.git",
                    DOCUMENTATION_IA_SELFMANAGED_VERSIONS,
                ),
            ],
        }

        def find_md_files(directory):
            return [
                os.path.join(root, file)
                for root, _, files in os.walk(directory)
                for file in files
                if file.endswith(".md")
            ]

        def extract_metadata_from_lines(lines):
            metadata = {"title": "", "description": "", "keywords": ""}
            for line in lines:
                if line.startswith("title:"):
                    metadata["title"] = line.split(":", 1)[1].strip()
                elif line.startswith("description:"):
                    metadata["description"] = line.split(":", 1)[1].strip()
                elif line.startswith("keywords:"):
                    metadata["keywords"] = line.split(":", 1)[1].strip()
            return metadata

        def clone_repo(repo_url, target_dir, branch):
            try:
                git.Repo.clone_from(repo_url, target_dir, branch=branch)
                return True
            except Exception:
                return False

        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        node_parser = MarkdownNodeParser()
        reader = MarkdownReader()
        file_extractor = {".md": reader}
        Settings.llm = OpenAI_Llama(model="gpt-3.5-turbo")
        pipeline = IngestionPipeline(transformations=[node_parser])
        excluded_metadata_keys = [
            "file_path",
            "file_name",
            "file_type",
            "file_size",
            "creation_date",
            "last_modified_date",
            "version",
            "github_url",
        ]

        for product, repo_branches in product_repos_dict.items():
            with tempfile.TemporaryDirectory() as temp_dir:
                product_dir = os.path.join(temp_dir, product)
                os.makedirs(product_dir, exist_ok=True)
                all_nodes = []
                for repo_url, branches in repo_branches:
                    for branch in branches:
                        repo_name = repo_url.rsplit("/", 1)[-1].replace(".git", "")
                        target_dir = os.path.join(product_dir, repo_name, branch)
                        if os.path.exists(target_dir):
                            shutil.rmtree(target_dir)
                        success = False
                        for attempt in range(5):
                            success = clone_repo(repo_url, target_dir, branch)
                            if success:
                                break
                            if attempt < 4:
                                await asyncio.sleep(10)
                            else:
                                msg = f"Max retries reached. Failed to clone {repo_url} ({branch}) into {target_dir}."
                                utils.logger.error(msg)
                                failed_clone_messages += f"{msg} "
                        md_files = find_md_files(target_dir)
                        for file in md_files:
                            with open(file, "r", encoding="utf-8") as f:
                                first_lines = [next(f).strip() for _ in range(5) if f.readable()]
                            metadata = extract_metadata_from_lines(first_lines)
                            metadata["version"] = branch
                            documents = SimpleDirectoryReader(
                                input_files=[file], file_extractor=file_extractor
                            ).load_data()
                            for doc in documents:
                                doc.metadata.update(metadata)
                                file_path = doc.metadata["file_path"]
                                relative_path = file_path.replace(f"docs/{product}/", "")
                                github_url = f"https://github.com/radiantlogic-v8/{relative_path}".replace(
                                    repo_name, f"{repo_name}/blob"
                                )
                                doc.metadata["github_url"] = github_url
                            nodes = pipeline.run(documents=documents, in_place=False)
                            for node in nodes:
                                node.excluded_llm_metadata_keys = excluded_metadata_keys
                                node.excluded_embed_metadata_keys = excluded_metadata_keys
                            all_nodes.extend(nodes)

                enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
                new_nodes = []
                for node in all_nodes:
                    if len(enc.encode(node.text)) > 7000:
                        sub_nodes = splitter.get_nodes_from_documents(
                            [Document(text=node.text, metadata=node.metadata)]
                        )
                        new_nodes.extend(sub_nodes)
                    else:
                        new_nodes.append(node)
                all_nodes = new_nodes

                Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
                try:
                    db.drop_table(product)
                except Exception:
                    utils.logger.exception("Table %s seems to have been deleted.", product)
                vector_store = LanceDBVectorStore(uri="lancedb", table_name=product, query_type="hybrid")
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex(nodes=all_nodes[:1], storage_context=storage_context)
                index.insert_nodes(all_nodes[1:])
                retriever = index.as_retriever(similarity_top_k=50)
                update_retriever(product, retriever)

        return f"Rebuilt DB successfully!{failed_clone_messages}"
    except jwt.exceptions.InvalidSignatureError:
        raise HTTPException(status_code=401, detail="Failed signature verification. Unauthorized.")
    except Exception as exc:
        utils.logger.critical("Internal error in rebuild_docs(): %s", exc)
        raise HTTPException(status_code=500, detail="Internal error in rebuild_docs()") from exc


@app.get("/", response_class=HTMLResponse)
def home():
    """Returns a status page with a streaming example."""
    return """
    <html>
        <head>
            <title>LIL LISA SERVER Streaming Example</title>
        </head>
        <body>
            <h1>LIL LISA SERVER is up and running!</h1>
            <div id="chat-output" style="border: 1px solid #ccc; padding: 10px; height: 300px; overflow: auto;"></div>
            <script>
                fetch("http://127.0.0.1:8080/invoke_stream/?session_id=session1&locale=en&product=IDDM&nl_query=Please+provide+streaming+output+details+for+here&is_expert_answering=false")
                    .then(response => {
                        if (!response.body) {
                          throw new Error("ReadableStream not supported in this browser.");
                        }
                        const reader = response.body.getReader();
                        const decoder = new TextDecoder();
                        function readStream() {
                            reader.read().then(({ done, value }) => {
                                if (done) {
                                    console.log("Stream complete");
                                    return;
                                }
                                const chunk = decoder.decode(value, { stream: true });
                                document.getElementById("chat-output").innerHTML += chunk;
                                readStream();
                            }).catch(error => console.error("Error reading stream:", error));
                        }
                        readStream();
                    })
                    .catch(error => console.error("Streaming error:", error));
            </script>
            <p>For usage instructions, see the <a href='./docs'>Swagger API</a></p>
        </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, lifespan="on")
