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
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
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
    update_retrievers,
    update_indices,
    create_lancedb_retrievers_and_indices,
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
            raise ValueError(f"Failed to generate response: {str(e)}") from e

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

class StreamingReActAgent(ReActAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = LiteLLM(model=LLM_MODEL)
    def iter_steps(self, prompt: str):
        task = self.create_task(prompt)
        while True:
            step_output = self.run_step(task.task_id)
            yield step_output
            if step_output.is_last:
                break

    def stream_chat(self, prompt: str):
        accumulated = ""
        try:
            for step_output in self.iter_steps(prompt):
                step = step_output.output
                # Append each chunk to the accumulated text.
                accumulated += step.response
                if not step_output.is_last:
                    yield "cot", f"{step.response}"
                else:
                    yield "ans", f"{step.response}"
                    # Update last_thought in non-streaming context.
                    self.llm.last_thought = accumulated
                    return
        except Exception as e:
            utils.logger.error(f"Stream error: {str(e)}")
            # Check if the error indicates we reached max iterations.
            if "Reached max iterations." in str(e):
                final_answer = accumulated.strip()
                # If nothing has been accumulated, perform a fallback non-streaming call.
                if not final_answer:
                    try:
                        # Call the non-streaming chat to retrieve the final observation.
                        response = self.llm.chat([ChatMessage(role="user", content=prompt)]).message.content
                        final_answer = response
                    except Exception as fallback_e:
                        utils.logger.error(f"Fallback call failed: {fallback_e}")
                        final_answer = "No response produced."
                yield "ans", final_answer
            else:
                yield "ans", f"ANS: Error: Unable to process request due to {str(e)}. Please try again."



# -----------------------------------------------------------------------------
# Application Lifecycle Management
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Manages startup and shutdown tasks for the FastAPI application."""
    global REACT_AGENT_PROMPT, LANCEDB_FOLDERPATH, AUTHENTICATION_KEY, DOCUMENTATION_FOLDERPATH, QA_PAIRS_GITHUB_REPO_URL, QA_PAIRS_FOLDERPATH, DOCUMENTATION_NEW_VERSIONS, DOCUMENTATION_EOC_VERSIONS, DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS, DOCUMENTATION_IA_PRODUCT_VERSIONS, DOCUMENTATION_IA_SELFMANAGED_VERSIONS, MAX_ITERATIONS, LLM_MODEL

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

    # Validate LanceDB folder path
    if os.path.exists(LANCEDB_FOLDERPATH):
        create_lancedb_retrievers_and_indices(LANCEDB_FOLDERPATH)
    else:
        await init_lance_databases()

    yield
    os.unsetenv("OPENAI_API_KEY")
    litellm.api_key = None

async def init_lance_databases():
    try:
        os.makedirs(LANCEDB_FOLDERPATH, exist_ok=True)
    except Exception:
        utils.logger.exception("Failed to create LanceDB folder at %s", LANCEDB_FOLDERPATH)
        raise

    startup_encrypted_key = jwt.encode({"some": "payload"}, AUTHENTICATION_KEY, algorithm="HS256")

    try:
        await _run_rebuild_docs_task()
        utils.logger.info("Automatic documentation rebuild task completed successfully during startup.")
    except Exception as rebuild_exc:
        utils.logger.critical("Automatic documentation rebuild failed during startup: %s", rebuild_exc, exc_info=True)
        raise RuntimeError("Failed to automatically rebuild documentation during startup.") from rebuild_exc

    for product in PRODUCT:
        try:
            await _run_update_golden_qa_pairs_task(product.value)
            utils.logger.info("Rebuilt golden QA pairs for %s completed successfully during startup.", product.value)
        except Exception as rebuild_exc:
            utils.logger.critical("Golden QA pairs rebuild for %s failed during startup: %s", product.value, rebuild_exc, exc_info=True)
            raise RuntimeError("Golden QA pairs rebuild failed during startup.") from rebuild_exc

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
@app.post("/invoke_stream_html/", response_class=StreamingResponse)
async def invoke_stream_html(
    session_id: str,
    locale: str,
    product: str,
    nl_query: str,
    is_expert_answering: bool
):
    """
    Processes a natural language query ans stream the reasoning and final answer.

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
    llsc = get_llsc(session_id, LOCALE.get_locale(locale), PRODUCT.get_product(product))
    if is_expert_answering:
        llsc.add_to_conversation_history("Expert", nl_query)
        async def expert_gen():
            formatted_q = nl_query.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            formatted_q = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", formatted_q)
            formatted_q = re.sub(r"(https?://[^\s'\"<]+)", r'<a href="\1" target="_blank">\1</a>', formatted_q)
            formatted_q = formatted_q.replace("\n", "<br>")
            yield f"ANS: {formatted_q}\n"
        return StreamingResponse(expert_gen(), media_type="text/html")
    # Build agent
    conversation_history = "\n".join(f"{p}: {m}" for p, m in llsc.conversation_history)
    tools = [
        FunctionTool.from_defaults(fn=improve_query),
        FunctionTool.from_defaults(fn=answer_from_document_retrieval, return_direct=True),
        FunctionTool.from_defaults(fn=handle_user_answer, return_direct=True),
    ]
    llm = LiteLLM(model=LLM_MODEL)
    react_agent = StreamingReActAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=False,
        max_iterations=MAX_ITERATIONS,
    )
    prompt = (
        REACT_AGENT_PROMPT
        .replace("<PRODUCT>", product)
        .replace("<CONVERSATION_HISTORY>", conversation_history)
        .replace("<QUERY>", nl_query)
    )

    llsc.add_to_conversation_history("User", nl_query)

    def chunk_text(text: str, max_length: int) -> list[str]:
        """
        Splits `text` into a list of chunks where each chunk (except possibly
        single very long words) is no longer than max_length characters, without
        splitting words.
        """
        words = text.split()
        chunks = []
        current_chunk = ""
        for word in words:
            # If the current word itself is longer than max_length,
            # yield the current_chunk (if any) and then yield the full word.
            if len(word) > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(word)
                continue
            # If adding the word would exceed max_length, finish the current chunk.
            if len(current_chunk) + len(word) + 1 > max_length:
                chunks.append(current_chunk.strip())
                current_chunk = f"{word} "
            else:
                current_chunk += f"{word} "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def html_chunk_text(html_text: str) -> list[str]:
        """
        Splits an HTML string into a list of substrings immediately before each
        <br> tag, keeping the tag with the following chunk.
        
        Args:
            html_text: The input HTML string.
            
        Returns:
            A list of strings split immediately before <br> tags.
        """
        br_pattern = r'<br\s*/?>'
        matches = list(re.finditer(br_pattern, html_text, flags=re.IGNORECASE))
        chunks = []

        # Handle the case with no <br> tags
        if not matches:
            return [html_text] if html_text else []

        # Process first chunk (before first <br>)
        first_match = matches[0]
        if first_match.start() > 0:
            chunks.append(html_text[:first_match.start()])

        # Process chunks at each <br> tag
        for i in range(len(matches)):
            current_match = matches[i]
            current_br = html_text[current_match.start():current_match.end()]

            # If this is the last <br> tag
            if i == len(matches) - 1:
                chunks.append(current_br + html_text[current_match.end():])
            else:
                next_match = matches[i + 1]
                chunks.append(current_br + html_text[current_match.end():next_match.start()])

        return chunks

    async def streamer() -> AsyncGenerator[str, None]:
        def format_to_html(t: str) -> str:
            # Convert markdown bold to HTML strong tag.
            t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
            # Replace URLs with an anchor tag that has inline styling.
            t = re.sub(
                r"(https?://[^\s'\"<]+)",
                r'<a href="\1" target="_blank" style="color:blue;text-decoration:underline;">\1</a>',
                t
            )

            return t.replace("\n", "<br>")



        for phase, text in react_agent.stream_chat(prompt):
            if phase == "cot":
                chunks = chunk_text(text, 50)  # 50-character chunks for CoT
                for chunk in chunks:
                    yield f"COT: {format_to_html(chunk)}\n"
                    await asyncio.sleep(0.5)
            else:  # phase == "ans"
                llsc.add_to_conversation_history("Assistant", text)
                html_answer = format_to_html(text)
                chunks = html_chunk_text(html_answer)
                for chunk in chunks:
                    yield f"ANS: {chunk}\n"
                    # await asyncio.sleep(0.025)

    return StreamingResponse(streamer(), media_type="text/html")

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


async def _run_update_golden_qa_pairs_task(product: str):
    """Contains the core logic for updating golden QA pairs, run as a background task."""
    try:
        utils.logger.info(f"Background task: Starting golden QA pair update for {product}.")
        
        # Clone repo (Consider running sync git call in executor)
        temp_qa_folder = None
        try:
            temp_qa_folder = tempfile.mkdtemp()
            utils.logger.info(f"Background task: Cloning {QA_PAIRS_GITHUB_REPO_URL} into {temp_qa_folder}")
            git.Repo.clone_from(QA_PAIRS_GITHUB_REPO_URL, temp_qa_folder)
        except Exception as clone_exc:
            utils.logger.error(f"Background task: Failed to clone QA pairs repo: {clone_exc}", exc_info=True)
            if temp_qa_folder and os.path.exists(temp_qa_folder):
                 shutil.rmtree(temp_qa_folder)
            return # Stop execution if clone fails

        filepath = f"{temp_qa_folder}/{product.lower()}_qa_pairs.md"
        
        if not os.path.exists(filepath):
            utils.logger.error(f"Background task: QA pairs file not found at {filepath}")
            if os.path.exists(temp_qa_folder):
                 shutil.rmtree(temp_qa_folder)
            return # Stop execution if file doesn't exist
        
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                file_content = file.read()
        except Exception as read_exc:
            utils.logger.error(f"Background task: Failed to read QA pairs file {filepath}: {read_exc}", exc_info=True)
            if os.path.exists(temp_qa_folder):
                 shutil.rmtree(temp_qa_folder)
            return # Stop execution if read fails
        finally:
             # Clean up the temporary folder after reading or on error
            if temp_qa_folder and os.path.exists(temp_qa_folder):
                utils.logger.info(f"Background task: Cleaning up temporary folder {temp_qa_folder}")
                shutil.rmtree(temp_qa_folder)

        # Process content and update DB
        db = lancedb.connect(LANCEDB_FOLDERPATH)
        table_name = f"{product}_QA_PAIRS"
        try:
            utils.logger.info(f"Background task: Dropping existing table {table_name} if it exists.")
            db.drop_table(table_name)
        except Exception:
            # This is expected if the table doesn't exist, log as info
            utils.logger.info(f"Background task: Table {table_name} does not exist or could not be dropped, proceeding.")

        qa_pairs = [pair.strip() for pair in file_content.split("# Question/Answer Pair") if pair.strip()]
        if not qa_pairs:
            utils.logger.warning(f"Background task: No QA pairs found in the file for {product}. Skipping DB update.")
            return
            
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
            else:
                 utils.logger.warning(f"Background task: Could not parse QA pair: {pair[:100]}...")

        if not documents:
            utils.logger.warning(f"Background task: No valid documents could be created from QA pairs for {product}. Skipping DB update.")
            return

        splitter = SentenceSplitter(chunk_size=10000) # QA pairs are typically short, large chunk size is fine
        nodes = splitter.get_nodes_from_documents(documents=documents, show_progress=False) # Turn off progress for background task
        
        # Note: Consider if Settings need to be set per request/task
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large") 
        
        vector_store = LanceDBVectorStore(uri="lancedb", table_name=table_name, query_type="hybrid")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        utils.logger.info(f"Background task: Creating/updating index for {table_name} with {len(nodes)} nodes.")
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
        retriever = index.as_retriever(similarity_top_k=8)
        update_retrievers(table_name, retriever)
        update_indices(table_name, index)
        utils.logger.info(f"Background task: Successfully inserted {len(nodes)} QA pairs into DB for {product}.")

    except jwt.exceptions.InvalidSignatureError:
        utils.logger.error(f"Background task: Failed signature verification for update_golden_qa_pairs ({product}). Unauthorized.")
    except Exception as exc:
        utils.logger.critical(f"Background task: Internal error in update_golden_qa_pairs ({product}): {exc}", exc_info=True)
        # Ensure temporary folder is cleaned up even if unexpected error occurs before finally block
        if 'temp_qa_folder' in locals() and temp_qa_folder and os.path.exists(temp_qa_folder):
             utils.logger.warning(f"Background task: Cleaning up temporary folder {temp_qa_folder} due to error.")
             shutil.rmtree(temp_qa_folder)

@app.post("/update_golden_qa_pairs/", response_model=str, response_class=PlainTextResponse)
async def update_golden_qa_pairs(product: str, encrypted_key: str, background_tasks: BackgroundTasks) -> str:
    """
    Initiates the update of golden QA pairs in LanceDB for a specified product in the background.

    Args:
        product (str): Product name ("IDA" or "IDDM").
        encrypted_key (str): JWT key for authentication.
        background_tasks (BackgroundTasks): FastAPI background task manager.

    Returns:
        str: Immediate confirmation message.
    """
    jwt.decode(encrypted_key, AUTHENTICATION_KEY, algorithms="HS256")
    background_tasks.add_task(_run_update_golden_qa_pairs_task, product)
    return "Golden QA pair update initiated. Please wait for ~2 minutes before using Rocknbot"


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


async def _run_rebuild_docs_task():
    """Contains the core logic for rebuilding docs, run as a background task."""
    try:
        utils.logger.info("Background task: Starting documentation rebuild.")
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
            # Note: This synchronous function is called within an async context.
            # Consider making it async or running in a thread pool executor for large repos.
            try:
                git.Repo.clone_from(repo_url, target_dir, branch=branch)
                return True
            except Exception as clone_exc:
                utils.logger.error(f"Background task: Failed to clone {repo_url} ({branch}): {clone_exc}", exc_info=True)
                return False

        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        node_parser = MarkdownNodeParser()
        reader = MarkdownReader()
        file_extractor = {".md": reader}
        Settings.llm = OpenAI_Llama(model="gpt-3.5-turbo") # Note: Consider if this needs to be set per request/task
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
                            success = clone_repo(repo_url, target_dir, branch) # Consider executor for sync git call
                            if success:
                                break
                            if attempt < 4:
                                await asyncio.sleep(10)
                            else:
                                msg = f"Max retries reached. Failed to clone {repo_url} ({branch}) into {target_dir}."
                                utils.logger.error(f"Background task: {msg}")
                                failed_clone_messages += f"{msg} "
                        if not success: # Skip processing if clone failed
                            continue
                        md_files = find_md_files(target_dir)
                        for file in md_files:
                            try:
                                with open(file, "r", encoding="utf-8") as f:
                                    # Read up to 5 lines safely
                                    first_lines = []
                                    for _ in range(5):
                                        try:
                                            line = next(f).strip()
                                            first_lines.append(line)
                                        except StopIteration:
                                            break
                                metadata = extract_metadata_from_lines(first_lines)
                                metadata["version"] = branch
                            except Exception as e:
                                utils.logger.error(f"Background task: Failed to process file {file}: {e}", exc_info=True)
                                continue
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

                if not all_nodes: # Skip DB update if no nodes were generated for the product
                    utils.logger.warning(f"Background task: No documents processed for product {product}. Skipping DB update.")
                    continue

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

                Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large") # Note: Consider if this needs to be set per request/task
                try:
                    if product in db.table_names():
                        utils.logger.info(f"Background task: Dropping existing table {product}.")
                        db.drop_table(product)

                    vector_store = LanceDBVectorStore(uri="lancedb", table_name=product, query_type="hybrid")
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    # Ensure there's at least one node before creating/inserting
                    if all_nodes:
                        utils.logger.info(f"Background task: Creating/updating index for {product} with {len(all_nodes)} nodes.")
                        index = VectorStoreIndex(nodes=all_nodes[:1], storage_context=storage_context)
                        if len(all_nodes) > 1:
                            index.insert_nodes(all_nodes[1:])
                        retriever = index.as_retriever(similarity_top_k=50)
                        update_retrievers(product, retriever)
                        update_indices(product, index)
                        utils.logger.info(f"Background task: Successfully updated retriever for {product}.")
                    else:
                         utils.logger.warning(f"Background task: No nodes to index for product {product}.")

                except Exception as db_exc:
                    utils.logger.critical(f"Background task: Could not rebuild table {product}. Error: {db_exc}", exc_info=True)

        result_message = f"Rebuilt DB successfully!{failed_clone_messages}" # This message is now only logged
        utils.logger.info(f"Background task: Documentation rebuild finished. Result: {result_message}")

    except jwt.exceptions.InvalidSignatureError:
        # Log the authentication error specifically
        utils.logger.error("Background task: Failed signature verification for rebuild_docs. Unauthorized.")
    except Exception as exc:
        # Log any other errors during the rebuild process
        utils.logger.critical(f"Background task: Documentation rebuild failed unexpectedly. Error: {exc}", exc_info=True)


@app.post("/rebuild_docs/", response_model=str, response_class=PlainTextResponse)
async def rebuild_docs(encrypted_key: str, background_tasks: BackgroundTasks) -> str:
    """
    Initiates the documentation database rebuild in the background.

    Args:
        encrypted_key (str): JWT key for authentication.
        background_tasks (BackgroundTasks): FastAPI background task manager.

    Returns:
        str: Immediate confirmation message.
    """
    jwt.decode(encrypted_key, AUTHENTICATION_KEY, algorithms="HS256")
    background_tasks.add_task(_run_rebuild_docs_task)
    return "Documentation rebuild initiated. Please wait for ~10 minutes before using Rocknbot"


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>LIL LISA SERVER Streaming Example</title>
        </head>
        <body>
            <h1>LIL LISA SERVER is up and running!</h1>
            <p>For usage instructions, see the <a href='./docs'>Swagger API</a></p>
        </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, lifespan="on")
