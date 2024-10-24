"""
FastAPI application for lil lisa server
"""

import io
import os
import re
import shutil
import time
import traceback
import zipfile
from contextlib import asynccontextmanager
from difflib import get_close_matches
from typing import Optional

import git
import jwt
import tiktoken
import uvicorn  # type: ignore
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    StreamingResponse,
)
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as OpenAI_Llama

import lancedb
from speedict import Rdict  # pylint: disable=no-name-in-module

from src import utils
from src.agent_and_tools import (
    PRODUCT,
    answer_from_document_retrieval,
    handle_user_answer,
    improve_query,
    update_retriever,
    get_matching_versions,
    IDDM_PRODUCT_VERSIONS,
    IDA_PRODUCT_VERSIONS
)
from src.lillisa_server_context import LOCALE, LilLisaServerContext
from src.llama_index_lancedb_vector_store import LanceDBVectorStore
from src.llama_index_markdown_reader import MarkdownReader

REACT_AGENT_PROMPT = None
LANCEDB_FOLDERPATH = None
AUTHENTICATION_KEY = None
DOCUMENTATION_FOLDERPATH = None
QA_PAIRS_GITHUB_REPO_URL = None
QA_PAIRS_FOLDERPATH = None
DOCUMENTATION_NEW_VERSIONS = None
DOCUMENTATION_EOC_VERSIONS = None
DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS = None
DOCUMENTATION_IA_PRODUCT_VERSIONS = None
DOCUMENTATION_IA_SELFMANAGED_VERSIONS = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Context manager to perform startup and shutdown actions"""
    global REACT_AGENT_PROMPT, LANCEDB_FOLDERPATH, AUTHENTICATION_KEY, DOCUMENTATION_FOLDERPATH, QA_PAIRS_GITHUB_REPO_URL, QA_PAIRS_FOLDERPATH, DOCUMENTATION_NEW_VERSIONS, DOCUMENTATION_EOC_VERSIONS, DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS, DOCUMENTATION_IA_PRODUCT_VERSIONS, DOCUMENTATION_IA_SELFMANAGED_VERSIONS

    lillisa_server_env = utils.LILLISA_SERVER_ENV_DICT

    if react_agent_prompt_filepath := lillisa_server_env["REACT_AGENT_PROMPT_FILEPATH"]:
        react_agent_prompt_filepath = str(react_agent_prompt_filepath)
    else:
        utils.logger.critical("REACT_AGENT_PROMPT_FILEPATH not found in lillisa_server.env")
        raise ValueError("REACT_AGENT_PROMPT_FILEPATH not found in lillisa_server.env")

    if not os.path.exists(react_agent_prompt_filepath):
        utils.logger.critical("%s not found", react_agent_prompt_filepath)
        raise NotImplementedError(f"{react_agent_prompt_filepath} not found")

    with open(react_agent_prompt_filepath, "r", encoding="utf-8") as file:
        REACT_AGENT_PROMPT = file.read()

    if lancedb_folderpath := lillisa_server_env["LANCEDB_FOLDERPATH"]:
        LANCEDB_FOLDERPATH = str(lancedb_folderpath)
    else:
        utils.logger.critical("LANCEDB_FOLDERPATH not found in lillisa_server.env")
        raise ValueError("LANCEDB_FOLDERPATH not found in lillisa_server.env")

    if not os.path.exists(LANCEDB_FOLDERPATH):
        utils.logger.critical("%s not found", LANCEDB_FOLDERPATH)
        raise NotImplementedError(f"{LANCEDB_FOLDERPATH} not found")

    if authentication_key := lillisa_server_env["AUTHENTICATION_KEY"]:
        AUTHENTICATION_KEY = str(authentication_key)
    else:
        utils.logger.critical("AUTHENTICATION_KEY not found in lillisa_server.env")
        raise ValueError("AUTHENTICATION_KEY not found in lillisa_server.env")

    if documentation_folderpath := lillisa_server_env["DOCUMENTATION_FOLDERPATH"]:
        DOCUMENTATION_FOLDERPATH = str(documentation_folderpath)
    else:
        utils.logger.critical("DOCUMENTATION_FOLDERPATH not found in lillisa_server.env")
        raise ValueError("DOCUMENTATION_FOLDERPATH not found in lillisa_server.env")

    if qa_pairs_github_repo_url := lillisa_server_env["QA_PAIRS_GITHUB_REPO_URL"]:
        QA_PAIRS_GITHUB_REPO_URL = str(qa_pairs_github_repo_url)
    else:
        utils.logger.critical("QA_PAIRS_GITHUB_REPO_URL not found in lillisa_server.env")
        raise ValueError("QA_PAIRS_GITHUB_REPO_URL not found in lillisa_server.env")

    if qa_pairs_folderpath := lillisa_server_env["QA_PAIRS_FOLDERPATH"]:
        QA_PAIRS_FOLDERPATH = str(qa_pairs_folderpath)
    else:
        utils.logger.critical("QA_PAIRS_FOLDERPATH not found in lillisa_server.env")
        raise ValueError("QA_PAIRS_FOLDERPATH not found in lillisa_server.env")
    
    if documentation_new_versions := lillisa_server_env["DOCUMENTATION_NEW_VERSIONS"]:
        DOCUMENTATION_NEW_VERSIONS = str(documentation_new_versions).split(", ")
    else:
        utils.logger.critical("DOCUMENTATION_NEW_VERSIONS not found in lillisa_server.env")
        raise ValueError("DOCUMENTATION_NEW_VERSIONS not found in lillisa_server.env")
    
    if documentation_eoc_versions := lillisa_server_env["DOCUMENTATION_EOC_VERSIONS"]:
        DOCUMENTATION_EOC_VERSIONS = str(documentation_eoc_versions).split(", ")
    else:
        utils.logger.critical("DOCUMENTATION_EOC_VERSIONS not found in lillisa_server.env")
        raise ValueError("DOCUMENTATION_EOC_VERSIONS not found in lillisa_server.env")
    
    if documentation_identity_analytics_versions := lillisa_server_env["DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS"]:
        DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS = str(documentation_identity_analytics_versions).split(", ")
    else:
        utils.logger.critical("DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS not found in lillisa_server.env")
        raise ValueError("DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS not found in lillisa_server.env")
    
    if documentation_ia_product_versions := lillisa_server_env["DOCUMENTATION_IA_PRODUCT_VERSIONS"]:
        DOCUMENTATION_IA_PRODUCT_VERSIONS = str(documentation_ia_product_versions).split(", ")
    else:
        utils.logger.critical("DOCUMENTATION_IA_PRODUCT_VERSIONS not found in lillisa_server.env")
        raise ValueError("DOCUMENTATION_IA_PRODUCT_VERSIONS not found in lillisa_server.env")
    
    if documentation_ia_selfmanaged_versions := lillisa_server_env["DOCUMENTATION_IA_SELFMANAGED_VERSIONS"]:
        DOCUMENTATION_IA_SELFMANAGED_VERSIONS = str(documentation_ia_selfmanaged_versions).split(", ")
    else:
        utils.logger.critical("DOCUMENTATION_IA_SELFMANAGED_VERSIONS not found in lillisa_server.env")
        raise ValueError("DOCUMENTATION_IA_SELFMANAGED_VERSIONS not found in lillisa_server.env")

    yield
    # anything to do on application close goes below
    os.unsetenv("OPENAI_API_KEY")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"],
)


def get_llsc(
    session_id: str, locale: Optional[LOCALE] = None, product: Optional[PRODUCT] = None
) -> LilLisaServerContext:
    """Get lil lisa server context"""
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


@app.post(
    "/invoke/",
    response_model=str,
    response_class=PlainTextResponse,
    summary="Invoke the AI with a natural language query",
    response_description="Return answer after retrieving helpful documents",
    responses={
        200: {
            "description": "Query response successfully returned",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "response": "There are 67 entitlements with 9 unique identities and 9 unique account logins. The most frequent identities are Edward CARSON and Nellie MALDONADO (16.42%) while the least frequent is Bradley CHAMBERS (4.48%). The most frequent account logins are CARSON and NMALDONA17 (16.42%) while the least frequent is BCHAMBER10 (4.48%). There are 23 unique permissions, with the most frequent being SAGE-Accounting-Bill of Materials Clerk and SAGE-Accounting-Purchase Order Clerk (10.45%) and the least frequent at 1.49%. Here are the first 5 rows of data"
                        }
                    ]
                }
            },
        },
        404: {"description": "Session id is not found"},
        422: {"description": "Input parameters failed validation"},
        500: {"description": "Internal error. Review logs for details"},
    },
)
async def invoke(session_id: str, locale: str, product: str, nl_query: str, is_expert_answering: bool) -> str:
    """
    * session_id cannot be empty
    * product must be one of either "IDA" or "IDDM"
    * nl_query cannot be empty
    * response: string
    * For HTTP 422 and 500: The error info will be returned in the response json
    """
    try:
        utils.logger.info(
            "session_id: %s, locale: %s, product: %s, nl_query: %s", session_id, locale, product, nl_query
        )

        llsc = get_llsc(session_id, LOCALE.get_locale(locale), PRODUCT.get_product(product))

        if is_expert_answering:
            llsc.add_to_conversation_history("Expert", nl_query)
            return nl_query
        conversation_history_list = llsc.conversation_history

        conversation_history = ""
        for poster, message in conversation_history_list:
            conversation_history += f"{poster}: {message}\n"

        improve_query_tool = FunctionTool.from_defaults(fn=improve_query)
        answer_from_document_retrieval_tool = FunctionTool.from_defaults(
            fn=answer_from_document_retrieval, return_direct=True
        )
        handle_user_answer_tool = FunctionTool.from_defaults(fn=handle_user_answer, return_direct=True)

        llm = OpenAI_Llama(model="gpt-4o-mini")
        react_agent = ReActAgent.from_tools(
            tools=[handle_user_answer_tool, improve_query_tool, answer_from_document_retrieval_tool],
            llm=llm,
            verbose=True if utils.LOG_LEVEL == utils.logging.DEBUG else False
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
    except Exception as exc:  # pylint: disable=broad-except
        traceback.print_exc()
        utils.logger.critical(
            "Internal error in invoke() for session_id: %s and nl_query: %s. Error: %s", session_id, nl_query, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in invoke() for session_id: {session_id} and nl_query: {nl_query}",
        ) from exc


@app.post(
    "/record_endorsement/",
    response_model=str,
    response_class=PlainTextResponse,
    summary="Record the endorsement for the conversation",
    response_description="ok",
    responses={
        200: {"description": "Response successfully returned", "content": "ok"},
        404: {"description": "Session id is not found"},
        422: {"description": "Input parameters failed validation"},
        500: {"description": "Internal error. Review logs for details"},
    },
)
async def record_endorsement(session_id: str, is_expert: bool) -> str:
    """
    * session_id cannot be empty
    * is_expert must be True or False
    * response: string
    * For HTTP 422 and 500: The error info will be returned in the response json
    """
    try:
        utils.logger.info("session_id: %d, is_expert: %s", session_id, is_expert)

        llsc = get_llsc(session_id)
        llsc.record_endorsement(is_expert)
        return "ok"
    except HTTPException as exc:
        raise exc
    except Exception as exc:  # pylint: disable=broad-except
        traceback.print_exc()
        utils.logger.critical(
            "Internal error in record_endorsement() for session_id: %s and is_expert: %s. Error: %s",
            session_id,
            is_expert,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in record_endorsement() for session_id: {session_id} and is_expert: {is_expert}",
        ) from exc


@app.post(
    "/get_golden_qa_pairs/",
    summary="Retrives the golden qa pairs stored in a local directory",
    response_description="done",
    responses={
        200: {"description": "golden qa pairs retrieved successfully", "content": "done"},
        401: {"description": "Failed signature verification. Unauthorized."},
        404: {"description": "Session id is not found"},
        422: {"description": "Input parameters failed validation"},
        500: {"description": "Internal error. Review logs for details"},
    },
)
async def get_golden_qa_pairs(product: str, encrypted_key: str) -> FileResponse:
    """
    * response: string
    * For HTTP 422 and 500: The error info will be returned in the response json
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
    except jwt.exceptions.InvalidSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Failed signature verification. Unauthorized."
        ) from exc
    except HTTPException as exc:
        raise exc
    except Exception as exc:  # pylint: disable=broad-except
        traceback.print_exc()
        utils.logger.critical("Internal error in get_golden_qa_pairs()", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error in get_golden_qa_pairs()",
        ) from exc


@app.post(
    "/update_golden_qa_pairs/",
    response_model=str,
    response_class=PlainTextResponse,
    summary="Updates the golden qa pairs stored in lancedb",
    response_description="done",
    responses={
        200: {
            "description": "golden qa pairs inserted successfully into lancedb",
            "content": "Succesfully inserted QA pairs into DB.",
        },
        401: {"description": "Failed signature verification. Unauthorized."},
        404: {"description": "Session id is not found"},
        422: {"description": "Input parameters failed validation"},
        500: {"description": "Internal error. Review logs for details"},
    },
)
async def update_golden_qa_pairs(product: str, encrypted_key: str) -> str:
    """
    * response: string
    * For HTTP 422 and 500: The error info will be returned in the response json
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
        table_name = product + "_QA_PAIRS"
        try:
            db.drop_table(table_name)
        except Exception:
            utils.logger.exception("Table %s seems to have been deleted. Continuing with insertion process", product)

        # Split the content by the delimiter
        qa_pairs = file_content.split("# Question/Answer Pair")

        # Remove any leading/trailing whitespace and filter out any empty strings
        qa_pairs = [pair.strip() for pair in qa_pairs if pair.strip()]

        # Extract question and answer pairs
        documents = []
        qa_pattern = re.compile(r"Question:\s*(.*?)\nAnswer:\s*(.*)", re.DOTALL)

        if product == "IDDM":
            product_versions = IDDM_PRODUCT_VERSIONS
            version_pattern = re.compile(r"v?\d+\.\d+", re.IGNORECASE)
        else:
            product_versions = IDA_PRODUCT_VERSIONS
            version_pattern = re.compile(r"\b(?:IAP[- ]\d+\.\d+|version[- ]\d+\.\d+|descartes(?:-dev)?)\b", re.IGNORECASE)

        for pair in qa_pairs:
            match = qa_pattern.search(pair)
            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip()
                document = Document(text=f"{question}")
                document.metadata["answer"] = answer
                matched_versions = get_matching_versions(question, product_versions, version_pattern)
                document.metadata["version"] = matched_versions[0] if matched_versions else "none"
                document.excluded_embed_metadata_keys.append("version")
                document.excluded_embed_metadata_keys.append("answer")
                documents.append(document)

        splitter = SentenceSplitter(chunk_size=10000)

        nodes = splitter.get_nodes_from_documents(documents=documents, show_progress=True)

        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

        vector_store = LanceDBVectorStore(uri="lancedb", table_name=table_name, query_type="hybrid")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
        retriever = index.as_retriever(similarity_top_k=8)
        update_retriever(table_name, retriever)

        return "Succesfully inserted QA pairs into DB."
    except jwt.exceptions.InvalidSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Failed signature verification. Unauthorized."
        ) from exc
    except HTTPException as exc:
        raise exc
    except Exception as exc:  # pylint: disable=broad-except
        traceback.print_exc()
        utils.logger.critical("Internal error in update_golden_qa_pairs()", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error in update_golden_qa_pairs()",
        ) from exc


@app.post(
    "/get_conversations/",
    summary="Retrives the appropriate conversations based on who endorsed it.",
    response_description="done",
    responses={
        200: {
            "description": "conversations retrieved successfully",
            "content": "file containing all of the appropriate conversations named conversations.zip",
        },
        401: {"description": "Failed signature verification. Unauthorized."},
        404: {"description": "Session id is not found"},
        422: {"description": "Input parameters failed validation"},
        500: {"description": "Internal error. Review logs for details"},
    },
)
async def get_conversations(product: str, endorsed_by: str, encrypted_key: str) -> StreamingResponse:
    """
    * response: string
    * For HTTP 422 and 500: The error info will be returned in the response json
    """
    try:
        jwt.decode(encrypted_key, AUTHENTICATION_KEY, algorithms="HS256")

        entry_names = os.listdir(LilLisaServerContext.SPEEDICT_FOLDERPATH)
        session_ids = [
            entry
            for entry in entry_names
            if os.path.isdir(os.path.join(LilLisaServerContext.SPEEDICT_FOLDERPATH, entry))
        ]
        useful_conversations = []

        product_enum = PRODUCT.get_product(product)

        for session_id in session_ids:
            endorsements = None
            llsc = get_llsc(session_id)
            llsc_product = llsc.product
            if product_enum == llsc_product:
                if endorsed_by == "user":
                    endorsements = llsc.user_endorsements
                elif endorsed_by == "expert":
                    endorsements = llsc.expert_endorsements
            if endorsements:
                useful_conversations.append(llsc.conversation_history)

        if useful_conversations:
            zip_stream = io.BytesIO()

            with zipfile.ZipFile(zip_stream, "w") as zipf:
                for i, conversation in enumerate(useful_conversations, start=1):
                    filename = f"conversation_{i}.md"
                    conversation_history = ""
                    for poster, message in conversation:
                        conversation_history += f"{poster}: {message}\n"

                    # Create an in-memory file as a string buffer
                    in_memory_file = io.StringIO(conversation_history)

                    # Add the in-memory file to the zip archive
                    zipf.writestr(filename, in_memory_file.getvalue().encode("utf-8"))

            # Seek to the beginning of the byte stream
            zip_stream.seek(0)

            return StreamingResponse(
                zip_stream,
                media_type="application/zip",
                headers={"Content-Disposition": "attachment; filename=conversations.zip"},
            )
        return None
    except jwt.exceptions.InvalidSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Failed signature verification. Unauthorized."
        ) from exc
    except HTTPException as exc:
        raise exc
    except Exception as exc:  # pylint: disable=broad-except
        traceback.print_exc()
        utils.logger.critical("Internal error in get_conversations()", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error in get_conversations()",
        ) from exc


@app.post(
    "/rebuild_docs/",
    response_model=str,
    response_class=PlainTextResponse,
    summary="Rebuilds the documentation database",
    response_description="done",
    responses={
        200: {
            "description": "lancedb database reconstructed with up-to-date documentation",
            "content": "Rebuilt DB successfully!",
        },
        401: {"description": "Failed signature verification. Unauthorized."},
        404: {"description": "Session id is not found"},
        422: {"description": "Input parameters failed validation"},
        500: {"description": "Internal error. Review logs for details"},
    },
)
async def rebuild_docs(encrypted_key: str) -> str:
    """
    * response: string
    * For HTTP 422 and 500: The error info will be returned in the response json
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
                ("https://github.com/radiantlogic-v8/documentation-identity-analytics.git", DOCUMENTATION_IDENTITY_ANALYTICS_VERSIONS),
                ("https://github.com/radiantlogic-v8/documentation-ia-product.git", DOCUMENTATION_IA_PRODUCT_VERSIONS),
                ("https://github.com/radiantlogic-v8/documentation-ia-selfmanaged.git", DOCUMENTATION_IA_SELFMANAGED_VERSIONS),
            ],
        }

        # Function to get all .md files in a directory recursively
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

        # Initialize model and pipeline
        splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=20,
        )
        node_parser = MarkdownNodeParser()
        reader = MarkdownReader()
        file_extractor = {".md": reader}
        Settings.llm = OpenAI_Llama(model="gpt-3.5-turbo")
        pipeline = IngestionPipeline(transformations=[node_parser])

        os.makedirs(DOCUMENTATION_FOLDERPATH, exist_ok=True)

        excluded_metadata_keys = [
            "file_path",
            "file_name",
            "file_type",
            "file_size",
            "creation_date",
            "last_modified_date",
            "version",
            "github_url"
        ]

        for product, repo_branches in product_repos_dict.items():
            product_dir = os.path.join(DOCUMENTATION_FOLDERPATH, product)
            os.makedirs(product_dir, exist_ok=True)

            max_retries = 5

            all_nodes = []
            for repo_url, branches in repo_branches:
                for branch in branches:
                    repo_name = repo_url.rsplit("/", 1)[-1].replace(".git", "")
                    target_dir = os.path.join(product_dir, repo_name) + "/" + branch
                    if os.path.exists(target_dir):
                        shutil.rmtree(target_dir)
                    # Sometimes cloning a github repo fails.
                    attempt = 0
                    success = False
                    while attempt < max_retries and not success:
                        success = clone_repo(repo_url, target_dir, branch)
                        if not success:
                            attempt += 1
                            if attempt < max_retries:
                                time.sleep(10)
                            else:
                                failed_clone_messages += (
                                    "Max retries reached. Failed to clone {repo_url} ({branch}) into {target_dir}. "
                                )

                    md_files = find_md_files(target_dir)

                    for file in md_files:
                        with open(file, "r", encoding="utf-8") as f:
                            first_lines = []
                            for _ in range(5):
                                try:
                                    first_lines.append(next(f).strip())
                                except StopIteration:
                                    break

                        metadata = extract_metadata_from_lines(first_lines)
                        metadata["version"] = branch

                        documents = SimpleDirectoryReader(
                            input_files=[file], file_extractor=file_extractor
                        ).load_data()

                        for doc in documents:
                            for label, value in metadata.items():
                                doc.metadata[label] = value
                            file_path = doc.metadata['file_path']
                            relative_path = file_path.replace(f'docs/{product}/', '')
                            github_url = 'https://github.com/radiantlogic-v8/' + relative_path
                            github_url = github_url.replace(repo_name, repo_name + '/blob')
                            doc.metadata['github_url'] = github_url
                        nodes = pipeline.run(documents=documents, in_place=False)
                        for node in nodes:
                            node.excluded_llm_metadata_keys = excluded_metadata_keys
                            node.excluded_embed_metadata_keys = excluded_metadata_keys

                        all_nodes.extend(nodes)

            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
            new_nodes_to_add = []
            nodes_to_remove = []

            for node in all_nodes:
                length = len(enc.encode(node.text))
                if length > 7000:
                    nodes_to_remove.append(node)
                    document = Document(text=node.text, metadata=node.metadata)
                    new_nodes = splitter.get_nodes_from_documents(documents=[document])
                    new_nodes_to_add.extend(new_nodes)
            all_nodes = [node for node in all_nodes if node not in nodes_to_remove]

            all_nodes.extend(new_nodes_to_add)

            Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

            try:
                db.drop_table(product)
            except Exception:
                utils.logger.exception(
                    "Table %s seems to have been deleted. Continuing with insertion process", product
                )
            vector_store = LanceDBVectorStore(uri="lancedb", table_name=product, query_type="hybrid")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes=all_nodes[:1], storage_context=storage_context)
            index.insert_nodes(all_nodes[1:])
            retriever = index.as_retriever(similarity_top_k=50)
            update_retriever(product, retriever)

        shutil.rmtree(DOCUMENTATION_FOLDERPATH)
        return "Rebuilt DB successfully!" + failed_clone_messages
    except jwt.exceptions.InvalidSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Failed signature verification. Unauthorized."
        ) from exc
    except HTTPException as exc:
        raise exc
    except Exception as exc:  # pylint: disable=broad-except
        traceback.print_exc()
        utils.logger.critical("Internal error in rebuild_docs()", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error in rebuild_docs()",
        ) from exc


@app.get("/", response_class=HTMLResponse)
def home():
    """root path"""
    return """
    <html>
        <head>
            <title>LIL LISA SERVER</title>
        </head>
        <body>
            <h1>LIL LISA SERVER is up and running!</h1>
            For usage instructions, see the <a href='./docs'>Swagger API</a>
        </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, lifespan="on")
