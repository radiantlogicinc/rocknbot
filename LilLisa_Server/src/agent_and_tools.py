"""
ReAct agent that handles a query in an intelligent manner
"""



import os
import re
import traceback
from difflib import get_close_matches
from enum import Enum
import logging
import json


from litellm import completion
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI

import lancedb

from src import utils
from src.llama_index_lancedb_vector_store import LanceDBVectorStore

OPENAI_API_KEY = None
IDDM_RETRIEVER = None
IDA_RETRIEVER = None
IDDM_QA_PAIRS_RETRIEVER = None
IDA_QA_PAIRS_RETRIEVER = None
RERANKER = SentenceTransformerRerank(top_n=50, model="cross-encoder/ms-marco-MiniLM-L-12-v2")
CLIENT = None
OPENAI_CLIENT = None

QA_SYSTEM_PROMPT = None
QA_USER_PROMPT = None

IDDM_PRODUCT_VERSIONS = None
IDA_PRODUCT_VERSIONS = None

lillisa_server_env = utils.LILLISA_SERVER_ENV_DICT

if not (LLM_MODEL := lillisa_server_env.get("LLM_MODEL")):
    traceback.print_exc()
    utils.logger.critical("LLM_MODEL not found in lillisa_server.env")
    raise ValueError("LLM_MODEL not found in lillisa_server.env")

if fp := lillisa_server_env["SPEEDICT_FOLDERPATH"]:
    speedict_folderpath = str(fp)
else:
    traceback.print_exc()
    utils.logger.critical("SPEEDICT_FOLDERPATH not found in lillisa_server.env")
    raise ValueError("SPEEDICT_FOLDERPATH not found in lillisa_server.env")


if fp := lillisa_server_env["OPENAI_API_KEY_FILEPATH"]:
    openai_api_key_filepath = str(fp)
else:
    traceback.print_exc()
    utils.logger.critical("OPENAI_API_KEY_FILEPATH not found in lillisa_server.env")
    raise ValueError("OPENAI_API_KEY_FILEPATH not found in lillisa_server.env")

with open(openai_api_key_filepath, "r", encoding="utf-8") as file:
    OPENAI_API_KEY = file.read()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)


if fp := lillisa_server_env["QA_SYSTEM_PROMPT_FILEPATH"]:
    qa_system_prompt_filepath = str(fp)
else:
    traceback.print_exc()
    utils.logger.critical("QA_SYSTEM_PROMPT_FILEPATH not found in lillisa_server.env")
    raise ValueError("QA_SYSTEM_PROMPT_FILEPATH not found in lillisa_server.env")

with open(qa_system_prompt_filepath, "r", encoding="utf-8") as file:
    QA_SYSTEM_PROMPT = file.read()


if fp := lillisa_server_env["QA_USER_PROMPT_FILEPATH"]:
    qa_user_prompt_filepath = str(fp)
else:
    traceback.print_exc()
    utils.logger.critical("QA_USER_PROMPT_FILEPATH not found in lillisa_server.env")
    raise ValueError("QA_USER_PROMPT_FILEPATH not found in lillisa_server.env")

with open(qa_user_prompt_filepath, "r", encoding="utf-8") as file:
    QA_USER_PROMPT = file.read()


if fp := lillisa_server_env["AWS_ACCESS_KEY_ID_FILEPATH"]:
    aws_access_key_id_filepath = str(fp)
else:
    traceback.print_exc()
    utils.logger.critical("AWS_ACCESS_KEY_ID not found in lillisa_server.env")
    raise ValueError("AWS_ACCESS_KEY_ID not found in lillisa_server.env")

with open(aws_access_key_id_filepath, "r", encoding="utf-8") as file:
    aws_access_key_id = file.read()


if fp := lillisa_server_env["AWS_SECRET_ACCESS_KEY_FILEPATH"]:
    aws_secret_access_key_filepath = str(fp)
else:
    traceback.print_exc()
    utils.logger.critical("AWS_SECRET_ACCESS_KEY not found in lillisa_server.env")
    raise ValueError("AWS_SECRET_ACCESS_KEY not found in lillisa_server.env")

with open(aws_secret_access_key_filepath, "r", encoding="utf-8") as file:
    aws_secret_access_key = file.read()

if not os.path.exists(speedict_folderpath):
    os.makedirs(speedict_folderpath)

if fp := lillisa_server_env["LANCEDB_FOLDERPATH"]:
    lancedb_folderpath = str(fp)
else:
    traceback.print_exc()
    utils.logger.critical("LANCEDB_FOLDERPATH not found in lillisa_server.env")
    raise ValueError("LANCEDB_FOLDERPATH not found in lillisa_server.env")

if iddm_product_versions := lillisa_server_env["IDDM_PRODUCT_VERSIONS"]:
    IDDM_PRODUCT_VERSIONS = str(iddm_product_versions).split(", ")
else:
    traceback.print_exc()
    utils.logger.critical("IDDM_PRODUCT_VERSIONS not found in lillisa_server.env")
    raise ValueError("IDDM_PRODUCT_VERSIONS not found in lillisa_server.env")

if ida_product_versions := lillisa_server_env["IDA_PRODUCT_VERSIONS"]:
    IDA_PRODUCT_VERSIONS = str(ida_product_versions).split(", ")
else:
    traceback.print_exc()
    utils.logger.critical("IDA_PRODUCT_VERSIONS not found in lillisa_server.env")
    raise ValueError("IDA_PRODUCT_VERSIONS not found in lillisa_server.env")

IDDM_INDEX = None
IDDM_QA_PAIRS_INDEX = None
IDA_INDEX = None
IDA_QA_PAIRS_INDEX = None
IDDM_RETRIEVER = None
IDA_RETRIEVER = None
IDDM_QA_PAIRS_RETRIEVER = None
IDA_QA_PAIRS_RETRIEVER = None
def create_docdbs_lancedb_retrievers_and_indices(lancedb_folderpath: str) -> None:
    """Create indices and retrievers from lancedb tables, attempting to create indices if they don't exist."""
    global IDDM_RETRIEVER, IDA_RETRIEVER
    global IDDM_INDEX, IDA_INDEX

    lance_db = lancedb.connect(lancedb_folderpath)
    iddm_table = lance_db.open_table("IDDM")
    ida_table = lance_db.open_table("IDA")
    iddm_vector_store = LanceDBVectorStore.from_table(iddm_table)
    ida_vector_store = LanceDBVectorStore.from_table(ida_table)
    IDDM_INDEX = VectorStoreIndex.from_vector_store(vector_store=iddm_vector_store)
    IDA_INDEX = VectorStoreIndex.from_vector_store(vector_store=ida_vector_store)
    IDDM_RETRIEVER = IDDM_INDEX.as_retriever(similarity_top_k=50)
    IDA_RETRIEVER = IDA_INDEX.as_retriever(similarity_top_k=50)

def create_qa_pairs_lancedb_retrievers_and_indices(lancedb_folderpath: str) -> None:
    """Create indices and retrievers from lancedb tables, attempting to create indices if they don't exist."""
    global IDDM_QA_PAIRS_RETRIEVER, IDA_QA_PAIRS_RETRIEVER
    global IDDM_QA_PAIRS_INDEX, IDA_QA_PAIRS_INDEX

    lance_db = lancedb.connect(lancedb_folderpath)
    iddm_qa_pairs_table = lance_db.open_table("IDDM_QA_PAIRS")
    ida_qa_pairs_table = lance_db.open_table("IDA_QA_PAIRS")
    iddm_qa_pairs_vector_store = LanceDBVectorStore.from_table(iddm_qa_pairs_table, "vector")
    ida_qa_pairs_vector_store = LanceDBVectorStore.from_table(ida_qa_pairs_table, "vector")
    IDDM_QA_PAIRS_INDEX = VectorStoreIndex.from_vector_store(vector_store=iddm_qa_pairs_vector_store)
    IDA_QA_PAIRS_INDEX = VectorStoreIndex.from_vector_store(vector_store=ida_qa_pairs_vector_store)
    IDDM_QA_PAIRS_RETRIEVER = IDDM_QA_PAIRS_INDEX.as_retriever(similarity_top_k=8)
    IDA_QA_PAIRS_RETRIEVER = IDA_QA_PAIRS_INDEX.as_retriever(similarity_top_k=8)

def create_lancedb_retrievers_and_indices(lancedb_folderpath: str) -> None:
    """Create indices and retrievers from lancedb tables, attempting to create indices if they don't exist."""
    create_docdbs_lancedb_retrievers_and_indices(lancedb_folderpath)
    create_qa_pairs_lancedb_retrievers_and_indices(lancedb_folderpath)

class PRODUCT(str, Enum):
    """Product"""

    IDA = "IDA"
    IDDM = "IDDM"

    @staticmethod
    def get_product(product: str) -> "PRODUCT":
        """get product"""
        if product in (product.value for product in PRODUCT):
            return PRODUCT(product)
        raise ValueError(f"{product} does not exist")


# def update_retrievers(retriever_name, new_retriever):
#     """
#     Updates the reference to the appropriate retriever after "rebuild_docs" or "update_golden_qa_pairs" is called.
#     """
#     global IDDM_RETRIEVER, IDA_RETRIEVER, IDDM_QA_PAIRS_RETRIEVER, IDA_QA_PAIRS_RETRIEVER
#     if retriever_name == "IDDM":
#         IDDM_RETRIEVER = new_retriever
#     elif retriever_name == "IDA":
#         IDA_RETRIEVER = new_retriever
#     elif retriever_name == "IDDM_QA_PAIRS":
#         IDDM_QA_PAIRS_RETRIEVER = new_retriever
#     elif retriever_name == "IDA_QA_PAIRS":
#         IDA_QA_PAIRS_RETRIEVER = new_retriever
#     else:
#         raise ValueError(f"{retriever_name} does not exist")


# def update_indices(retriever_name, new_index):
#     """
#     Updates the reference to the appropriate indices after "rebuild_docs" or "update_golden_qa_pairs" is called.
#     """
#     global IDDM_INDEX, IDA_INDEX, IDDM_QA_PAIRS_INDEX, IDA_QA_PAIRS_INDEX
#     if retriever_name == "IDDM":
#         IDDM_INDEX = new_index
#     elif retriever_name == "IDA":
#         IDA_INDEX = new_index
#     elif retriever_name == "IDDM_QA_PAIRS":
#         IDDM_QA_PAIRS_INDEX = new_index
#     elif retriever_name == "IDA_QA_PAIRS":
#         IDA_QA_PAIRS_INDEX = new_index
#     else:
#         raise ValueError(f"{retriever_name} does not exist")


def handle_user_answer(answer: str) -> str:
    """
    Tool should be called when a user enters an answer to a previous question of theirs. Thank them and merely mimic their answer.
    """
    return answer

def improve_query(query: str, conversation_history: str) -> str:
    """
    Clears up vagueness from query with the help of the conversation history and returns a new query revealing the user's true intention, without distorting the meaning behind the original query. If needed, this should be the first tool called; else, should not be called at all.
    * query should be the original query that the user prompted the agent with, needing some clarification
    * conversation_history is the conversation history the user prompted the agent with
    """
    user_prompt = f"""
    ###CONVERSATION HISTORY###
    {conversation_history}

    ###QUERY###
    {query}

    Based on the conversation history and query, generate a new query that links the two, maximizing semantic understanding.
    """
    
    response = ""
    for chunk in completion(
        model=LLM_MODEL, 
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        stream=True,  # Enable streaming
    ):
        # Process each chunk as it arrives
        if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            response += content

    return response


def format_tables_in_chunks(chunks: str) -> str:
    """Detect and format tables in retrieved chunks at query time.

    This function processes the input string to identify Markdown tables,
    preserving their original format and adding a key-value representation.

    Args:
        chunks (str): Concatenated text from retrieved document nodes.

    Returns:
        str: Formatted text with tables in both Markdown and key-value formats.
    """
    lines = chunks.splitlines()
    result_lines = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Check if the line might start a table (contains multiple pipes)
        if line.count("|") >= 2:
            table_start = i
            table_lines = [line]
            i += 1

            # Collect lines that form the table (pipes or separator lines)
            while i < len(lines) and (lines[i].count("|") >= 2 or set(lines[i].strip()) <= {"|", "-", " "}):
                table_lines.append(lines[i])
                i += 1

            # Verify table structure (needs at least header and separator)
            if len(table_lines) >= 2:
                # Preserve the original Markdown table
                result_lines.extend(table_lines)
                result_lines.append("")  # Separator

                # Generate key-value format
                try:
                    headers = [col.strip() for col in table_lines[0].strip("|").split("|")]
                    data_lines = [ln for ln in table_lines[2:] if "|" in ln]

                    if headers and data_lines:
                        result_lines.append("**Same table in key-value format:**")
                        for idx, row in enumerate(data_lines, start=1):
                            cols = [col.strip() for col in row.strip("|").split("|")]
                            result_lines.append(f"{idx}.")
                            for header, value in zip(headers, cols):
                                if header and value:
                                    result_lines.append(f"{idx}.{header}={value}")
                            result_lines.append("")  # Blank line between rows
                except Exception as e:
                    logging.error("Error processing table formatting: %s", e, exc_info=True)
                    result_lines.append("An error occurred while formatting the table.")

            else:
                # Not a valid table, treat as regular text
                result_lines.extend(table_lines)
        else:
            # Non-table line
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines)


def answer_from_document_retrieval(
    product: str, original_query: str, generated_query: str, conversation_history: str
) -> str:
    """
    RAG Search. Searches through a database of 10,000 documents, and based on a query, returns the top-10 relevant documents and synthesizes an answer.
    Return a JSON string with response and top 10 reranked nodes.
    """
    response = ""
    qa_system_prompt = QA_SYSTEM_PROMPT
    query = generated_query or original_query

    product_enum = PRODUCT.get_product(product)
    if product_enum == PRODUCT.IDDM:
        product_versions = IDDM_PRODUCT_VERSIONS
        version_pattern = re.compile(r"v?\d+\.\d+", re.IGNORECASE)
        document_index = IDDM_INDEX
        qa_pairs_index = IDDM_QA_PAIRS_INDEX
        default_document_retriever = IDDM_RETRIEVER
        default_qa_pairs_retriever = IDDM_QA_PAIRS_RETRIEVER
    else:
        product_versions = IDA_PRODUCT_VERSIONS
        version_pattern = re.compile(r"\b(?:IAP[- ]\d+\.\d+|version[- ]\d+\.\d+|descartes(?:-dev)?)\b", re.IGNORECASE)
        document_index = IDA_INDEX
        qa_pairs_index = IDA_QA_PAIRS_INDEX
        default_document_retriever = IDA_RETRIEVER
        default_qa_pairs_retriever = IDA_QA_PAIRS_RETRIEVER

    if matched_versions := get_matching_versions(
        original_query, product_versions, version_pattern
    ):
        qa_system_prompt += f"\n10. Mention the product version(s) you used to craft your response were '{' and '.join(matched_versions)}'"
        lance_filter_documents = " OR ".join(f"(metadata.version = '{version}')" for version in matched_versions)
        lance_filter_qa_pairs = (
            f"(metadata.version = 'none') OR {lance_filter_documents}"
        )
        document_retriever = document_index.as_retriever(
            vector_store_kwargs={"where": lance_filter_documents}, similarity_top_k=50
        )
        qa_pairs_retriever = qa_pairs_index.as_retriever(
            vector_store_kwargs={"where": lance_filter_qa_pairs}, similarity_top_k=8
        )
    else:
        qa_system_prompt += "\n10. At the beginning of your response, mention that because a specific product version was not specified, information from all available versions was used."
        document_retriever = default_document_retriever
        qa_pairs_retriever = default_qa_pairs_retriever

    qa_nodes = qa_pairs_retriever.retrieve(query)
    relevant_qa_nodes = []
    potentially_relevant_qa_nodes = []

    for node in qa_nodes:
        if 0.85 <= node.score <= 1.0:
            relevant_qa_nodes.append(node)
        elif 0.7 <= node.score < 0.85:
            potentially_relevant_qa_nodes.append(node)

    # if relevant_qa_nodes:
    #     response += "Here are some relevant QA pairs that have been verified by an expert!\n"
    #     for idx, node in enumerate(relevant_qa_nodes, start=1):
    #         response += f"\nMatch {idx}:\nQuestion: {node.text}\nAnswer: {node.metadata['answer']}\n"
    #     response += "\n\nAfter searching through the documentation database, this was found:\n"

    try:
        nodes = document_retriever.retrieve(query)
    except Warning:
        return "No relevant documents were found for this query."
    
    combined_nodes = nodes + relevant_qa_nodes
    reranked_nodes = RERANKER.postprocess_nodes(nodes=combined_nodes, query_str=query)[:10]
    # Safely extract github_urls only from nodes that have them
    useful_links = []
    for node in reranked_nodes:
        if url := node.metadata.get("github_url"):  # Only add URLs if they exist
            useful_links.append(url)
    useful_links = list(dict.fromkeys(useful_links))[:3]  # Take top 3 github_url

    chunks = []
    for node in reranked_nodes:
        # Check if this is a QA node
        if 'answer' in node.metadata:
            # Format as a QA pair with clear prefix
            answer = node.metadata.get('answer', 'No answer found')
            chunks.append(f"EXPERT VERIFIED ANSWER: {answer}")
        else:
            # Regular document node
            chunks.append(node.text)
    
    raw_chunks = "\n\n".join(chunks)
    formatted_chunks = format_tables_in_chunks(raw_chunks)

    user_prompt = QA_USER_PROMPT.replace("<CONTEXT>", formatted_chunks)
    user_prompt = user_prompt.replace("<CONVERSATION_HISTORY>", conversation_history)
    user_prompt = user_prompt.replace("<QUESTION>", original_query)

    llm_response = ""
    for chunk in completion(
        model=LLM_MODEL, 
        messages=[
            {"role": "system", "content": qa_system_prompt}, 
            {"role": "user", "content": user_prompt}
        ],
        stream=True,  # Enable streaming
    ):
        if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            llm_response += content
    
    response += llm_response

    # Format links as plain markdown links to avoid double HTML conversion
    response += "\n\nHere are some potentially helpful documentation links:"
    for link in useful_links:
        # Extract the last part of the URL to use as link text
        path_parts = link.split('/')
        filename = path_parts[-1]
        response += f"\n- [{filename}]({link})"

    if potentially_relevant_qa_nodes and not relevant_qa_nodes:
        response += "\n\n\n"
        response += "In addition, here are some potentially relevant QA pairs that have been verified by an expert!\n"
        for idx, node in enumerate(potentially_relevant_qa_nodes, start=1):
            response += f"\nMatch {idx}:\nQuestion: {node.text}\nAnswer: {node.metadata['answer']}\n"

    nodes_info = []
    for node in reranked_nodes:
        if 'answer' in node.metadata:
            # For QA nodes, include both question and answer in the text field
            nodes_info.append({
                "text": f"EXPERT VERIFIED QA PAIR:\nQuestion: {node.text}\nAnswer: {node.metadata['answer']}",
                "metadata": node.metadata
            })
        else:
            # Regular document node
            nodes_info.append({"text": node.text, "metadata": node.metadata})
    
    response_dict = {"response": response, "reranked_nodes": nodes_info}
    return json.dumps(response_dict)


def get_matching_versions(query, product_versions, version_pattern):
    """
    Not a tool for the ReAct agent but instead a helper function for "answer_from_document_retrieval.
    Helps extract a version from a query.
    """
    extracted_versions = version_pattern.findall(query)
    matched_versions = []
    for extracted_version in extracted_versions:
        if closest_match := get_close_matches(
            extracted_version, product_versions, n=1, cutoff=0.4
        ):
            matched_versions.append(closest_match[0])
    return matched_versions