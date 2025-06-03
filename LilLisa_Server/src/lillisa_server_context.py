"""
access review state
"""

import threading
from enum import Enum
from typing import List, Tuple  # , Union, Dict, Optional
import datetime
import json

from speedict import Rdict  # pylint: disable=no-name-in-module

from src import utils
from src.agent_and_tools import PRODUCT

keyvalue_db_lock = threading.Lock()


class LOCALE(str, Enum):
    """Locale"""

    EN = "en"
    ENUS = "en-US"
    ENGB = "en-GB"
    FR = "fr"

    @staticmethod
    def get_locale(locale: str) -> "LOCALE":
        """get locale"""
        if locale in (locale.value for locale in LOCALE):
            return LOCALE(locale)
        return LOCALE.EN


class LilLisaServerContext:  # pylint: disable=too-many-instance-attributes, too-many-public-methods
    """the access review state variables - these should be persisted across sessions"""

    lillisa_server_env = utils.LILLISA_SERVER_ENV_DICT
    if sf := lillisa_server_env["SPEEDICT_FOLDERPATH"]:
        SPEEDICT_FOLDERPATH = sf
    else:
        utils.logger.critical("SPEEDICT_FOLDERPATH is not set in lillisa_server.env")
        raise ValueError("SPEEDICT_FOLDERPATH is not set in lillisa_server.env")

    def __init__(  # pylint: disable=too-many-arguments
        self,
        session_id: str,
        locale: LOCALE,
        product: PRODUCT,
    ):
        self.locale = locale
        self.product = product

        self.session_id = session_id
        self.query_counter = 0
        self.conversation_history: List[Tuple[str, str, str]] = []
        self.query_artifacts: dict[str, dict] = {}
        
        self.save_context()
    # def update_conversation_history(self, conversation_list: list[Tuple[str, str]]):
    #     """ update the stage and step """
    #     self.conversation_history.extend(conversation_list)

    #     db_folderpath = LilLisaServerContext.get_db_folderpath(session_id)
    #     try:
    #     keyvalue_db = Rdict(db_folderpath)
    #     keyvalue_db[self.reviewer_login] = self
    #     finally:
    #     keyvalue_db.close()

    def save_context(self):
        """save the context"""
        db_folderpath = LilLisaServerContext.get_db_folderpath(self.session_id)
        try:
            keyvalue_db = Rdict(db_folderpath)
            keyvalue_db[self.session_id] = self
        finally:
            keyvalue_db.close()

    # # create a static method to close the session and delete session info from keyvalue_db
    # @staticmethod
    # def close_session(session_id: int):
    #     """close the session"""
    #     db_folderpath = LilLisaServerContext.get_db_folderpath(session_id)
    #     try:
    #         keyvalue_db = Rdict(db_folderpath)
    #         keyvalue_db.delete(session_id)
    #     finally:
    #         keyvalue_db.close()

    def add_to_conversation_history(self, poster: str, message: str, query_id: str = None):
        """
        Add to the conversation history with a unique queryId
        
        Args:
            poster (str): The poster of the message (User, Assistant, Expert)
            message (str): The message content
            query_id (str, optional): A specific query ID if provided, otherwise generated automatically
        """
        if poster == "User":
            # Increment query counter for new user queries
            self.query_counter += 1
            # Generate query ID if not provided
            if not query_id:
                query_id = f"{self.session_id}_{self.query_counter}"
        
        # For responses (Assistant/Expert), use the last query ID if not provided
        elif not query_id and self.conversation_history:
            # Find the last query ID from a User message
            for p, _, qid in reversed(self.conversation_history):
                if p == "User":
                    query_id = qid
                    break
            
            # If no previous User message found, generate a new ID
            if not query_id:
                self.query_counter += 1
                query_id = f"{self.session_id}_{self.query_counter}"
        
        # Append the message with its query ID
        self.conversation_history.append((poster, message, query_id))
        if poster == "Assistant":
            # Find the user query with the same query_id
            user_query = next(
                (msg for p, msg, qid in reversed(self.conversation_history) 
                 if p == "User" and qid == query_id), 
                None
            )
            if user_query:
                timestamp = datetime.datetime.utcnow().isoformat()
                log_message = {
                    "timestamp": timestamp,
                    "product": self.product.value,
                    "session_id": self.session_id,
                    "query_id": query_id,
                    "query": user_query,
                    "response": message,
                    "thumbs_up": None,
                    "feedback_src": None
                }
                utils.logger.info(f"Responses: {json.dumps(log_message)}")
        self.save_context()

    @staticmethod
    def get_db_folderpath(session_id: str) -> str:
        """get the db folder path"""
        session_id_str = session_id
        return f"{LilLisaServerContext.SPEEDICT_FOLDERPATH}/{session_id_str}"


# def _how_to_use():  # sourcery skip: extract-duplicate-method
#     locale = LOCALE.EN

#     arc = LilLisaServerContext(locale)
#     session_id = arc.get_session_id()
#     print(session_id)
#     print(arc.get_db_folderpath(session_id))
#     arc.add_to_conversation_history("here is my question", "here is my answer")


# if __name__ == "__main__":
#     _how_to_use()
