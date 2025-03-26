"""
access review state
"""

import threading
from enum import Enum
from typing import List, Tuple  # , Union, Dict, Optional

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
        self.conversation_history: List[Tuple[str, str]] = []
        self.user_endorsements: List[int] = []
        self.expert_endorsements: List[int] = []
        # self.conversation_history: List[ChatMessage] = []

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

    def add_to_conversation_history(self, poster: str, message: str):
        """add to the conversation history"""
        self.conversation_history.append((poster, message))
        self.save_context()

    def record_endorsement(self, is_expert: bool, thumbs_up: bool):
        """Record the endorsement for the response in conversation history"""
        index = len(self.conversation_history) - 1
        if is_expert:
            self.expert_endorsements.append(index)
        else:
            self.user_endorsements.append(index)
        self.save_context()

    # def update_conversation_history(self, conversation_history: List[ChatMessage]):
    #     self.conversation_history = conversation_history
    #     self.save_context()

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
