import json
import logging
import os
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

HISTORY_DIR = "data/json"
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.json")

os.makedirs(HISTORY_DIR, exist_ok=True)


def message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, HumanMessage):
        return {"type": "human", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"type": "ai", "content": message.content}
    else:
        logging.warning(
            f"FILE: app/history.py INFO: faced error in message_to_dict DETAILS: message = {str(message)}"
        )
        return {"type": "unknown", "content": str(message)}


def dict_to_message(message_dict: dict) -> BaseMessage:
    message_type = message_dict.get("type")
    content = message_dict.get("content", "")
    if message_type == "human":
        return HumanMessage(content=content)
    elif message_type == "ai":
        return AIMessage(content=content)
    else:
        logging.warning(
            f"FILE: app/history.py INFO: faced error in dict_to_message DETAILS: message = {str(message_dict)}"
        )
        return AIMessage(
            content=f"Error: Could not load message of type {message_type}"
        )


def get_history() -> dict:
    try:
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "w") as file:
                json.dump({}, file)
            return {}

        with open(HISTORY_FILE, "r") as file:
            content = json.load(file)
    except Exception as e:
        content = {}
        logging.error(
            f"FILE: app/history.py INFO: faced exception opening/reading json file '{HISTORY_FILE}' DETAILS: {e}"
        )

    return content if isinstance(content, dict) else {}


def get_all_conversation_ids() -> list:
    content = get_history()
    return list(content.keys())


def get_history_by_id(id: str) -> list:
    content = get_history()
    conversations = content.get(id, [])
    chat_history = [
        dict_to_message(msg_dict)
        for msg_dict in conversations
        if isinstance(msg_dict, dict)
    ]
    return chat_history


def save_history(id: str, conversations: list) -> None:
    content = get_history()
    chat_history = [
        message_to_dict(msg)
        for msg in conversations
        if message_to_dict(msg) is not None
    ]
    content[id] = chat_history

    try:
        with open(HISTORY_FILE, "w") as file:
            json.dump(content, file, indent=4)
    except Exception as e:
        logging.error(
            f"FILE: app/history.py INFO: faced exception saving json file '{HISTORY_FILE}' DETAILS: {e}"
        )


def create_new_conversation(id: str) -> None:
    """Explicitly creates an empty entry for a new conversation ID."""
    content = get_history()
    if id not in content:
        content[id] = []
        try:
            with open(HISTORY_FILE, "w") as file:
                json.dump(content, file, indent=4)
        except Exception as e:
            logging.error(
                f"FILE: app/history.py INFO: faced exception creating new conversation entry '{id}' DETAILS: {e}"
            )
