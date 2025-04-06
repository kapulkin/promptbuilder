from pydantic import BaseModel
from promptbuilder.llm_client.messages import Content
from typing import Any, TypeVar, Generic

MessageType = TypeVar("MessageType", bound=Any)

class DialogHistory(Generic[MessageType]):
    def last_messages(self) -> list[Content]:
        raise NotImplementedError("Subclasses must implement this method")

    def add_message(self, message: MessageType):
        raise NotImplementedError("Subclasses must implement this method")

    def clear(self):
        raise NotImplementedError("Subclasses must implement this method")

class InMemoryDialogHistory(DialogHistory[Content]):
    def __init__(self):
        self.messages = []

    def last_messages(self, n: int = 0) -> list[Content]:
        return self.messages[-n:]

    def add_message(self, message: Content):
        self.messages.append(message)

    def clear(self):
        self.messages = []


class Context(BaseModel, Generic[MessageType]):
    dialog_history: DialogHistory[MessageType]

    model_config = {
        "arbitrary_types_allowed": True
    }
