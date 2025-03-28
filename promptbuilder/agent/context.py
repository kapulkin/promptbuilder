from pydantic import BaseModel
from promptbuilder.llm_client.messages import Content

class DialogHistory:
    def last_messages(self) -> list[Content]:
        raise NotImplementedError("Subclasses must implement this method")

    def add_message(self, message: Content):
        raise NotImplementedError("Subclasses must implement this method")

    def clear(self):
        raise NotImplementedError("Subclasses must implement this method")

class InMemoryDialogHistory(DialogHistory):
    def __init__(self):
        self.messages = []

    def last_messages(self, n: int = 0) -> list[Content]:
        return self.messages[-n:]

    def add_message(self, message: Content):
        self.messages.append(message)

    def clear(self):
        self.messages = []


class Context(BaseModel):
    dialog_history: DialogHistory

    model_config = {
        "arbitrary_types_allowed": True
    }
