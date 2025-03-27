from pydantic import BaseModel
from promptbuilder.llm_client.messages import Content

class Context(BaseModel):
    messages: list[Content] = []
    history_length: int = 0

    def history(self) -> list[Content]:
        return self.messages[-self.history_length:]

    def add_message(self, message: Content):
        self.messages.append(message)

    def clear(self):
        self.messages = []
