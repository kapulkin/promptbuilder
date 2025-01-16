from pydantic import BaseModel
from promptbuilder.agent.message import Message

class Context(BaseModel):
    messages: list[Message] = []

    def history(self, length: int = 20) -> list[Message]:
        return self.messages[-length:]

    def add_message(self, message: Message):
        self.messages.append(message)

    def clear(self):
        self.messages = []


