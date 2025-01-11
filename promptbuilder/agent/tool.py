from typing import Type, Callable, Any, Optional
from pydantic import BaseModel
from viete.project.chat_agent.context import Context
from viete.project.chat_agent.message import Message


class Tool(BaseModel):
    name: str
    description: str
    args: Type | None
    function: Callable[[Message, Any, Context, ...], Any]

    async def __call__(self, message: Message, data: Any, context: Context, **kwargs: Any) -> Any:
        return await self.function(message, data, context, **kwargs)
