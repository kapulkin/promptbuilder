from typing import List, Dict, Any, Optional, Callable, Type, Generic, TypeVar, Literal, Union
from promptbuilder.llm_client import BaseLLMClient
from promptbuilder.agent.message import Message
from promptbuilder.agent.tool import Tool
from promptbuilder.agent.context import Context
from promptbuilder.prompt_builder import PromptBuilder
from pydantic import Field, create_model
import logging

logger = logging.getLogger(__name__)

ContextType = TypeVar("ContextType", bound=Context)

class Agent(Generic[ContextType]):
    def __init__(self, llm_client: BaseLLMClient, context: ContextType):
        self.llm_client = llm_client
        self.context = context

    async def __call__(self, user_message: Message, **kwargs: Any) -> str:
        raise NotImplementedError("Agent is not implemented")

    def system_message(self) -> Message:
        raise NotImplementedError("System message is not implemented")

    def _answer_with_llm(self):
        return self.llm_client.create(
            messages=[msg.llm_dict() for msg in [self.system_message()] + self.context.history()]
        )

class AgentRouter(Agent[ContextType]):
    def __init__(self, llm_client: BaseLLMClient, context: ContextType, tools: list[Tool] = []):
        super().__init__(llm_client, context)
        self.tools = tools
        self.last_used_tool = None
    
    async def __call__(self, user_message: Message, **kwargs: Any) -> str:
        self.context.add_message(user_message)

        if len(self.tools) > 0:
            response = self.llm_client.create_structured(
                messages=[msg.llm_dict() for msg in [self.system_message()] + self.context.history()]
            )

            tool_name = response["tool_name"]
            tool = next((tool for tool in self.tools if tool.name == tool_name), None)
            if tool is None:
                raise ValueError(f"Tool {tool_name} not found")
            args = response.get("args", None)
            logger.debug("Tool %s called with args: %s", tool_name, args)
            response = await tool(user_message, args, self.context)
            logger.debug("Tool %s response: %s", tool_name, response)
            self.last_used_tool = tool
        else:
            response = self._answer_with_llm()

        if response is not None and type(response) is str:
            self.context.messages.append(Message(role="assistant", content=response))
        
        logger.debug("Agent response: %s", response)
        return response
    
    def description(self) -> str:
        raise NotImplementedError("Description is not implemented")

    def system_message(self) -> Message:
        builder = PromptBuilder() \
            .paragraph(self.description())
        
        if len(self.tools) > 0:
            builder.header("Tools") \
                .paragraph(f"You can use the tools below.")

            for tool in self.tools:
                builder \
                    .paragraph(f"\n{tool.name}:\n{tool.description}")
                
                if tool.args is not None:
                    builder \
                        .paragraph("<arguments>") \
                        .structure(tool.args) \
                        .paragraph("</arguments>")
            

            tools_args = [tool.args for tool in self.tools if tool.args is not None]
            if len(tools_args) > 0:
                tool_call_model = create_model(
                    "ToolCall",
                    tool_name=(Literal[tuple(tool.name for tool in self.tools)], Field(..., description="Selected tool name")),
                    args=(tools_args, Field(..., description="Arguments for the selected tool"))
                )
            else:
                tool_call_model = create_model(
                    "ToolCall",
                    tool_name=(Literal[tuple(tool.name for tool in self.tools)], Field(..., description="Selected tool name")),
                )

            builder \
                .paragraph("Choose the tool that is most relevant to the user's message and provide the arguments for the tool.") \
                .set_structured_output(tool_call_model)
        prompt = builder.build()

        return Message(
            role="system",
            content=prompt.render()
        )

    def tool(self, description: str, args_model: Optional[Type] = None):
        def decorator(func: Callable[[Message, Any, Context, ...], Any]):
            tool = Tool(
                name=func.__name__,
                description=description,
                args=args_model,
                function=func
            )
            self.tools.append(tool)

            return func
        return decorator
