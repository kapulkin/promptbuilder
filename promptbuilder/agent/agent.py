from typing import List, Dict, Any, Optional, Callable, Type, Generic, TypeVar, Literal, Union
from promptbuilder.llm_client import BaseLLMClient, BaseLLMClientAsync, GoogleLLMClient, GoogleLLMClientAsync
from promptbuilder.agent.tool import CallableTool
from promptbuilder.agent.context import Context
from promptbuilder.prompt_builder import PromptBuilder
from pydantic import Field, create_model
from promptbuilder.llm_client.messages import Content, Part, Tool
from promptbuilder.agent.utils import run_async
import logging

logger = logging.getLogger(__name__)

ContextType = TypeVar("ContextType", bound=Context)

class Agent(Generic[ContextType]):
    def __init__(self, llm_client: BaseLLMClient | BaseLLMClientAsync, context: ContextType):
        self.llm_client = llm_client
        self.context = context

    async def __call__(self, **kwargs: Any) -> str:
        raise NotImplementedError("Agent is not implemented")

    def system_message(self, **kwargs: Any) -> str:
        raise NotImplementedError("System message is not implemented")

    async def _answer_with_llm(self, **kwargs: Any):
        return await run_async(self.llm_client.create,
            messages=[Content(parts=[Part(text=msg.content)], role=msg.role) for msg in self.context.dialog_history.last_messages()],
            system_message=self.system_message(**kwargs),
            **kwargs
        )


class AgentRouter(Agent[ContextType]):
    def __init__(self, llm_client: BaseLLMClient, context: ContextType):
        super().__init__(llm_client, context)
        self.callable_tools: dict[str, CallableTool] = {}
        self.routes: dict[str, CallableTool] = {}
        self.last_used_tool_name = None
    
    async def __call__(self, user_message: str, tools_to_exclude: set[str] = set(), **kwargs: Any) -> str:
        if len(tools_to_exclude) > 0:
            self.context.dialog_history.add_message(Content(parts=[Part(text=user_message)], role=BaseLLMClient.user_tag))

        callable_tools = [callable_tool for callable_tool in self.callable_tools.values() if callable_tool.name not in tools_to_exclude]
        tools = [callable_tool.tool for callable_tool in callable_tools]

        response = await self.llm_client.create(
            messages=[Content(parts=[Part(text=msg.content)], role=msg.role) for msg in self.context.dialog_history.last_messages()],
            system_message=self.system_message(callable_tools=callable_tools),
            tools=tools
        )
        content = response.candidates[0].content
        tool_name = None
        args = None
        for part in content.parts:
            if part.function_call is not None:
                tool_name = part.function_call.name
                args = part.function_call.arguments
                break

        if tool_name is not None:
            route = self.routes.get(tool_name)
            if route is not None:
                self.last_used_tool_name = tool_name
                logger.debug("Route %s called with args: %s", tool_name, args)
                result = await route(**args)
                logger.debug("Route %s result: %s", tool_name, result)
                return result
            
            callable_tool = self.callable_tools.get(tool_name)
            if callable_tool is not None:
                self.last_used_tool_name = tool_name
                self.context.dialog_history.add_message(content)
                logger.debug("Tool %s called with args: %s", tool_name, args)
                tool_response = await callable_tool(**args)
                logger.debug("Tool %s response: %s", tool_name, tool_response)
                self.context.dialog_history.add_message(tool_response.candidates[0].content)
                tools_to_exclude = tools_to_exclude | {tool_name}
                return await self(user_message, tools_to_exclude=tools_to_exclude, **kwargs)
            
            raise ValueError(f"Tool {tool_name} not found")
        self.context.dialog_history.add_message(content)
    
    def description(self) -> str:
        raise NotImplementedError("Description is not implemented")

    def system_message(self, **kwargs: Any) -> str:
        callable_tools = kwargs.get("callable_tools", [])

        builder = PromptBuilder() \
            .paragraph(self.description())
        
        if len(callable_tools) > 0:
            builder.header("Tools") \
                .paragraph(f"You can use the tools below.")

            for callable_tool in callable_tools:
                name = callable_tool.name
                description = callable_tool.tool.function_declarations[0].description

                indent = " " * 4
                description_with_indent = "\n".join([f"{indent}{line}" for line in description.splitlines()])

                builder \
                    .paragraph(f"\n  {name}\n{description_with_indent}")
                
                args = {name: type for name, type in callable_tool.function.__annotations__.items() if name != "return"}
                if len(args) > 0:
                    builder.paragraph(f"{indent}Parameters:")
                    for arg_name, arg_type in args.items():
                        arg_description = callable_tool.arg_descriptions.get(arg_name, None)
                        if arg_description is not None:
                            builder.paragraph(f"{indent}{arg_name} {arg_description}")
                        else:
                            builder.paragraph(f"{indent}{arg_name}")
            
        prompt = builder.build()

        return prompt.render()

    def add_tool(self, func: Callable[[...], Any], arg_descriptions: dict[str, str] = {}):
        callable_tool = CallableTool(
            function=func,
            arg_descriptions=arg_descriptions,
        )
        self.callable_tools[callable_tool.name] = callable_tool
        return callable_tool

    def add_route(self, func: Callable[[...], Any], arg_descriptions: dict[str, str] = {}):
        callable_tool = self.tool(func, arg_descriptions)
        self.routes[callable_tool.name] = callable_tool
        return callable_tool

    def tool(self, arg_descriptions: dict[str, str] = {}):
        def decorator(func: Callable[[...], Any]):
            self.add_tool(func, arg_descriptions)
            return func
        return decorator
    
    def route(self, arg_descriptions: dict[str, str] = {}):
        def decorator(func: Callable[[...], Any]):
            self.add_route(func, arg_descriptions)
            return func
        return decorator

