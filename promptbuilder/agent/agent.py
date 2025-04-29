import logging
from enum import Enum
from typing import Any, Callable, Generic, TypeVar, ParamSpec, Literal

from promptbuilder.llm_client import BaseLLMClient, BaseLLMClientAsync, Response, Content, Part, ToolConfig
from promptbuilder.agent.tool import CallableTool
from promptbuilder.agent.context import Context
from promptbuilder.prompt_builder import PromptBuilder
from promptbuilder.agent.utils import run_async


logger = logging.getLogger(__name__)

MessageType = TypeVar("MessageType", bound=Any)
ContextType = TypeVar("ContextType", bound=Context[Any])
RouteResponse = TypeVar("RouteResponse", str, None)
P = ParamSpec("P")
T = TypeVar("T")



class MessageFormat(Enum):
    MESSAGES_LIST = 0
    ONE_MESSAGE = 1

class Agent(Generic[MessageType, ContextType]):
    def __init__(self, llm_client: BaseLLMClient | BaseLLMClientAsync, context: ContextType, message_format: MessageFormat = MessageFormat.MESSAGES_LIST):
        self.llm_client = llm_client
        self.context = context
        self.message_format = message_format

    async def __call__(self, **kwargs: Any):
        raise NotImplementedError("Agent is not implemented")

    def system_message(self, **kwargs: Any) -> str:
        raise NotImplementedError("System message is not implemented")

    def _messages_to_one_message(self, messages: list[Content]) -> Content:
        text = "\n".join([
            f"{message.role}: {message.parts[0].text}"
            for message in messages
        ])
        return Content(parts=[Part(text=text)], role="user")

    def _formatted_messages(self, messages: list[Content]) -> list[Content]:
        if self.message_format == MessageFormat.ONE_MESSAGE:
            return [self._messages_to_one_message(messages)]
        else:
            return messages

    async def _answer_with_llm(self, **kwargs: Any) -> Response:
        messages = self._formatted_messages(self.context.dialog_history.last_messages())
        return await run_async(self.llm_client.create,
            messages=messages,
            system_message=self.system_message(**kwargs),
            **kwargs,
        )


class AgentRouter(Agent[MessageType, ContextType]):
    def __init__(self, llm_client: BaseLLMClient | BaseLLMClientAsync, context: ContextType, message_format: MessageFormat = MessageFormat.MESSAGES_LIST):
        super().__init__(llm_client, context, message_format)
        self.tr_names: list[str] = []
        self.tools: dict[str, CallableTool[..., Response]] = {}
        self.routes: dict[str, CallableTool[..., RouteResponse]] = {}
        self.last_used_tr_name = None
    
    async def __call__(self, user_message: MessageType, tr_choice_mode: Literal["ANY", "AUTO", "FIRST"] = "FIRST", trs_to_exclude: set[str] = set(), **kwargs: Any):
        if len(trs_to_exclude) == 0:
            self.context.dialog_history.add_message(user_message)

        callable_trs = [self.tools.get(name) or self.routes.get(name) for name in self.tr_names if name not in trs_to_exclude]
        trs = [callable_tr.tool for callable_tr in callable_trs]

        messages = self._formatted_messages(self.context.dialog_history.last_messages())
        response = await run_async(self.llm_client.create,
            messages=messages,
            system_message=self.system_message(callable_trs=callable_trs),
            tools=trs,
            tool_config=ToolConfig(function_calling_config={"mode": "ANY" if tr_choice_mode == "ANY" else "AUTO"}),
        )
        content = response.candidates[0].content
        
        for part in content.parts:
            if part.function_call is None:
                if part.text is not None:
                    self.context.dialog_history.add_message(Content(parts=[Part(text=part.text)], role="model"))
                if part.thought is not None:
                    self.context.dialog_history.add_message(Content(parts=[Part(thought=part.thought)], role="model"))
            else:
                tr_name = part.function_call.name
                args = part.function_call.args
                if args is None:
                    args = {}

                route = self.routes.get(tr_name)
                if route is not None:
                    self.last_used_tr_name = tr_name
                    logger.debug("Route %s called with args: %s", tr_name, args)
                    merged_args = {**kwargs, **args}
                    result = await route(**merged_args)
                    logger.debug("Route %s result: %s", tr_name, result)
                    trs_to_exclude = trs_to_exclude | {tr_name}
                    if result is not None:
                        self.context.dialog_history.add_message(Content(parts=[Part(text=result)], role="model"))
                    if tr_choice_mode == "FIRST":
                        return
                
                tool = self.tools.get(tr_name)
                if tool is not None:
                    self.last_used_tr_name = tr_name
                    self.context.dialog_history.add_message(content)
                    logger.debug("Tool %s called with args: %s", tr_name, args)
                    tool_response = await tool(**args)
                    logger.debug("Tool %s response: %s", tr_name, tool_response)
                    self.context.dialog_history.add_message(tool_response.candidates[0].content)
                    trs_to_exclude = trs_to_exclude | {tr_name}
                    if tr_choice_mode == "FIRST":
                        return await self(user_message, trs_to_exclude=trs_to_exclude, tr_choice_mode=tr_choice_mode, **kwargs)

                if route is None and tool is None:
                    raise ValueError(f"Tool/route {tr_name} not found")
    
    def description(self) -> str:
        raise NotImplementedError("Description is not implemented")

    def system_message(self, callable_trs: list[CallableTool] = [], **kwargs: Any) -> str:
        builder = PromptBuilder() \
            .paragraph(self.description())
        
        if len(callable_trs) > 0:
            builder.header("Tools") \
                .paragraph(f"You can use the tools below.")

            for callable_tr in callable_trs:
                name = callable_tr.name
                description = callable_tr.tool.function_declarations[0].description

                indent = " " * 4
                description_with_indent = "\n".join([f"{indent}{line}" for line in description.splitlines()])

                builder \
                    .paragraph(f"\n  {name}\n{description_with_indent}")
                
                args = {name: type for name, type in callable_tr.function.__annotations__.items() if name != "return"}
                if len(args) > 0:
                    builder.paragraph(f"{indent}Parameters:")
                    for arg_name, arg_type in args.items():
                        arg_description = callable_tr.arg_descriptions.get(arg_name, None)
                        if arg_description is not None:
                            builder.paragraph(f"{indent}{arg_name} {arg_description}")
                        else:
                            builder.paragraph(f"{indent}{arg_name}")
            
        prompt = builder.build()

        return prompt.render()

    def add_tool(self, func: Callable[P, Response], arg_descriptions: dict[str, str] = {}) -> CallableTool[P, Response]:
        tool = CallableTool(
            function=func,
            arg_descriptions=arg_descriptions,
        )
        if tool.name in self.tr_names:
            raise ValueError("")
        self.tools[tool.name] = tool
        self.tr_names.append(tool.name)
        return tool

    def add_route(self, func: Callable[P, RouteResponse], arg_descriptions: dict[str, str] = {}) -> CallableTool[P, RouteResponse]:
        route = CallableTool(
            function=func,
            arg_descriptions=arg_descriptions,
        )
        if route.name in self.tr_names:
            raise ValueError("")
        self.routes[route.name] = route
        self.tr_names.append(route.name)
        return route
    
    def remove_tool(self, name: str):
        if name in self.tools:
            self.tools.pop(name)
            while name in self.tr_names:
                self.tr_names.remove(name)
    
    def remove_route(self, name: str):
        if name in self.routes:
            self.routes.pop(name)
            while name in self.tr_names:
                self.tr_names.remove(name)

    def tool(self, arg_descriptions: dict[str, str] = {}) -> Callable[[Callable[P, Response]], Callable[P, Response]]:
        def decorator(func: Callable[P, Response]) -> Callable[P, Response]:
            self.add_tool(func, arg_descriptions)
            return func
        return decorator
    
    def route(self, arg_descriptions: dict[str, str] = {}) -> Callable[[Callable[P, RouteResponse]], Callable[P, RouteResponse]]:
        def decorator(func: Callable[P, RouteResponse]) -> Callable[P, RouteResponse]:
            self.add_route(func, arg_descriptions)
            return func
        return decorator
