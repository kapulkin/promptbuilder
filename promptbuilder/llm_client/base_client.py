import re
import json
from typing import Iterator, AsyncIterator, Literal, overload

from promptbuilder.llm_client.messages import Response, Content, Part, Tool, ToolConfig, FunctionCall, FunctionCallingConfig, Json, ThinkingConfig, PydanticStructure
import promptbuilder.llm_client.utils as utils


type ResultType = Literal["json"] | type[PydanticStructure] | None


class BaseLLMClient(utils.InheritDecoratorsMixin):
    def __init__(self, decorator_configs: utils.DecoratorConfigs = utils.DecoratorConfigs(), default_max_tokens: int = 8192, **kwargs):
        self._decorator_configs = decorator_configs
        self.default_max_tokens = default_max_tokens
    
    @property
    def model(self) -> str:
        """Return the model identifier used by this LLM client."""
        raise NotImplementedError
    
    def _as_json(self, text: str) -> Json:
        # Remove markdown code block formatting if present
        text = text.strip()
                
        code_block_pattern = r"```(?:json\s)?(.*)```"
        match = re.search(code_block_pattern, text, re.DOTALL)
        
        if match:
            # Use the content inside code blocks
            text = match.group(1).strip()

        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON:\n{text}")

    @utils.retry_cls
    @utils.rpm_limit_cls
    def create(
        self,
        messages: list[Content],
        result_type: ResultType = None,
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        tool_config: ToolConfig = ToolConfig(),
    ) -> Response:
        raise NotImplementedError
    
    @overload
    def create_value(
        self,
        messages: list[Content],
        result_type: None = None,
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: None = None,
        tool_choice_mode: Literal["NONE"] = "NONE",
    ) -> str: ...
    @overload
    def create_value(
        self,
        messages: list[Content],
        result_type: Literal["json"],
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: None = None,
        tool_choice_mode: Literal["NONE"] = "NONE",
    ) -> Json: ...
    @overload
    def create_value(
        self,
        messages: list[Content],
        result_type: type[PydanticStructure],
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: None = None,
        tool_choice_mode: Literal["NONE"] = "NONE",
    ) -> PydanticStructure: ...
    @overload
    def create_value(
        self,
        messages: list[Content],
        result_type: Literal["tools"],
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: list[Tool],
        tool_choice_mode: Literal["ANY"],
    ) -> list[FunctionCall]: ...

    def create_value(
        self,
        messages: list[Content],
        result_type: ResultType | Literal["tools"] = None,
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        tool_choice_mode: Literal["ANY", "NONE"] = "NONE",
    ):
        if result_type == "tools":
            response = self.create(
                messages=messages,
                result_type=None,
                thinking_config=thinking_config,
                system_message=system_message,
                max_tokens=max_tokens,
                tools=tools,
                tool_config=ToolConfig(function_calling_config=FunctionCallingConfig(mode=tool_choice_mode)),
            )
            functions: list[FunctionCall] = []
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.function_call is not None:
                        functions.append(part.function_call)
            return functions

        response = self.create(
            messages=messages,
            result_type=result_type,
            thinking_config=thinking_config,
            system_message=system_message,
            max_tokens=max_tokens,
            tools=tools,
            tool_config=ToolConfig(function_calling_config=FunctionCallingConfig(mode=tool_choice_mode)),
        )
        if result_type is None:
            return response.text
        else:
            return response.parsed
    
    @utils.retry_cls
    @utils.rpm_limit_cls
    def create_stream(self, messages: list[Content], *, system_message: str | None = None, max_tokens: int | None = None) -> Iterator[Response]:
        raise NotImplementedError
    
    @overload
    def from_text(
        self,
        prompt: str,
        result_type: None = None,
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: None = None,
        tool_choice_mode: Literal["NONE"] = "NONE",
    ) -> str: ...
    @overload
    def from_text(
        self,
        prompt: str,
        result_type: Literal["json"],
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: None = None,
        tool_choice_mode: Literal["NONE"] = "NONE",
    ) -> Json: ...
    @overload
    def from_text(
        self,
        prompt: str,
        result_type: type[PydanticStructure],
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: None = None,
        tool_choice_mode: Literal["NONE"] = "NONE",
    ) -> PydanticStructure: ...
    @overload
    def from_text(
        self,
        prompt: str,
        result_type: Literal["tools"],
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: list[Tool],
        tool_choice_mode: Literal["ANY"],
    ) -> list[FunctionCall]: ...
    
    def from_text(
        self,
        prompt: str,
        result_type: ResultType | Literal["tools"] = None,
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        tool_choice_mode: Literal["ANY", "NONE"] = "NONE",
    ):
        return self.create_value(
            messages=[Content(parts=[Part(text=prompt)], role="user")],
            result_type=result_type,
            thinking_config=thinking_config,
            system_message=system_message,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice_mode=tool_choice_mode,
        )


class BaseLLMClientAsync(utils.InheritDecoratorsMixin):
    def __init__(self, decorator_configs: utils.DecoratorConfigs = utils.DecoratorConfigs(), default_max_tokens: int = 8192, **kwargs):
        self._decorator_configs = decorator_configs
        self.default_max_tokens = default_max_tokens

    @property
    def model(self) -> str:
        """Return the model identifier used by this LLM client."""
        raise NotImplementedError
    
    def _as_json(self, text: str) -> Json:
        # Remove markdown code block formatting if present
        text = text.strip()
                
        code_block_pattern = r"```(?:json\s)?(.*)```"
        match = re.search(code_block_pattern, text, re.DOTALL)
        
        if match:
            # Use the content inside code blocks
            text = match.group(1).strip()

        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON:\n{text}")

    @utils.retry_cls_async
    @utils.rpm_limit_cls_async
    async def create(
        self,
        messages: list[Content],
        result_type: ResultType = None,
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        tool_config: ToolConfig = ToolConfig(),
    ) -> Response:
        raise NotImplementedError
    
    @overload
    async def create_value(
        self,
        messages: list[Content],
        result_type: None = None,
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: None = None,
        tool_choice_mode: Literal["NONE"] = "NONE",
    ) -> str: ...
    @overload
    async def create_value(
        self,
        messages: list[Content],
        result_type: Literal["json"],
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: None = None,
        tool_choice_mode: Literal["NONE"] = "NONE",
    ) -> Json: ...
    @overload
    async def create_value(
        self,
        messages: list[Content],
        result_type: type[PydanticStructure],
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: None = None,
        tool_choice_mode: Literal["NONE"] = "NONE",
    ) -> PydanticStructure: ...
    @overload
    async def create_value(
        self,
        messages: list[Content],
        result_type: Literal["tools"],
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: list[Tool],
        tool_choice_mode: Literal["ANY"],
    ) -> list[FunctionCall]: ...

    async def create_value(
        self,
        messages: list[Content],
        result_type: ResultType | Literal["tools"] = None,
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        tool_choice_mode: Literal["ANY", "NONE"] = "NONE",
    ):
        if result_type == "tools":
            response = await self.create(
                messages=messages,
                result_type=None,
                thinking_config=thinking_config,
                system_message=system_message,
                max_tokens=max_tokens,
                tools=tools,
                tool_config=ToolConfig(function_calling_config=FunctionCallingConfig(mode=tool_choice_mode)),
            )
            functions: list[FunctionCall] = []
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.function_call is not None:
                        functions.append(part.function_call)
            return functions

        response = await self.create(
            messages=messages,
            result_type=result_type,
            thinking_config=thinking_config,
            system_message=system_message,
            max_tokens=max_tokens,
            tools=tools,
            tool_config=ToolConfig(function_calling_config=FunctionCallingConfig(mode=tool_choice_mode)),
        )
        if result_type is None:
            return response.text
        else:
            return response.parsed
    
    @utils.retry_cls_async
    @utils.rpm_limit_cls_async
    async def create_stream(self, messages: list[Content], *, system_message: str | None = None, max_tokens: int | None = None) -> AsyncIterator[Response]:
        raise NotImplementedError
    
    @overload
    async def from_text(
        self,
        prompt: str,
        result_type: None = None,
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: None = None,
        tool_choice_mode: Literal["NONE"] = "NONE",
    ) -> str: ...
    @overload
    async def from_text(
        self,
        prompt: str,
        result_type: Literal["json"],
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: None = None,
        tool_choice_mode: Literal["NONE"] = "NONE",
    ) -> Json: ...
    @overload
    async def from_text(
        self,
        prompt: str,
        result_type: type[PydanticStructure],
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: None = None,
        tool_choice_mode: Literal["NONE"] = "NONE",
    ) -> PydanticStructure: ...
    @overload
    async def from_text(
        self,
        prompt: str,
        result_type: Literal["tools"],
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: list[Tool],
        tool_choice_mode: Literal["ANY"],
    ) -> list[FunctionCall]: ...
    
    async def from_text(
        self,
        prompt: str,
        result_type: ResultType | Literal["tools"] = None,
        *,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        tool_choice_mode: Literal["ANY", "NONE"] = "NONE",
    ):
        return await self.create_value(
            messages=[Content(parts=[Part(text=prompt)], role="user")],
            result_type=result_type,
            thinking_config=thinking_config,
            system_message=system_message,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice_mode=tool_choice_mode,
        )
