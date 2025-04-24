import re
import json
from typing import Awaitable, Iterator, AsyncIterator, Literal, overload

from promptbuilder.llm_client.messages import Response, Content, Part, Json, ThinkingConfig, PydanticStructure
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
    def create(self, messages: list[Content], result_type: ResultType = None, thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> Response:
        raise NotImplementedError
    
    @overload
    def create_value(self, messages: list[Content], result_type: None = None, thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> str: ...
    @overload
    def create_value(self, messages: list[Content], result_type: Literal["json"], thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> Json: ...
    @overload
    def create_value(self, messages: list[Content], result_type: type[PydanticStructure], thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> PydanticStructure: ...

    def create_value(self, messages: list[Content], result_type: ResultType = None, thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs):
        response = self.create(
            messages=messages,
            result_type=result_type,
            thinking_config=thinking_config,
            system_message=system_message,
            max_tokens=max_tokens,
            **kwargs,
        )
        if result_type is None:
            return response.text
        else:
            return response.parsed
    
    def create_stream(self, messages: list[Content], system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> Iterator[Response]:
        raise NotImplementedError
    
    @overload
    def from_text(self, prompt: str, result_type: None = None, thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> str: ...
    @overload
    def from_text(self, prompt: str, result_type: Literal["json"], thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> Json: ...
    @overload
    def from_text(self, prompt: str, result_type: type[PydanticStructure], thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> PydanticStructure: ...
    
    def from_text(self, prompt: str, result_type: ResultType = None, thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs):
        return self.create_value(
            messages=[Content(parts=[Part(text=prompt)], role="user")],
            result_type=result_type,
            thinking_config=thinking_config,
            system_message=system_message,
            max_tokens=max_tokens,
            **kwargs,
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
    async def create(self, messages: list[Content], result_type: ResultType = None, thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> Response:
        raise NotImplementedError
    
    @overload
    async def create_value(self, messages: list[Content], result_type: None = None, thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> str: ...
    @overload
    async def create_value(self, messages: list[Content], result_type: Literal["json"], thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> Json: ...
    @overload
    async def create_value(self, messages: list[Content], result_type: type[PydanticStructure], thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> PydanticStructure: ...

    async def create_value(self, messages: list[Content], result_type: ResultType = None, thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs):
        response = await self.create(
            messages=messages,
            result_type=result_type,
            thinking_config=thinking_config,
            system_message=system_message,
            max_tokens=max_tokens,
            **kwargs,
        )
        if result_type is None:
            return response.text
        else:
            return response.parsed
    
    async def create_stream(self, messages: list[Content], system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> Awaitable[AsyncIterator[Response]]:
        raise NotImplementedError
    
    @overload
    async def from_text(self, prompt: str, result_type: None = None, thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> str: ...
    @overload
    async def from_text(self, prompt: str, result_type: Literal["json"], thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> Json: ...
    @overload
    async def from_text(self, prompt: str, result_type: type[PydanticStructure], thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs) -> PydanticStructure: ...
    
    async def from_text(self, prompt: str, result_type: ResultType = None, thinking_config: ThinkingConfig = ThinkingConfig(), system_message: str | None = None, max_tokens: int | None = None, **kwargs):
        return await self.create_value(
            messages=[Content(parts=[Part(text=prompt)], role="user")],
            result_type=result_type,
            thinking_config=thinking_config,
            system_message=system_message,
            max_tokens=max_tokens,
            **kwargs,
        )
