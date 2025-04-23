import re
import json
from typing import Awaitable, Iterator, AsyncIterator, Literal, overload

from promptbuilder.llm_client.messages import Response, Content, Part, Json, PydanticStructure
import promptbuilder.llm_client.utils as utils


type ResultType = Literal["json"] | type[PydanticStructure] | None


class BaseLLMClient(utils.InheritDecoratorsMixin):
    user_tag: str = "user"
    assistant_tag: str = "model"

    def __init__(self, decorator_configs: utils.DecoratorConfigs = utils.DecoratorConfigs(), **kwargs):
        self._decorator_configs = decorator_configs
    
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
    def generate_response(self, messages: list[Content], result_type: ResultType = None, use_thinking: bool = False, system_message: str | None = None, **kwargs) -> Response:
        raise NotImplementedError
    
    @overload
    def create(self, messages: list[Content], result_type: None = None, use_thinking: bool = False, system_message: str | None = None, **kwargs) -> str: ...
    @overload
    def create(self, messages: list[Content], result_type: Literal["json"], use_thinking: bool = False, system_message: str | None = None, **kwargs) -> Json: ...
    @overload
    def create(self, messages: list[Content], result_type: type[PydanticStructure], use_thinking: bool = False, system_message: str | None = None, **kwargs) -> PydanticStructure: ...

    def create(self, messages: list[Content], result_type: ResultType = None, use_thinking: bool = False, system_message: str | None = None, **kwargs):
        response = self.generate_response(
            messages=messages,
            result_type=result_type,
            use_thinking=use_thinking,
            system_message=system_message,
            **kwargs,
        )
        if result_type is None:
            return response.text
        else:
            return response.parsed
    
    def create_stream(self, messages: list[Content], system_message: str | None = None, **kwargs) -> Iterator[Response]:
        raise NotImplementedError
    
    @overload
    def from_text(self, prompt: str, result_type: None = None, use_thinking: bool = False, system_message: str | None = None, **kwargs) -> str: ...
    @overload
    def from_text(self, prompt: str, result_type: Literal["json"], use_thinking: bool = False, system_message: str | None = None, **kwargs) -> Json: ...
    @overload
    def from_text(self, prompt: str, result_type: type[PydanticStructure], use_thinking: bool = False, system_message: str | None = None, **kwargs) -> PydanticStructure: ...
    
    def from_text(self, prompt: str, result_type: ResultType = None, use_thinking: bool = False, system_message: str | None = None, **kwargs):
        return self.create(
            messages=[Content(parts=[Part(text=prompt)], role=BaseLLMClient.user_tag)],
            result_type=result_type,
            use_thinking=use_thinking,
            system_message=system_message,
            **kwargs,
        )


class BaseLLMClientAsync(utils.InheritDecoratorsMixin):
    user_tag: str = "user"
    assistant_tag: str = "model"
    
    def __init__(self, decorator_configs: utils.DecoratorConfigs = utils.DecoratorConfigs(), **kwargs):
        self._decorator_configs = decorator_configs

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
    async def generate_response(self, messages: list[Content], result_type: ResultType = None, use_thinking: bool = False, system_message: str | None = None, **kwargs) -> Response:
        raise NotImplementedError
    
    @overload
    async def create(self, messages: list[Content], result_type: None = None, use_thinking: bool = False, system_message: str | None = None, **kwargs) -> str: ...
    @overload
    async def create(self, messages: list[Content], result_type: Literal["json"], use_thinking: bool = False, system_message: str | None = None, **kwargs) -> Json: ...
    @overload
    async def create(self, messages: list[Content], result_type: type[PydanticStructure], use_thinking: bool = False, system_message: str | None = None, **kwargs) -> PydanticStructure: ...

    async def create(self, messages: list[Content], result_type: ResultType = None, use_thinking: bool = False, system_message: str | None = None, **kwargs):
        response = await self.generate_response(
            messages=messages,
            result_type=result_type,
            use_thinking=use_thinking,
            system_message=system_message,
            **kwargs,
        )
        if result_type is None:
            return response.text
        else:
            return response.parsed
    
    async def create_stream(self, messages: list[Content], system_message: str | None = None, **kwargs) -> Awaitable[AsyncIterator[Response]]:
        raise NotImplementedError
    
    @overload
    async def from_text(self, prompt: str, result_type: None = None, use_thinking: bool = False, system_message: str | None = None, **kwargs) -> str: ...
    @overload
    async def from_text(self, prompt: str, result_type: Literal["json"], use_thinking: bool = False, system_message: str | None = None, **kwargs) -> Json: ...
    @overload
    async def from_text(self, prompt: str, result_type: type[PydanticStructure], use_thinking: bool = False, system_message: str | None = None, **kwargs) -> PydanticStructure: ...
    
    async def from_text(self, prompt: str, result_type: ResultType = None, use_thinking: bool = False, system_message: str | None = None, **kwargs):
        return await self.create(
            messages=[Content(parts=[Part(text=prompt)], role=BaseLLMClient.user_tag)],
            result_type=result_type,
            use_thinking=use_thinking,
            system_message=system_message,
            **kwargs,
        )
