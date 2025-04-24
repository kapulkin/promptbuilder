import os
from typing import Awaitable, AsyncIterator, Iterator

from pydantic import BaseModel
from anthropic import Anthropic, AsyncAnthropic, Stream, AsyncStream
from anthropic.types import RawMessageStreamEvent

from promptbuilder.llm_client.base_client import BaseLLMClient, BaseLLMClientAsync, ResultType
from promptbuilder.llm_client.messages import Response, Content, Candidate, UsageMetadata, Part, ThinkingConfig
from promptbuilder.llm_client.base_configs import DecoratorConfigs, base_decorator_configs, base_default_max_tokens_configs
from promptbuilder.prompt_builder import schema_to_ts


class AnthropicStreamIterator:
    def __init__(self, anthropic_iterator: Stream[RawMessageStreamEvent]):
        self._anthropic_iterator = anthropic_iterator

    def __next__(self) -> Response:
        while True:
            next_event = self._anthropic_iterator.__next__()
            if next_event.type == "content_block_delta":
                parts = [Part(text=next_event.delta.text)]
                return Response(candidates=[Candidate(content=Content(parts=parts, role="model"))])

    def __iter__(self) -> Iterator[Response]:
        for next_event in self._anthropic_iterator:
            if next_event.type == "content_block_delta":
                parts = [Part(text=next_event.delta.text)]
                yield Response(candidates=[Candidate(content=Content(parts=parts, role="model"))])


class AnthropicLLMClient(BaseLLMClient):
    def __init__(
        self,
        model: str,
        api_key: str = os.getenv("ANTHROPIC_API_KEY"),
        decorator_configs: DecoratorConfigs | None = None,
        default_max_tokens: int | None = None,
        **kwargs,
    ):
        if decorator_configs is None:
            decorator_configs = base_decorator_configs["anthropic:" + model]
        if default_max_tokens is None:
            default_max_tokens = base_default_max_tokens_configs["anthropic:" + model]
        super().__init__(decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        self.client = Anthropic(api_key=api_key)
        self._model = model
    
    @property
    def model(self) -> str:
        return "anthropic:" + self._model
    
    def create(
        self,
        messages: list[Content],
        result_type: ResultType = None,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Response:
        anthropic_messages: list[dict[str, str]] = []
        for message in messages:
            if message.role == "user":
                anthropic_messages.append({"role": "user", "content": message.as_str()})
            elif message.role == "model":
                anthropic_messages.append({"role": "assistant", "content": message.as_str()})
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        
        anthropic_kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        
        if thinking_config.include_thoughts:
            anthropic_kwargs["thinking"] = {
                "budget_tokens": thinking_config.thinking_budget,
                "type": "enabled",
            }
        else:
            anthropic_kwargs["thinking"] = {
                "type": "disabled",
            }
        
        if system_message is not None:
            anthropic_kwargs["system"] = system_message
        
        if result_type is None:
            response = self.client.messages.create(**anthropic_kwargs)
            
            parts: list[Part] = []
            for content in response.content:
                if content.type == "thinking":
                    parts.append(Part(text=content.thinking, thought=True))
                elif content.type == "text":
                    parts.append(Part(text=content.text))
            
            return Response(
                candidates=[Candidate(content=Content(parts=parts, role="model"))],
                usage_metadata=UsageMetadata(
                    candidates_token_count=response.usage.output_tokens,
                    prompt_token_count=response.usage.input_tokens,
                    total_token_count=response.usage.output_tokens + response.usage.input_tokens,
                ),
            )
        elif result_type == "json":
            response = self.client.messages.create(**anthropic_kwargs)
            parts: list[Part] = []
            text = ""
            for content in response.content:
                if content.type == "thinking":
                    parts.append(Part(text=content.thinking, thought=True))
                elif content.type == "text":
                    text += content.text + "\n"
                    parts.append(Part(text=content.text))
            parsed = self._as_json(text)
            
            return Response(
                candidates=[Candidate(content=Content(parts=parts, role="model"))],
                usage_metadata=UsageMetadata(
                    candidates_token_count=response.usage.output_tokens,
                    prompt_token_count=response.usage.input_tokens,
                    total_token_count=response.usage.output_tokens + response.usage.input_tokens,
                ),
                parsed=parsed,
            )
        elif isinstance(result_type, type(BaseModel)):
            message_with_structure = f"Return result in a following JSON structure:\n"
            message_with_structure += f"{schema_to_ts(result_type)}\n"
            message_with_structure += "Your output should consist solely of the JSON object, with no additional text."
            anthropic_kwargs["messages"].append({"role": "user", "content": message_with_structure})
            
            response = self.client.messages.create(**anthropic_kwargs)
            parts: list[Part] = []
            text = ""
            for content in response.content:
                if content.type == "thinking":
                    parts.append(Part(text=content.thinking, thought=True))
                elif content.type == "text":
                    text += content.text + "\n"
                    parts.append(Part(text=content.text))
            parsed = self._as_json(text)
            parsed_pydantic = result_type.model_construct(**parsed)
            
            return Response(
                candidates=[Candidate(content=Content(parts=parts, role="model"))],
                usage_metadata=UsageMetadata(
                    candidates_token_count=response.usage.output_tokens,
                    prompt_token_count=response.usage.input_tokens,
                    total_token_count=response.usage.output_tokens + response.usage.input_tokens,
                ),
                parsed=parsed_pydantic,
            )
    
    def create_stream(
        self,
        messages: list[Content],
        system_message: str | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Iterator[Response]:
        anthropic_messages: list[dict[str, str]] = []
        for message in messages:
            if message.role == "user":
                anthropic_messages.append({"role": "user", "content": message.as_str()})
            elif message.role == "model":
                anthropic_messages.append({"role": "assistant", "content": message.as_str()})
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        
        anthropic_kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "stream": True,
        }
        
        if system_message is not None:
            anthropic_kwargs["system"] = system_message
        
        anthropic_iterator = self.client.messages.create(**anthropic_kwargs)
        return AnthropicStreamIterator(anthropic_iterator)


class AnthropicStreamIteratorAsync:
    def __init__(self, anthropic_iterator: AsyncStream[RawMessageStreamEvent]):
        self._anthropic_iterator = anthropic_iterator

    async def __anext__(self) -> Response:
        while True:
            next_event = await self._anthropic_iterator.__anext__()
            if next_event.type == "content_block_delta":
                parts = [Part(text=next_event.delta.text)]
                return Response(candidates=[Candidate(content=Content(parts=parts, role="model"))])

    async def __aiter__(self) -> AsyncIterator[Response]:
        async for next_event in self._anthropic_iterator:
            if next_event.type == "content_block_delta":
                parts = [Part(text=next_event.delta.text)]
                yield Response(candidates=[Candidate(content=Content(parts=parts, role="model"))])


class AnthropicLLMClientAsync(BaseLLMClientAsync):
    def __init__(
        self,
        model: str,
        api_key: str = os.getenv("ANTHROPIC_API_KEY"),
        decorator_configs: DecoratorConfigs | None = None,
        default_max_tokens: int | None = None,
        **kwargs,
    ):
        if decorator_configs is None:
            decorator_configs = base_decorator_configs["anthropic:" + model]
        if default_max_tokens is None:
            default_max_tokens = base_default_max_tokens_configs["anthropic:" + model]
        super().__init__(decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        self.client = AsyncAnthropic(api_key=api_key)
        self._model = model
    
    @property
    def model(self) -> str:
        return "anthropic:" + self._model
    
    async def create(
        self,
        messages: list[Content],
        result_type: ResultType = None,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Response:
        anthropic_messages: list[dict[str, str]] = []
        for message in messages:
            if message.role == "user":
                anthropic_messages.append({"role": "user", "content": message.as_str()})
            elif message.role == "model":
                anthropic_messages.append({"role": "assistant", "content": message.as_str()})
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        
        anthropic_kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        
        if thinking_config.include_thoughts:
            anthropic_kwargs["thinking"] = {
                "budget_tokens": thinking_config.thinking_budget,
                "type": "enabled",
            }
        else:
            anthropic_kwargs["thinking"] = {
                "type": "disabled",
            }
        
        if system_message is not None:
            anthropic_kwargs["system"] = system_message
        
        if result_type is None:
            response = await self.client.messages.create(**anthropic_kwargs)
            
            parts: list[Part] = []
            for content in response.content:
                if content.type == "thinking":
                    parts.append(Part(text=content.thinking, thought=True))
                elif content.type == "text":
                    parts.append(Part(text=content.text))
            
            return Response(
                candidates=[Candidate(content=Content(parts=parts, role="model"))],
                usage_metadata=UsageMetadata(
                    candidates_token_count=response.usage.output_tokens,
                    prompt_token_count=response.usage.input_tokens,
                    total_token_count=response.usage.output_tokens + response.usage.input_tokens,
                ),
            )
        elif result_type == "json":
            response = await self.client.messages.create(**anthropic_kwargs)
            parts: list[Part] = []
            text = ""
            for content in response.content:
                if content.type == "thinking":
                    parts.append(Part(text=content.thinking, thought=True))
                elif content.type == "text":
                    text += content.text + "\n"
                    parts.append(Part(text=content.text))
            parsed = self._as_json(text)
            
            return Response(
                candidates=[Candidate(content=Content(parts=parts, role="model"))],
                usage_metadata=UsageMetadata(
                    candidates_token_count=response.usage.output_tokens,
                    prompt_token_count=response.usage.input_tokens,
                    total_token_count=response.usage.output_tokens + response.usage.input_tokens,
                ),
                parsed=parsed,
            )
        elif isinstance(result_type, type(BaseModel)):
            message_with_structure = f"Return result in a following JSON structure:\n"
            message_with_structure += f"{schema_to_ts(result_type)}\n"
            message_with_structure += "Your output should consist solely of the JSON object, with no additional text."
            anthropic_kwargs["messages"].append({"role": "user", "content": message_with_structure})
            
            response = await self.client.messages.create(**anthropic_kwargs)
            parts: list[Part] = []
            text = ""
            for content in response.content:
                if content.type == "thinking":
                    parts.append(Part(text=content.thinking, thought=True))
                elif content.type == "text":
                    text += content.text + "\n"
                    parts.append(Part(text=content.text))
            parsed = self._as_json(text)
            parsed_pydantic = result_type.model_construct(**parsed)
            
            return Response(
                candidates=[Candidate(content=Content(parts=parts, role="model"))],
                usage_metadata=UsageMetadata(
                    candidates_token_count=response.usage.output_tokens,
                    prompt_token_count=response.usage.input_tokens,
                    total_token_count=response.usage.output_tokens + response.usage.input_tokens,
                ),
                parsed=parsed_pydantic,
            )
    
    async def create_stream(
        self,
        messages: list[Content],
        system_message: str | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> Awaitable[AsyncIterator[Response]]:
        anthropic_messages: list[dict[str, str]] = []
        for message in messages:
            if message.role == "user":
                anthropic_messages.append({"role": "user", "content": message.as_str()})
            elif message.role == "model":
                anthropic_messages.append({"role": "assistant", "content": message.as_str()})
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        
        anthropic_kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "stream": True,
        }
        
        if system_message is not None:
            anthropic_kwargs["system"] = system_message
        
        anthropic_iterator = await self.client.messages.create(**anthropic_kwargs)
        return AnthropicStreamIteratorAsync(anthropic_iterator)
