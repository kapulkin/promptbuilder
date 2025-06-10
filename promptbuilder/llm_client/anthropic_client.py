import os
from typing import AsyncIterator, Iterator
import base64
from pydantic import BaseModel
from anthropic import Anthropic, AsyncAnthropic, Stream, AsyncStream
from anthropic.types import RawMessageStreamEvent

from promptbuilder.llm_client.base_client import BaseLLMClient, BaseLLMClientAsync, ResultType
from promptbuilder.llm_client.types import Response, Content, Candidate, UsageMetadata, Part, ThinkingConfig, Tool, ToolConfig, FunctionCall, MessageDict
from promptbuilder.llm_client.config import DecoratorConfigs
from promptbuilder.prompt_builder import PromptBuilder


def sum_optional_ints(a: int | None, b: int | None) -> int | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a + b


class DefaultMaxTokensStrategy:
    def for_create(self, model: str) -> int:
        raise NotImplementedError
    
    def for_create_stream(self, model: str) -> int:
        raise NotImplementedError

# The Anthropic API requires an explicit integer value for the 'max_tokens' parameter. 
# Unlike other APIs where 'None' might imply using the model's maximum, Anthropic's API does not permit this.
# Furthermore, the official Anthropic Python library itself has different
# internal default token limits depending on whether the request is for a streaming or non-streaming response.
class AnthropicDefaultMaxTokensStrategy(DefaultMaxTokensStrategy):
    def for_create(self, model: str) -> int:
        if "claude-3-haiku" in model:
            return 4096
        elif "claude-3-opus" in model:
            return 4096
        elif "claude-3-5-haiku" in model:
            return 8192
        elif "claude-3-5-sonnet" in model:
            return 8192
        elif "claude-3-7-sonnet" in model:
            return 8192
        elif "claude-sonnet-4" in model:
            return 8192
        elif "claude-opus-4" in model:
            return 8192
        else:
            return 8192
    
    def for_create_stream(self, model: str) -> int:
        if "claude-3-haiku" in model:
            return 4096
        elif "claude-3-opus" in model:
            return 4096
        elif "claude-3-5-haiku" in model:
            return 8192
        elif "claude-3-5-sonnet" in model:
            return 8192
        elif "claude-3-7-sonnet" in model:
            return 64000
        elif "claude-sonnet-4" in model:
            return 64000
        elif "claude-opus-4" in model:
            return 32000
        else:
            return 32000


class AnthropicStreamIterator:
    def __init__(self, anthropic_iterator: Stream[RawMessageStreamEvent]):
        self._anthropic_iterator = anthropic_iterator

    def __iter__(self) -> Iterator[Response]:
        input_tokens: int | None = None
        output_tokens: int | None = None
        total_tokens : int | None = None
        for next_event in self._anthropic_iterator:
            if next_event.type == "message_start":
                input_tokens = sum_optional_ints(input_tokens, next_event.message.usage.input_tokens)
                output_tokens = sum_optional_ints(output_tokens, next_event.message.usage.output_tokens)
            elif next_event.type == "content_block_delta":
                if next_event.delta.type == "thinking_delta":
                    parts = [Part(text=next_event.delta.thinking, thought=True)]
                    yield Response(candidates=[Candidate(content=Content(parts=parts, role="model"))])
                elif next_event.delta.type == "text_delta":
                    parts = [Part(text=next_event.delta.text)]
                    yield Response(candidates=[Candidate(content=Content(parts=parts, role="model"))])
            elif next_event.type == "message_delta":
                input_tokens = sum_optional_ints(input_tokens, next_event.usage.input_tokens)
                output_tokens = sum_optional_ints(output_tokens, next_event.usage.output_tokens)
        
        if input_tokens is not None or output_tokens is not None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        usage_metadata = UsageMetadata(
            candidates_token_count=output_tokens,
            prompt_token_count=input_tokens,
            total_token_count=total_tokens,
        )
        yield Response(candidates=[Candidate(content=Content(parts=[Part(text="")], role="model"))], usage_metadata=usage_metadata)


class AnthropicLLMClient(BaseLLMClient):
    PROVIDER: str = "anthropic"
    
    def __init__(
        self,
        model: str,
        api_key: str = os.getenv("ANTHROPIC_API_KEY"),
        decorator_configs: DecoratorConfigs | None = None,
        default_max_tokens: int | None = None,
        default_max_tokens_strategy: DefaultMaxTokensStrategy = AnthropicDefaultMaxTokensStrategy(),
        **kwargs,
    ):
        if api_key is None or not isinstance(api_key, str):
            raise ValueError("To create an anthropic llm client you need to either set the environment variable ANTHROPIC_API_KEY or pass the api_key in string format")
        super().__init__(AnthropicLLMClient.PROVIDER, model, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        self._api_key = api_key
        self.client = Anthropic(api_key=api_key)
        self.default_max_tokens_strategy = default_max_tokens_strategy
    
    @property
    def api_key(self) -> str:
        return self._api_key
    
    def content_to_anthropic_messages(self, messages: list[Content]) -> list[MessageDict]:
        anthropic_messages: list[MessageDict] = []
        for message in messages:
            role = "user" if message.role == "user" else "assistant"
            if message.parts is None:
                anthropic_messages.append({"role": role, "content": message.as_str()})
            else:
                content = []
                for part in message.parts:
                    if part.inline_data is not None and part.inline_data.data is not None:                        
                        match part.inline_data.mime_type:
                            case "application/pdf":
                                data_type = "document"
                            case "image/png" | "image/jpeg" | "image/webp":
                                data_type = "image"
                            case _:
                                raise ValueError(f"Unsupported data mime type: {part.inline_data.mime_type}")

                        base64_file_content = base64.b64encode(part.inline_data.data).decode('utf-8')
                        content.append({
                            "type": data_type,
                            "source": {
                                "type": "base64",
                                "media_type": part.inline_data.mime_type,
                                "data": base64_file_content,
                            }
                        })
                    else:
                        content.append({
                            "type": "text",
                            "text": part.as_str()
                        })
                anthropic_messages.append({"role": role, "content": content})
        return anthropic_messages

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
        anthropic_messages = self.content_to_anthropic_messages(messages)
        if max_tokens is None:
            if self.default_max_tokens is None:
                max_tokens = self.default_max_tokens_strategy.for_create(self.model)
            else:
                max_tokens = self.default_max_tokens
        
        anthropic_kwargs = {
            "model": self.model,
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
        
        if tools is not None:
            anthropic_tools = []
            allowed_function_names = None
            if tool_config.function_calling_config is not None:
                allowed_function_names = tool_config.function_calling_config.allowed_function_names
            for tool in tools:
                for func_decl in tool.function_declarations:
                    if allowed_function_names is None or func_decl.name in allowed_function_names:
                        schema = func_decl.parameters
                        if schema is not None:
                            schema = schema.model_dump(exclude_none=True)
                        else:
                            schema = {"type": "object", "properties": {}}
                        anthropic_tools.append({
                            "name": func_decl.name,
                            "description": func_decl.description,
                            "input_schema": schema,
                        })
            anthropic_kwargs["tools"] = anthropic_tools
            
            tool_choice_mode = "AUTO"
            if tool_config.function_calling_config is not None:
                if tool_config.function_calling_config.mode is not None:
                    tool_choice_mode = tool_config.function_calling_config.mode
            anthropic_kwargs["tool_choice"] = {"type": tool_choice_mode.lower()}
        
        if result_type is None:
            response = self.client.messages.create(**anthropic_kwargs)
            
            parts: list[Part] = []
            for content in response.content:
                if content.type == "thinking":
                    parts.append(Part(text=content.thinking, thought=True))
                elif content.type == "text":
                    parts.append(Part(text=content.text))
                elif content.type == "tool_use":
                    parts.append(Part(function_call=FunctionCall(args=content.input, name=content.name)))
            
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
                elif content.type == "tool_use":
                    parts.append(Part(function_call=FunctionCall(args=content.input, name=content.name)))
            parsed = BaseLLMClient.as_json(text)
            
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
            message_with_structure = PromptBuilder().set_structured_output(result_type).build().render()
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
                elif content.type == "tool_use":
                    parts.append(Part(function_call=FunctionCall(args=content.input, name=content.name)))
            parsed = BaseLLMClient.as_json(text)
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
        else:
            raise ValueError(f"Unsupported result type: {result_type}")
    
    def create_stream(
        self,
        messages: list[Content],
        *,
        system_message: str | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[Response]:
        anthropic_messages = self.content_to_anthropic_messages(messages)
        
        if max_tokens is None:
            if self.default_max_tokens is None:
                max_tokens = self.default_max_tokens_strategy.for_create_stream(self.model)
            else:
                max_tokens = self.default_max_tokens
        
        anthropic_kwargs = {
            "model": self.model,
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

    async def __aiter__(self) -> AsyncIterator[Response]:
        input_tokens: int | None = None
        output_tokens: int | None = None
        total_tokens : int | None = None
        async for next_event in self._anthropic_iterator:
            if next_event.type == "message_start":
                input_tokens = sum_optional_ints(input_tokens, next_event.message.usage.input_tokens)
                output_tokens = sum_optional_ints(output_tokens, next_event.message.usage.output_tokens)
            elif next_event.type == "content_block_delta":
                if next_event.delta.type == "thinking_delta":
                    parts = [Part(text=next_event.delta.thinking, thought=True)]
                    yield Response(candidates=[Candidate(content=Content(parts=parts, role="model"))])
                elif next_event.delta.type == "text_delta":
                    parts = [Part(text=next_event.delta.text)]
                    yield Response(candidates=[Candidate(content=Content(parts=parts, role="model"))])
            elif next_event.type == "message_delta":
                input_tokens = sum_optional_ints(input_tokens, next_event.usage.input_tokens)
                output_tokens = sum_optional_ints(output_tokens, next_event.usage.output_tokens)
        
        if input_tokens is not None or output_tokens is not None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        usage_metadata = UsageMetadata(
            candidates_token_count=output_tokens,
            prompt_token_count=input_tokens,
            total_token_count=total_tokens,
        )
        yield Response(candidates=[Candidate(content=Content(parts=[Part(text="")], role="model"))], usage_metadata=usage_metadata)


class AnthropicLLMClientAsync(BaseLLMClientAsync):
    PROVIDER: str = "anthropic"
    
    def __init__(
        self,
        model: str,
        api_key: str = os.getenv("ANTHROPIC_API_KEY"),
        decorator_configs: DecoratorConfigs | None = None,
        default_max_tokens: int | None = None,
        default_max_tokens_strategy: DefaultMaxTokensStrategy = AnthropicDefaultMaxTokensStrategy(),
        **kwargs,
    ):
        if api_key is None or not isinstance(api_key, str):
            raise ValueError("To create an anthropic llm client you need to either set the environment variable ANTHROPIC_API_KEY or pass the api_key in string format")
        super().__init__(AnthropicLLMClientAsync.PROVIDER, model, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        self._api_key = api_key
        self.client = AsyncAnthropic(api_key=api_key)
        self.default_max_tokens_strategy = default_max_tokens_strategy
    
    @property
    def api_key(self) -> str:
        return self._api_key
    
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
        anthropic_messages: list[dict[str, str]] = []
        for message in messages:
            if message.role == "user":
                anthropic_messages.append({"role": "user", "content": message.as_str()})
            elif message.role == "model":
                anthropic_messages.append({"role": "assistant", "content": message.as_str()})
        
        if max_tokens is None:
            if self.default_max_tokens is None:
                max_tokens = self.default_max_tokens_strategy.for_create(self.model)
            else:
                max_tokens = self.default_max_tokens
        
        anthropic_kwargs = {
            "model": self.model,
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
        
        if tools is not None:
            anthropic_tools = []
            allowed_function_names = None
            if tool_config.function_calling_config is not None:
                allowed_function_names = tool_config.function_calling_config.allowed_function_names
            for tool in tools:
                for func_decl in tool.function_declarations:
                    if allowed_function_names is None or func_decl.name in allowed_function_names:
                        schema = func_decl.parameters
                        if schema is not None:
                            schema = schema.model_dump(exclude_none=True)
                        else:
                            schema = {"type": "object", "properties": {}}
                        anthropic_tools.append({
                            "name": func_decl.name,
                            "description": func_decl.description,
                            "input_schema": schema,
                        })
            anthropic_kwargs["tools"] = anthropic_tools
            
            tool_choice_mode = "AUTO"
            if tool_config.function_calling_config is not None:
                if tool_config.function_calling_config.mode is not None:
                    tool_choice_mode = tool_config.function_calling_config.mode
            anthropic_kwargs["tool_choice"] = {"type": tool_choice_mode.lower()}
        
        if result_type is None:
            response = await self.client.messages.create(**anthropic_kwargs)
            
            parts: list[Part] = []
            for content in response.content:
                if content.type == "thinking":
                    parts.append(Part(text=content.thinking, thought=True))
                elif content.type == "text":
                    parts.append(Part(text=content.text))
                elif content.type == "tool_use":
                    parts.append(Part(function_call=FunctionCall(args=content.input, name=content.name)))
            
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
                elif content.type == "tool_use":
                    parts.append(Part(function_call=FunctionCall(args=content.input, name=content.name)))
            parsed = BaseLLMClient.as_json(text)
            
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
            message_with_structure = PromptBuilder().set_structured_output(result_type).build().render()
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
                elif content.type == "tool_use":
                    parts.append(Part(function_call=FunctionCall(args=content.input, name=content.name)))
            parsed = BaseLLMClient.as_json(text)
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
        *,
        system_message: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[Response]:
        anthropic_messages: list[dict[str, str]] = []
        for message in messages:
            if message.role == "user":
                anthropic_messages.append({"role": "user", "content": message.as_str()})
            elif message.role == "model":
                anthropic_messages.append({"role": "assistant", "content": message.as_str()})
        
        if max_tokens is None:
            if self.default_max_tokens is None:
                max_tokens = self.default_max_tokens_strategy.for_create_stream(self.model)
            else:
                max_tokens = self.default_max_tokens
        
        anthropic_kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
            "stream": True,
        }
        
        if system_message is not None:
            anthropic_kwargs["system"] = system_message
        
        anthropic_iterator = await self.client.messages.create(**anthropic_kwargs)
        return AnthropicStreamIteratorAsync(anthropic_iterator)
