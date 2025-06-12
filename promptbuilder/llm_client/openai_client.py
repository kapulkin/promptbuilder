import os
import json
import base64
from typing import AsyncIterator, Iterator

from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI, Stream, AsyncStream
from openai.types.responses import ResponseStreamEvent

from promptbuilder.llm_client.base_client import BaseLLMClient, BaseLLMClientAsync, ResultType
from promptbuilder.llm_client.types import Response, Content, Candidate, UsageMetadata, Part, ThinkingConfig, Tool, ToolConfig, FunctionCall, MessageDict, Model
from promptbuilder.llm_client.config import DecoratorConfigs


class OpenaiStreamIterator:
    def __init__(self, openai_iterator: Stream[ResponseStreamEvent]):
        self._openai_iterator = openai_iterator

    def __iter__(self) -> Iterator[Response]:
        output_tokens: int | None = None
        input_tokens: int | None = None
        total_tokens: int | None = None
        for next_event in self._openai_iterator:
            if next_event.type == "response.output_text.delta":
                parts = [Part(text=next_event.delta)]
                yield Response(candidates=[Candidate(content=Content(parts=parts, role="model"))])
            elif next_event.type == "response.completed":
                output_tokens = next_event.response.usage.output_tokens
                input_tokens = next_event.response.usage.input_tokens
                total_tokens = next_event.response.usage.total_tokens
        
        usage_metadata = UsageMetadata(
            candidates_token_count=output_tokens,
            prompt_token_count=input_tokens,
            total_token_count=total_tokens,
        )
        yield Response(candidates=[Candidate(content=Content(parts=[Part(text="")], role="model"))], usage_metadata=usage_metadata)


class OpenaiLLMClient(BaseLLMClient):
    PROVIDER = "openai"

    def __init__(
        self,
        model: str,
        api_key: str = os.getenv("OPENAI_API_KEY"),
        decorator_configs: DecoratorConfigs | None = None,
        default_max_tokens: int | None = None,
        **kwargs,
    ):
        if api_key is None or not isinstance(api_key, str):
            raise ValueError("To create an openai llm client you need to either set the environment variable OPENAI_API_KEY or pass the api_key in string format")
        super().__init__(OpenaiLLMClient.PROVIDER, model, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        self._api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
    @property
    def api_key(self) -> str:
        return self._api_key
    
    @staticmethod
    def _content_to_openai_messages(messages: list[Content], system_message: str | None = None) -> list[MessageDict]:
        openai_messages: list[MessageDict] = []
        if system_message is not None:
            openai_messages.append({"role": "developer", "content": system_message})
        for message in messages:
            role = "user" if message.role == "user" else "assistant"
            if message.parts is None:
                openai_messages.append({"role": role, "content": message.as_str()})
            else:
                content = []
                for part in message.parts:
                    if part.inline_data is not None and part.inline_data.data is not None:
                        base64_file_content = base64.b64encode(part.inline_data.data).decode('utf-8')
                        file_data = {
                            "file_data": f"data:{part.inline_data.mime_type};base64,{base64_file_content}"
                        }
                        if part.inline_data.display_name is not None:
                            file_data["filename"] = part.inline_data.display_name
                        match part.inline_data.mime_type:
                            case "application/pdf":
                                file_data["type"] = "input_file"
                            case "image/png" | "image/jpeg" | "image/webp":
                                file_data["type"] = "input_image"
                            case _:
                                raise ValueError(f"Unsupported inline data mime type: {part.inline_data.mime_type}. Supported types are: application/pdf, image/png, image/jpeg, image/webp.")
                        content.append(file_data)
                    else:
                        text_type = "input_text" if message.role == "user" else "output_text"
                        content.append({"type": text_type, "text": part.as_str()})
                openai_messages.append({"role": role, "content": content})
        return openai_messages

    @staticmethod
    def _process_thinking_config(thinking_config: ThinkingConfig) -> dict[str, str]:
        openai_thinking_config = {}
        if thinking_config.include_thoughts:
            # openai_thinking_config["summary"] = "auto"
            match thinking_config.thinking_budget:
                case 0 | None:
                    openai_thinking_config["reasoning"] = {"effort": "high"} # default
                case 1:
                    openai_thinking_config["reasoning"] = {"effort": "low"}
                case 2:
                    openai_thinking_config["reasoning"] = {"effort": "medium"}
                case 3:
                    openai_thinking_config["reasoning"] = {"effort": "high"}
                case _:
                    openai_thinking_config["reasoning"] = {"effort": "high"}
        return openai_thinking_config

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
        openai_messages: list[MessageDict] = OpenaiLLMClient._content_to_openai_messages(messages, system_message)
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        
        openai_kwargs = {
            "model": self.model,
            "max_output_tokens": max_tokens,
            "input": openai_messages,
        }
        
        openai_kwargs.update(OpenaiLLMClient._process_thinking_config(thinking_config))

        if tools is not None:
            openai_kwargs["parallel_tool_calls"] = True
            
            openai_tools = []
            allowed_function_names = None
            if tool_config.function_calling_config is not None:
                allowed_function_names = tool_config.function_calling_config.allowed_function_names
            for tool in tools:
                for func_decl in tool.function_declarations:
                    if allowed_function_names is None or func_decl.name in allowed_function_names:
                        parameters = func_decl.parameters
                        if parameters is not None:
                            parameters = parameters.model_dump(exclude_none=True)
                            parameters["additionalProperties"] = False
                        else:
                            parameters = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
                        openai_tools.append({
                            "type": "function",
                            "name": func_decl.name,
                            "description": func_decl.description,
                            "strict": True,
                            "parameters": parameters,
                        })
            openai_kwargs["tools"] = openai_tools
            
            tool_choice_mode = "AUTO"
            if tool_config.function_calling_config is not None:
                if tool_config.function_calling_config.mode is not None:
                    tool_choice_mode = tool_config.function_calling_config.mode
            if tool_choice_mode == "NONE":
                openai_kwargs["tool_choice"] = "none"
            elif tool_choice_mode == "AUTO":
                openai_kwargs["tool_choice"] = "auto"
            elif tool_choice_mode == "ANY":
                openai_kwargs["tool_choice"] = "required"
        
        if result_type is None:
            response = self.client.responses.create(**openai_kwargs)
            
            parts: list[Part] = []
            for output_item in response.output:
                if output_item.type == "message":
                    for content in output_item.content:
                        parts.append(Part(text=content.text))
                elif output_item.type == "reasoning":
                    for summary in output_item.summary:
                        parts.append(Part(text=summary.text, thought=True))
                elif output_item.type == "function_call":
                    parts.append(Part(function_call=FunctionCall(args=json.loads(output_item.arguments), name=output_item.name)))
            
            return Response(
                candidates=[Candidate(content=Content(parts=parts, role="model"))],
                usage_metadata=UsageMetadata(
                    candidates_token_count=response.usage.output_tokens,
                    prompt_token_count=response.usage.input_tokens,
                    total_token_count=response.usage.total_tokens,
                ),
            )
        elif result_type == "json":
            response = self.client.responses.create(**openai_kwargs)
            
            parts: list[Part] = []
            text = ""
            for output_item in response.output:
                if output_item.type == "message":
                    for content in output_item.content:
                        parts.append(Part(text=content.text))
                        text += content.text + "\n"
                elif output_item.type == "reasoning":
                    for summary in output_item.summary:
                        parts.append(Part(text=summary.text, thought=True))
                elif output_item.type == "function_call":
                    parts.append(Part(function_call=FunctionCall(args=json.loads(output_item.arguments), name=output_item.name)))
            parsed = BaseLLMClient.as_json(text)
            
            return Response(
                candidates=[Candidate(content=Content(parts=parts, role="model"))],
                usage_metadata=UsageMetadata(
                    candidates_token_count=response.usage.output_tokens,
                    prompt_token_count=response.usage.input_tokens,
                    total_token_count=response.usage.total_tokens,
                ),
                parsed=parsed,
            )
        elif isinstance(result_type, type(BaseModel)):
            response = self.client.responses.parse(**openai_kwargs, text_format=result_type)
            
            parts: list[Part] = []
            for output_item in response.output:
                if output_item.type == "message":
                    for content in output_item.content:
                        parts.append(Part(text=content.text))
                elif output_item.type == "reasoning":
                    for summary in output_item.summary:
                        parts.append(Part(text=summary.text, thought=True))
                elif output_item.type == "function_call":
                    parts.append(Part(function_call=FunctionCall(args=json.loads(output_item.arguments), name=output_item.name)))
            parsed = response.output_parsed
            
            return Response(
                candidates=[Candidate(content=Content(parts=parts, role="model"))],
                usage_metadata=UsageMetadata(
                    candidates_token_count=response.usage.output_tokens,
                    prompt_token_count=response.usage.input_tokens,
                    total_token_count=response.usage.total_tokens,
                ),
                parsed=parsed,
            )
        else:
            raise ValueError(f"Unsupported result type: {result_type}. Supported types are None, 'json', or a Pydantic model class.")
        
    def create_stream(
        self,
        messages: list[Content],
        *,
        system_message: str | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[Response]:
        openai_messages = OpenaiLLMClient._content_to_openai_messages(messages, system_message)
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        
        openai_kwargs = {
            "model": self.model,
            "max_output_tokens": max_tokens,
            "input": openai_messages,
        }
        response = self.client.responses.create(**openai_kwargs, stream=True)
        return OpenaiStreamIterator(response)

    @staticmethod
    def models_list() -> list[Model]:
        models: list[Model] = []
        client = OpenAI()
        for openai_model in client.models.list():
            model_name = openai_model.id
            if "tts" in model_name or "whisper" in model_name:
                continue
            if "emb" in model_name:
                continue
            if "davinci" in model_name or "babbage" in model_name:
                continue
            if "image" in model_name or "dall" in model_name:
                continue
            if "moderation" in model_name:
                continue
            if "2024" in model_name or "2025" in model_name:
                continue
            
            models.append(Model(
                full_model_name=OpenaiLLMClient.PROVIDER + ":" + model_name,
                model=model_name,
                provider=OpenaiLLMClient.PROVIDER,
            ))
        return models


class OpenaiStreamIteratorAsync:
    def __init__(self, openai_iterator: AsyncStream[ResponseStreamEvent]):
        self._openai_iterator = openai_iterator

    async def __aiter__(self) -> AsyncIterator[Response]:
        output_tokens: int | None = None
        input_tokens: int | None = None
        total_tokens: int | None = None
        async for next_event in self._openai_iterator:
            if next_event.type == "response.output_text.delta":
                parts = [Part(text=next_event.delta)]
                yield Response(candidates=[Candidate(content=Content(parts=parts, role="model"))])
            elif next_event.type == "response.completed":
                output_tokens = next_event.response.usage.output_tokens
                input_tokens = next_event.response.usage.input_tokens
                total_tokens = next_event.response.usage.total_tokens
                
        usage_metadata = UsageMetadata(
            candidates_token_count=output_tokens,
            prompt_token_count=input_tokens,
            total_token_count=total_tokens,
        )
        yield Response(candidates=[Candidate(content=Content(parts=[Part(text="")], role="model"))], usage_metadata=usage_metadata)


class OpenaiLLMClientAsync(BaseLLMClientAsync):
    PROVIDER = "openai"
    
    def __init__(
        self,
        model: str,
        api_key: str = os.getenv("OPENAI_API_KEY"),
        decorator_configs: DecoratorConfigs | None = None,
        default_max_tokens: int | None = None,
        **kwargs,
    ):
        if api_key is None or not isinstance(api_key, str):
            raise ValueError("To create an openai llm client you need to either set the environment variable OPENAI_API_KEY or pass the api_key in string format")
        super().__init__(OpenaiLLMClientAsync.PROVIDER, model, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        self._api_key = api_key
        self.client = AsyncOpenAI(api_key=api_key)
    
    @property
    def api_key(self) -> str:
        return self._api_key
    
    async def create(
        self,
        messages: list[Content],
        result_type: ResultType = None,
        thinking_config: ThinkingConfig = ThinkingConfig(),
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        tool_config: ToolConfig = ToolConfig(),
    ) -> Response:
        openai_messages = OpenaiLLMClient._content_to_openai_messages(messages, system_message)
        if system_message is not None:
            openai_messages.append({"role": "developer", "content": system_message})
        for message in messages:
            if message.role == "user":
                openai_messages.append({"role": "user", "content": message.as_str()})
            elif message.role == "model":
                openai_messages.append({"role": "assistant", "content": message.as_str()})
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        
        openai_kwargs = {
            "model": self.model,
            "max_output_tokens": max_tokens,
            "input": openai_messages,
        }
        
        openai_kwargs.update(OpenaiLLMClient._process_thinking_config(thinking_config))
        
        if tools is not None:
            openai_kwargs["parallel_tool_calls"] = True
            
            openai_tools = []
            allowed_function_names = None
            if tool_config.function_calling_config is not None:
                allowed_function_names = tool_config.function_calling_config.allowed_function_names
            for tool in tools:
                for func_decl in tool.function_declarations:
                    if allowed_function_names is None or func_decl.name in allowed_function_names:
                        parameters = func_decl.parameters
                        if parameters is not None:
                            parameters = parameters.model_dump(exclude_none=True)
                            parameters["additionalProperties"] = False
                        else:
                            parameters = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
                        openai_tools.append({
                            "type": "function",
                            "name": func_decl.name,
                            "description": func_decl.description,
                            "strict": True,
                            "parameters": parameters,
                        })
            openai_kwargs["tools"] = openai_tools
            
            tool_choice_mode = "AUTO"
            if tool_config.function_calling_config is not None:
                if tool_config.function_calling_config.mode is not None:
                    tool_choice_mode = tool_config.function_calling_config.mode
            if tool_choice_mode == "NONE":
                openai_kwargs["tool_choice"] = "none"
            elif tool_choice_mode == "AUTO":
                openai_kwargs["tool_choice"] = "auto"
            elif tool_choice_mode == "ANY":
                openai_kwargs["tool_choice"] = "required"
        
        if result_type is None:
            response = await self.client.responses.create(**openai_kwargs)
            
            parts: list[Part] = []
            for output_item in response.output:
                if output_item.type == "message":
                    for content in output_item.content:
                        parts.append(Part(text=content.text))
                elif output_item.type == "reasoning":
                    for summary in output_item.summary:
                        parts.append(Part(text=summary.text, thought=True))
                elif output_item.type == "function_call":
                    parts.append(Part(function_call=FunctionCall(args=json.loads(output_item.arguments), name=output_item.name)))
            
            return Response(
                candidates=[Candidate(content=Content(parts=parts, role="model"))],
                usage_metadata=UsageMetadata(
                    candidates_token_count=response.usage.output_tokens,
                    prompt_token_count=response.usage.input_tokens,
                    total_token_count=response.usage.total_tokens,
                ),
            )
        elif result_type == "json":
            response = await self.client.responses.create(**openai_kwargs)
            
            parts: list[Part] = []
            text = ""
            for output_item in response.output:
                if output_item.type == "message":
                    for content in output_item.content:
                        parts.append(Part(text=content.text))
                        text += content.text + "\n"
                elif output_item.type == "reasoning":
                    for summary in output_item.summary:
                        parts.append(Part(text=summary.text, thought=True))
                elif output_item.type == "function_call":
                    parts.append(Part(function_call=FunctionCall(args=json.loads(output_item.arguments), name=output_item.name)))
            parsed = BaseLLMClient.as_json(text)
            
            return Response(
                candidates=[Candidate(content=Content(parts=parts, role="model"))],
                usage_metadata=UsageMetadata(
                    candidates_token_count=response.usage.output_tokens,
                    prompt_token_count=response.usage.input_tokens,
                    total_token_count=response.usage.total_tokens,
                ),
                parsed=parsed,
            )
        elif isinstance(result_type, type(BaseModel)):
            response = await self.client.responses.parse(**openai_kwargs, text_format=result_type)
            
            parts: list[Part] = []
            for output_item in response.output:
                if output_item.type == "message":
                    for content in output_item.content:
                        parts.append(Part(text=content.text))
                elif output_item.type == "reasoning":
                    for summary in output_item.summary:
                        parts.append(Part(text=summary.text, thought=True))
                elif output_item.type == "function_call":
                    parts.append(Part(function_call=FunctionCall(args=json.loads(output_item.arguments), name=output_item.name)))
            parsed = response.output_parsed
            
            return Response(
                candidates=[Candidate(content=Content(parts=parts, role="model"))],
                usage_metadata=UsageMetadata(
                    candidates_token_count=response.usage.output_tokens,
                    prompt_token_count=response.usage.input_tokens,
                    total_token_count=response.usage.total_tokens,
                ),
                parsed=parsed,
            )
        
    async def create_stream(
        self,
        messages: list[Content],
        *,
        system_message: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[Response]:
        openai_messages = OpenaiLLMClient._content_to_openai_messages(messages, system_message)
        if system_message is not None:
            openai_messages.append({"role": "developer", "content": system_message})
        for message in messages:
            if message.role == "user":
                openai_messages.append({"role": "user", "content": message.as_str()})
            elif message.role == "model":
                openai_messages.append({"role": "assistant", "content": message.as_str()})
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        
        openai_kwargs = {
            "model": self.model,
            "max_output_tokens": max_tokens,
            "input": openai_messages,
        }
        response = await self.client.responses.create(**openai_kwargs, stream=True)
        return OpenaiStreamIteratorAsync(response)

    @staticmethod
    def models_list() -> list[Model]:
        return OpenaiLLMClient.models_list()
