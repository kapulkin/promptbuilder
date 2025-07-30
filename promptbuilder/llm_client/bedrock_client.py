import os
from functools import wraps
from typing import AsyncIterator, Iterator, Any, Callable, ParamSpec, Awaitable

import boto3
from boto3.exceptions import Boto3Error
import aioboto3
from pydantic import BaseModel, ConfigDict
from botocore.eventstream import EventStream
from botocore.exceptions import ClientError, BotoCoreError

from promptbuilder.llm_client.base_client import BaseLLMClient, BaseLLMClientAsync, ResultType
from promptbuilder.llm_client.types import Response, Content, Candidate, UsageMetadata, Part, ThinkingConfig, Tool, ToolConfig, FunctionCall, CustomApiKey, Model
from promptbuilder.llm_client.config import DecoratorConfigs
from promptbuilder.prompt_builder import PromptBuilder
from promptbuilder.llm_client.utils import inherited_decorator
from promptbuilder.llm_client.exceptions import APIError


P = ParamSpec("P")

class BedrockApiKey(BaseModel, CustomApiKey):
    model_config = ConfigDict(frozen=True)
    
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str


@inherited_decorator
def _error_handler(func: Callable[P, Response]) -> Callable[P, Response]:
    """
    Decorator to catch error from boto libs and transform it into unified one
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (Boto3Error, BotoCoreError, ClientError) as e:
            code = None
            response = None
            status = None
            response_json = {
                "status": status,
                "message": str(e.args),
            }
            raise APIError(code, response_json, response)
    return wrapper

class BedrockStreamIterator:
    def __init__(self, bedrock_iterator: EventStream):
        self._bedrock_iterator = bedrock_iterator

    def __iter__(self) -> Iterator[Response]:
        output_tokens: int | None = None
        input_tokens: int | None = None
        total_tokens: int | None = None
        for chunk in self._bedrock_iterator:
            if "contentBlockDelta" in chunk:
                parts = [Part(text=chunk["contentBlockDelta"]["delta"]["text"])]
                yield Response(candidates=[Candidate(content=Content(parts=parts, role="model"))])
            elif "metadata" in chunk:
                output_tokens = chunk["metadata"]["usage"]["outputTokens"]
                input_tokens = chunk["metadata"]["usage"]["inputTokens"]
                total_tokens = chunk["metadata"]["usage"]["totalTokens"]
        
        usage_metadata = UsageMetadata(
            candidates_token_count=output_tokens,
            prompt_token_count=input_tokens,
            total_token_count=total_tokens,
        )
        yield Response(candidates=[Candidate(content=Content(parts=[Part(text="")], role="model"))], usage_metadata=usage_metadata)


class BedrockLLMClient(BaseLLMClient):
    PROVIDER = "bedrock"

    def __init__(
        self,
        model: str,
        api_key: BedrockApiKey | None = None,
        decorator_configs: DecoratorConfigs | None = None,
        default_thinking_config: ThinkingConfig | None = None,
        default_max_tokens: int | None = None,
        **kwargs,
    ):
        if api_key is None:
            api_key = BedrockApiKey(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )
        if not isinstance(api_key, BedrockApiKey):
            raise ValueError(
                "To create a bedrock llm client you need to either set the environment variables "
                "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and optional AWS_DEFAULT_REGION or pass the api_key as BedrockApiKey instance"
            )
        super().__init__(BedrockLLMClient.PROVIDER, model, decorator_configs=decorator_configs, default_thinking_config=default_thinking_config, default_max_tokens=default_max_tokens)
        self._api_key = api_key
    
    @property
    def api_key(self) -> BedrockApiKey:
        return self._api_key
    
    @_error_handler
    def create(
        self,
        messages: list[Content],
        result_type: ResultType = None,
        *,
        thinking_config: ThinkingConfig | None = None,
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        tool_config: ToolConfig = ToolConfig(),
    ) -> Response:
        bedrock_kwargs : dict[str, Any] = {
            "modelId": self.model,
        }
        
        if system_message is not None:
            bedrock_kwargs["system"] = [{"text": system_message}]
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        if max_tokens is not None:
            bedrock_kwargs["inferenceConfig"] = {"maxTokens": max_tokens}
            
        bedrock_messages: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "user":
                bedrock_messages.append({"role": "user", "content": [{"text": message.as_str()}]})
            elif message.role == "model":
                bedrock_messages.append({"role": "assistant", "content": [{"text": message.as_str()}]})
        bedrock_kwargs["messages"] = bedrock_messages
        
        # if thinking_config.include_thoughts:
        #     bedrock_kwargs["reasoning"] = {"effort": "medium"}
        
        if tools is not None:            
            bedrock_tool_config = {"tools": []}
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
                        bedrock_tool_config["tools"].append({"toolSpec": {
                            "name": func_decl.name,
                            "description": func_decl.description,
                            "inputSchema": {"json": parameters},
                        }})
            
            tool_choice_mode = "AUTO"
            if tool_config.function_calling_config is not None:
                if tool_config.function_calling_config.mode is not None:
                    tool_choice_mode = tool_config.function_calling_config.mode
            if tool_choice_mode == "NONE":
                pass
            elif tool_choice_mode == "AUTO":
                bedrock_tool_config["toolChoice"] = {"auto": {}}
                bedrock_kwargs["toolConfig"] = bedrock_tool_config
            elif tool_choice_mode == "ANY":
                bedrock_tool_config["toolChoice"] = {"any": {}}
                bedrock_kwargs["toolConfig"] = bedrock_tool_config
        
        bedrock_runtime_client = boto3.client(
            "bedrock-runtime",
            region_name=self._api_key.aws_region,
            aws_access_key_id=self._api_key.aws_access_key_id,
            aws_secret_access_key=self._api_key.aws_secret_access_key,
        )
        
        if result_type is None:
            response = bedrock_runtime_client.converse(**bedrock_kwargs)
            
            parts: list[Part] = []
            for output_item in response["output"]["message"]["content"]:
                if "reasoningContent" in output_item:
                    parts.append(Part(text=output_item["reasoningContent"]["reasoningText"]["text"], thought=True))
                if "text" in output_item:
                    parts.append(Part(text=output_item["text"]))
                if "toolUse" in output_item:
                    parts.append(Part(function_call=FunctionCall(
                        id=output_item["toolUse"]["toolUseId"],
                        args=output_item["toolUse"]["input"],
                        name=output_item["toolUse"]["name"],
                    )))
            
            return Response(
                candidates=[Candidate(content=Content(parts=parts, role="model"))],
                usage_metadata=UsageMetadata(
                    candidates_token_count=response["usage"]["outputTokens"],
                    prompt_token_count=response["usage"]["inputTokens"],
                    total_token_count=response["usage"]["totalTokens"],
                ),
            )
        elif result_type == "json":
            response = bedrock_runtime_client.converse(**bedrock_kwargs)
            
            parts: list[Part] = []
            text = ""
            for output_item in response["output"]["message"]["content"]:
                if "reasoningContent" in output_item:
                    parts.append(Part(text=output_item["reasoningContent"]["reasoningText"]["text"], thought=True))
                if "text" in output_item:
                    parts.append(Part(text=output_item["text"]))
                    text += output_item["text"] + "\n"
                if "toolUse" in output_item:
                    parts.append(Part(function_call=FunctionCall(
                        id=output_item["toolUse"]["toolUseId"],
                        args=output_item["toolUse"]["input"],
                        name=output_item["toolUse"]["name"],
                    )))
            parsed = BaseLLMClient.as_json(text)
            
            return Response(
                candidates=[Candidate(content=Content(parts=parts, role="model"))],
                usage_metadata=UsageMetadata(
                    candidates_token_count=response["usage"]["outputTokens"],
                    prompt_token_count=response["usage"]["inputTokens"],
                    total_token_count=response["usage"]["totalTokens"],
                ),
                parsed=parsed,
            )
        elif isinstance(result_type, type(BaseModel)):
            message_with_structure = PromptBuilder().set_structured_output(result_type).build().render()
            bedrock_kwargs["messages"].append({"role": "user", "content": [{"text": message_with_structure}]})
            response = bedrock_runtime_client.converse(**bedrock_kwargs)
            
            parts: list[Part] = []
            text = ""
            for output_item in response["output"]["message"]["content"]:
                if "reasoningContent" in output_item:
                    parts.append(Part(text=output_item["reasoningContent"]["reasoningText"]["text"], thought=True))
                if "text" in output_item:
                    parts.append(Part(text=output_item["text"]))
                    text += output_item["text"] + "\n"
                if "toolUse" in output_item:
                    parts.append(Part(function_call=FunctionCall(
                        id=output_item["toolUse"]["toolUseId"],
                        args=output_item["toolUse"]["input"],
                        name=output_item["toolUse"]["name"],
                    )))
            parsed = BaseLLMClient.as_json(text)
            parsed_pydantic = result_type.model_construct(**parsed)
            
            return Response(
                candidates=[Candidate(content=Content(parts=parts, role="model"))],
                usage_metadata=UsageMetadata(
                    candidates_token_count=response["usage"]["outputTokens"],
                    prompt_token_count=response["usage"]["inputTokens"],
                    total_token_count=response["usage"]["totalTokens"],
                ),
                parsed=parsed_pydantic,
            )
    
    @_error_handler
    def create_stream(
        self,
        messages: list[Content],
        *,
        thinking_config: ThinkingConfig | None = None,
        system_message: str | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[Response]:
        bedrock_kwargs : dict[str, Any] = {
            "modelId": self.model,
        }
        
        if system_message is not None:
            bedrock_kwargs["system"] = [{"text": system_message}]
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        if max_tokens is not None:
            bedrock_kwargs["inferenceConfig"] = {"maxTokens": max_tokens}
            
        bedrock_messages: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "user":
                bedrock_messages.append({"role": "user", "content": [{"text": message.as_str()}]})
            elif message.role == "model":
                bedrock_messages.append({"role": "assistant", "content": [{"text": message.as_str()}]})
        bedrock_kwargs["messages"] = bedrock_messages
        
        bedrock_runtime_client = boto3.client(
            "bedrock-runtime",
            region_name=self._api_key.aws_region,
            aws_access_key_id=self._api_key.aws_access_key_id,
            aws_secret_access_key=self._api_key.aws_secret_access_key,
        )
        response = bedrock_runtime_client.converse_stream(**bedrock_kwargs)
        return BedrockStreamIterator(response["stream"])
    
    @staticmethod
    def models_list() -> list[Model]:
        models: list[Model] = []
        bedrock_client = boto3.client("bedrock")
        response = bedrock_client.list_inference_profiles(maxResults=256, typeEquals="SYSTEM_DEFINED")
        for bedrock_model in response["inferenceProfileSummaries"]:
            models.append(Model(
                full_model_name=BedrockLLMClient.PROVIDER + ":" + bedrock_model["inferenceProfileArn"],
                provider=BedrockLLMClient.PROVIDER,
                model=bedrock_model["inferenceProfileArn"],
                display_name=bedrock_model["inferenceProfileName"],
            ))
        return models


@inherited_decorator
def _error_handler_async(func: Callable[P, Awaitable[Response]]) -> Callable[P, Awaitable[Response]]:
    """
    Decorator to catch error from boto libs and transform it into unified one
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (Boto3Error, BotoCoreError, ClientError) as e:
            code = None
            response = None
            status = None
            response_json = {
                "status": status,
                "message": str(e.args),
            }
            raise APIError(code, response_json, response)
    return wrapper

class BedrockStreamIteratorAsync:
    def __init__(self, aioboto_session: aioboto3.Session, **bedrock_kwargs):
        self._aioboto_session = aioboto_session
        self._bedrock_kwargs = bedrock_kwargs

    async def __aiter__(self) -> AsyncIterator[Response]:
        async with self._aioboto_session.client("bedrock-runtime") as bedrock_runtime_client_async:
            response = await bedrock_runtime_client_async.converse_stream(**self._bedrock_kwargs)
            
            output_tokens: int | None = None
            input_tokens: int | None = None
            total_tokens: int | None = None
            async for chunk in response["stream"]:
                if "contentBlockDelta" in chunk:
                    parts = [Part(text=chunk["contentBlockDelta"]["delta"]["text"])]
                    yield Response(candidates=[Candidate(content=Content(parts=parts, role="model"))])
                elif "metadata" in chunk:
                    output_tokens = chunk["metadata"]["usage"]["outputTokens"]
                    input_tokens = chunk["metadata"]["usage"]["inputTokens"]
                    total_tokens = chunk["metadata"]["usage"]["totalTokens"]
            
            usage_metadata = UsageMetadata(
                candidates_token_count=output_tokens,
                prompt_token_count=input_tokens,
                total_token_count=total_tokens,
            )
            yield Response(candidates=[Candidate(content=Content(parts=[Part(text="")], role="model"))], usage_metadata=usage_metadata)


class BedrockLLMClientAsync(BaseLLMClientAsync):
    PROVIDER = "bedrock"

    def __init__(
        self,
        model: str,
        api_key: BedrockApiKey | None = None,
        decorator_configs: DecoratorConfigs | None = None,
        default_thinking_config: ThinkingConfig | None = None,
        default_max_tokens: int | None = None,
        **kwargs,
    ):
        if api_key is None:
            api_key = BedrockApiKey(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )
        if not isinstance(api_key, BedrockApiKey):
            raise ValueError(
                "To create a bedrock llm client you need to either set the environment variables "
                "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and optional AWS_DEFAULT_REGION or pass the api_key as BedrockApiKey instance"
            )
        super().__init__(BedrockLLMClient.PROVIDER, model, decorator_configs=decorator_configs, default_thinking_config=default_thinking_config, default_max_tokens=default_max_tokens)
        self._api_key = api_key
        self._aioboto_session = aioboto3.Session(
            aws_access_key_id=api_key.aws_access_key_id,
            aws_secret_access_key=api_key.aws_secret_access_key,
            region_name=api_key.aws_region,
        )
    
    @property
    def api_key(self) -> BedrockApiKey:
        return self._api_key
    
    @_error_handler_async
    async def create(
        self,
        messages: list[Content],
        result_type: ResultType = None,
        *,
        thinking_config: ThinkingConfig | None = None,
        system_message: str | None = None,
        max_tokens: int | None = None,
        tools: list[Tool] | None = None,
        tool_config: ToolConfig = ToolConfig(),
    ) -> Response:
        bedrock_kwargs : dict[str, Any] = {
            "modelId": self.model,
        }
        
        if system_message is not None:
            bedrock_kwargs["system"] = [{"text": system_message}]
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        if max_tokens is not None:
            bedrock_kwargs["inferenceConfig"] = {"maxTokens": max_tokens}
            
        bedrock_messages: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "user":
                bedrock_messages.append({"role": "user", "content": [{"text": message.as_str()}]})
            elif message.role == "model":
                bedrock_messages.append({"role": "assistant", "content": [{"text": message.as_str()}]})
        bedrock_kwargs["messages"] = bedrock_messages
        
        # if thinking_config.include_thoughts:
        #     bedrock_kwargs["reasoning"] = {"effort": "medium"}
        
        if tools is not None:            
            bedrock_tool_config = {"tools": []}
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
                        bedrock_tool_config["tools"].append({"toolSpec": {
                            "name": func_decl.name,
                            "description": func_decl.description,
                            "inputSchema": {"json": parameters},
                        }})
            
            tool_choice_mode = "AUTO"
            if tool_config.function_calling_config is not None:
                if tool_config.function_calling_config.mode is not None:
                    tool_choice_mode = tool_config.function_calling_config.mode
            if tool_choice_mode == "NONE":
                pass
            elif tool_choice_mode == "AUTO":
                bedrock_tool_config["toolChoice"] = {"auto": {}}
                bedrock_kwargs["toolConfig"] = bedrock_tool_config
            elif tool_choice_mode == "ANY":
                bedrock_tool_config["toolChoice"] = {"any": {}}
                bedrock_kwargs["toolConfig"] = bedrock_tool_config
        
        async with self._aioboto_session.client("bedrock-runtime") as bedrock_runtime_client_async:
            if result_type is None:
                response = await bedrock_runtime_client_async.converse(**bedrock_kwargs)
                
                parts: list[Part] = []
                for output_item in response["output"]["message"]["content"]:
                    if "reasoningContent" in output_item:
                        parts.append(Part(text=output_item["reasoningContent"]["reasoningText"]["text"], thought=True))
                    if "text" in output_item:
                        parts.append(Part(text=output_item["text"]))
                    if "toolUse" in output_item:
                        parts.append(Part(function_call=FunctionCall(
                            id=output_item["toolUse"]["toolUseId"],
                            args=output_item["toolUse"]["input"],
                            name=output_item["toolUse"]["name"],
                        )))
                
                return Response(
                    candidates=[Candidate(content=Content(parts=parts, role="model"))],
                    usage_metadata=UsageMetadata(
                        candidates_token_count=response["usage"]["outputTokens"],
                        prompt_token_count=response["usage"]["inputTokens"],
                        total_token_count=response["usage"]["totalTokens"],
                    ),
                )
            elif result_type == "json":
                response = await bedrock_runtime_client_async.converse(**bedrock_kwargs)
                
                parts: list[Part] = []
                text = ""
                for output_item in response["output"]["message"]["content"]:
                    if "reasoningContent" in output_item:
                        parts.append(Part(text=output_item["reasoningContent"]["reasoningText"]["text"], thought=True))
                    if "text" in output_item:
                        parts.append(Part(text=output_item["text"]))
                        text += output_item["text"] + "\n"
                    if "toolUse" in output_item:
                        parts.append(Part(function_call=FunctionCall(
                            id=output_item["toolUse"]["toolUseId"],
                            args=output_item["toolUse"]["input"],
                            name=output_item["toolUse"]["name"],
                        )))
                parsed = BaseLLMClient.as_json(text)
                
                return Response(
                    candidates=[Candidate(content=Content(parts=parts, role="model"))],
                    usage_metadata=UsageMetadata(
                        candidates_token_count=response["usage"]["outputTokens"],
                        prompt_token_count=response["usage"]["inputTokens"],
                        total_token_count=response["usage"]["totalTokens"],
                    ),
                    parsed=parsed,
                )
            elif isinstance(result_type, type(BaseModel)):
                message_with_structure = PromptBuilder().set_structured_output(result_type).build().render()
                bedrock_kwargs["messages"].append({"role": "user", "content": [{"text": message_with_structure}]})
                response = await bedrock_runtime_client_async.converse(**bedrock_kwargs)
                
                parts: list[Part] = []
                text = ""
                for output_item in response["output"]["message"]["content"]:
                    if "reasoningContent" in output_item:
                        parts.append(Part(text=output_item["reasoningContent"]["reasoningText"]["text"], thought=True))
                    if "text" in output_item:
                        parts.append(Part(text=output_item["text"]))
                        text += output_item["text"] + "\n"
                    if "toolUse" in output_item:
                        parts.append(Part(function_call=FunctionCall(
                            id=output_item["toolUse"]["toolUseId"],
                            args=output_item["toolUse"]["input"],
                            name=output_item["toolUse"]["name"],
                        )))
                parsed = BaseLLMClient.as_json(text)
                parsed_pydantic = result_type.model_construct(**parsed)
                
                return Response(
                    candidates=[Candidate(content=Content(parts=parts, role="model"))],
                    usage_metadata=UsageMetadata(
                        candidates_token_count=response["usage"]["outputTokens"],
                        prompt_token_count=response["usage"]["inputTokens"],
                        total_token_count=response["usage"]["totalTokens"],
                    ),
                    parsed=parsed_pydantic,
                )
    
    @_error_handler_async
    async def create_stream(
        self,
        messages: list[Content],
        *,
        thinking_config: ThinkingConfig | None = None,
        system_message: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[Response]:
        bedrock_kwargs : dict[str, Any] = {
            "modelId": self.model,
        }
        
        if system_message is not None:
            bedrock_kwargs["system"] = [{"text": system_message}]
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        if max_tokens is not None:
            bedrock_kwargs["inferenceConfig"] = {"maxTokens": max_tokens}
            
        bedrock_messages: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "user":
                bedrock_messages.append({"role": "user", "content": [{"text": message.as_str()}]})
            elif message.role == "model":
                bedrock_messages.append({"role": "assistant", "content": [{"text": message.as_str()}]})
        bedrock_kwargs["messages"] = bedrock_messages
        
        return BedrockStreamIteratorAsync(self._aioboto_session, **bedrock_kwargs)

    @staticmethod
    def models_list() -> list[Model]:
        return BedrockLLMClient.models_list()
