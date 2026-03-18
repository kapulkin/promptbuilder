import os
from functools import wraps
from typing import AsyncIterator, Iterator, Callable, ParamSpec, Awaitable

from pydantic import BaseModel
from tenacity import RetryError
from google.genai import Client, types

from promptbuilder.llm_client.base_client import BaseLLMClient, BaseLLMClientAsync, ResultType
from promptbuilder.llm_client.types import Response, Content, Part, ThinkingConfig, Tool, ToolConfig, Model, FunctionCall, FunctionResponse, Blob, FunctionDeclaration, Schema, FunctionCallingConfig, PartLike
from promptbuilder.llm_client.config import DecoratorConfigs
from promptbuilder.llm_client.utils import inherited_decorator
from promptbuilder.llm_client.exceptions import APIError


P = ParamSpec("P")


def _convert_blob_to_genai(blob: Blob | None) -> types.Blob | None:
    """Convert custom Blob to google.genai.types.Blob"""
    if blob is None:
        return None
    return types.Blob.model_construct(**blob.__dict__)


def _convert_function_call_to_genai(fc: FunctionCall | None) -> types.FunctionCall | None:
    """Convert custom FunctionCall to google.genai.types.FunctionCall"""
    if fc is None:
        return None
    return types.FunctionCall.model_construct(**fc.__dict__)


def _convert_function_response_to_genai(fr: FunctionResponse | None) -> types.FunctionResponse | None:
    """Convert custom FunctionResponse to google.genai.types.FunctionResponse"""
    if fr is None:
        return None
    return types.FunctionResponse.model_construct(**fr.__dict__)


def _convert_part_to_genai(part: Part | PartLike) -> types.Part:
    """Convert custom Part or PartLike to google.genai.types.Part"""
    # Handle inline_data conversion
    inline_data = None
    if part.inline_data is not None:
        if isinstance(part.inline_data, Blob):
            inline_data = _convert_blob_to_genai(part.inline_data)
        else:
            # It's already a types.Blob or compatible object
            inline_data = types.Blob.model_construct(**part.inline_data.__dict__)
    
    # Handle function_call conversion
    function_call = None
    if part.function_call is not None:
        if isinstance(part.function_call, FunctionCall):
            function_call = _convert_function_call_to_genai(part.function_call)
        else:
            # It's already a compatible type
            function_call = types.FunctionCall.model_construct(**part.function_call.__dict__)
    
    # Handle function_response conversion
    function_response = None
    if part.function_response is not None:
        if isinstance(part.function_response, FunctionResponse):
            function_response = _convert_function_response_to_genai(part.function_response)
        else:
            # It's already a compatible type
            function_response = types.FunctionResponse.model_construct(**part.function_response.__dict__)
    
    return types.Part.model_construct(
        text=part.text,
        function_call=function_call,
        function_response=function_response,
        thought=part.thought,
        thought_signature=part.thought_signature,
        inline_data=inline_data,
    )


def _convert_content_to_genai(content: Content) -> types.Content:
    """Convert custom Content to google.genai.types.Content"""
    genai_parts: list[types.Part] | None = None
    if content.parts is not None:
        genai_parts = [_convert_part_to_genai(p) for p in content.parts]
    return types.Content.model_construct(
        role=content.role,
        parts=genai_parts,
    )


def _convert_messages_to_genai(messages: list[Content]) -> list[types.Content]:
    """Convert list of custom Content to list of google.genai.types.Content"""
    return [_convert_content_to_genai(msg) for msg in messages]


def _convert_thinking_config_to_genai(thinking_config: ThinkingConfig | None) -> types.ThinkingConfig | None:
    """Convert custom ThinkingConfig to google.genai.types.ThinkingConfig"""
    if thinking_config is None:
        return None
    return types.ThinkingConfig.model_construct(**thinking_config.__dict__)


def _convert_schema_to_genai(schema: Schema | None) -> types.Schema | None:
    """Convert custom Schema to google.genai.types.Schema"""
    if schema is None:
        return None
    return types.Schema.model_construct(
        example=schema.example,
        pattern=schema.pattern,
        minimum=schema.minimum,
        default=schema.default,
        any_of=[_convert_schema_to_genai(s) for s in schema.any_of] if schema.any_of else None,
        max_length=schema.max_length,
        title=schema.title,
        min_length=schema.min_length,
        min_properties=schema.min_properties,
        maximum=schema.maximum,
        max_properties=schema.max_properties,
        description=schema.description,
        enum=schema.enum,
        format=schema.format,
        items=_convert_schema_to_genai(schema.items),
        max_items=schema.max_items,
        min_items=schema.min_items,
        nullable=schema.nullable,
        properties={k: _convert_schema_to_genai(v) for k, v in schema.properties.items()} if schema.properties else None,
        property_ordering=schema.property_ordering,
        required=schema.required,
        type=schema.type,
    )


def _convert_function_declaration_to_genai(fd: FunctionDeclaration) -> types.FunctionDeclaration:
    """Convert custom FunctionDeclaration to google.genai.types.FunctionDeclaration"""
    return types.FunctionDeclaration.model_construct(
        response=_convert_schema_to_genai(fd.response),
        description=fd.description,
        name=fd.name,
        parameters=_convert_schema_to_genai(fd.parameters),
    )


def _convert_tool_to_genai(tool: Tool) -> types.Tool:
    """Convert custom Tool to google.genai.types.Tool"""
    genai_declarations = None
    if tool.function_declarations is not None:
        genai_declarations = [_convert_function_declaration_to_genai(fd) for fd in tool.function_declarations]
    return types.Tool.model_construct(
        function_declarations=genai_declarations,
    )


def _convert_tools_to_genai(tools: list[Tool] | None) -> list[types.Tool] | None:
    """Convert list of custom Tool to list of google.genai.types.Tool"""
    if tools is None:
        return None
    return [_convert_tool_to_genai(t) for t in tools]


def _convert_function_calling_config_to_genai(fcc: FunctionCallingConfig | None) -> types.FunctionCallingConfig | None:
    """Convert custom FunctionCallingConfig to google.genai.types.FunctionCallingConfig"""
    if fcc is None:
        return None
    return types.FunctionCallingConfig(
        mode=types.FunctionCallingConfigMode(fcc.mode.value) if fcc.mode is not None else None,
        allowed_function_names=fcc.allowed_function_names,
    )


def _convert_tool_config_to_genai(tool_config: ToolConfig | None) -> types.ToolConfig | None:
    """Convert custom ToolConfig to google.genai.types.ToolConfig"""
    if tool_config is None:
        return None
    return types.ToolConfig.model_construct(
        function_calling_config=_convert_function_calling_config_to_genai(tool_config.function_calling_config),
    )


@inherited_decorator
def _error_handler(func: Callable[P, Response]) -> Callable[P, Response]:
    """
    Decorator to catch error from google.genai and transform it into unified one
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RetryError as retry_error:
            e = retry_error.last_attempt._exception
            if e is None:
                raise APIError()
            code = e.code
            response_json = {
                "status": e.status,
                "message": e.message,
            }
            response = e.response
            raise APIError(code, response_json, response)
    return wrapper


class GoogleLLMClient(BaseLLMClient):
    PROVIDER: str = "google"
    
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        decorator_configs: DecoratorConfigs | None = None,
        default_thinking_config: ThinkingConfig | None = None,
        default_max_tokens: int | None = None,
        **kwargs,
    ):
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None or not isinstance(api_key, str):
            raise ValueError("To create a google llm client you need to either set the environment variable GOOGLE_API_KEY or pass the api_key in string format")
        super().__init__(GoogleLLMClient.PROVIDER, model, decorator_configs=decorator_configs, default_thinking_config=default_thinking_config, default_max_tokens=default_max_tokens)
        self._api_key = api_key
        self.client = Client(api_key=api_key, **kwargs)
    
    @property
    def api_key(self) -> str:
        return self._api_key
    
    @staticmethod
    def _preprocess_messages(messages: list[Content]) -> list[Content]:
        new_messages = []
        for message in messages:
            new_parts = []
            if message.parts:
                for part in message.parts:
                    if part.inline_data is not None:
                        new_part = Part.model_copy(part, deep=True)
                        new_part.inline_data.display_name = None
                    else:
                        new_part = part
                    new_parts.append(new_part)
            new_message = Content(
                role=message.role,
                parts=new_parts
            )
            new_messages.append(new_message)
        return new_messages
    
    @_error_handler
    def _create(
        self,
        messages: list[Content],
        result_type: ResultType = None,
        *,
        thinking_config: ThinkingConfig | None = None,
        system_message: str | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        tools: list[Tool] | None = None,
        tool_config: ToolConfig = ToolConfig(),
        without_cache: bool = False
    ) -> Response:
        messages = self._preprocess_messages(messages)
        # Convert custom types to google.genai.types
        genai_messages = _convert_messages_to_genai(messages)
        genai_tools = _convert_tools_to_genai(tools)
        genai_tool_config = _convert_tool_config_to_genai(tool_config)
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        config = types.GenerateContentConfig(
            system_instruction=system_message,
            max_output_tokens=max_tokens,
            tools=genai_tools,
            tool_config=genai_tool_config,
        )
        if timeout is not None:
            # Google processes timeout via HttpOptions on the request/config
            config.http_options = types.HttpOptions(timeout=int(timeout * 1_000))
        
        if thinking_config is None:
            thinking_config = self.default_thinking_config
        config.thinking_config = _convert_thinking_config_to_genai(thinking_config)
        
        if result_type is None:
            return self.client.models.generate_content(
                model=self.model,
                contents=genai_messages,
                config=config,
            )
        elif result_type == "json":
            config.response_mime_type = "application/json"
            response = self.client.models.generate_content(
                model=self.model,
                contents=genai_messages,
                config=config,
            )
            response.parsed = BaseLLMClient.as_json(response.text)
            return response
        elif isinstance(result_type, type(BaseModel)):
            config.response_mime_type = "application/json"
            config.response_schema = result_type
            return self.client.models.generate_content(
                model=self.model,
                contents=genai_messages,
                config=config,
            )
        else:
            raise ValueError(f"Unsupported result_type: {result_type}. Supported types are: None, 'json', or a Pydantic model.")
    
    @_error_handler
    def _create_stream(
        self,
        messages: list[Content],
        result_type: ResultType = None,
        *,
        thinking_config: ThinkingConfig | None = None,
        system_message: str | None = None,
        max_tokens: int | None = None,
        without_cache: bool = False
    ) -> Iterator[Response]:
        messages = self._preprocess_messages(messages)
        # Convert custom types to google.genai.types
        genai_messages = _convert_messages_to_genai(messages)
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        config = types.GenerateContentConfig(
            system_instruction=system_message,
            max_output_tokens=max_tokens,
        )
        
        if thinking_config is None:
            thinking_config = self.default_thinking_config
        config.thinking_config = _convert_thinking_config_to_genai(thinking_config)

        if result_type == "json":
            config.response_mime_type = "application/json"
        elif isinstance(result_type, type(BaseModel)):
            config.response_mime_type = "application/json"
            config.response_schema = result_type
        elif result_type is not None:
            raise ValueError(f"Unsupported result_type: {result_type}. Supported types are: None, 'json', or a Pydantic model.")
        
        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=genai_messages,
            config=config,
        )
        return response
    
    @staticmethod
    def models_list() -> list[Model]:
        models: list[Model] = []
        client = Client()
        for google_model in client.models.list():
            for action in google_model.supported_actions:
                if action == "generateContent":
                    model_name = google_model.name
                    if model_name.startswith("models/"):
                        model_name = model_name[7:]
                    
                    if "tts" in model_name.lower():
                        continue
                    if "emb" in model_name.lower():
                        continue
                    if "image-generation" in model_name.lower():
                        continue
                    if "gemini" not in model_name.lower():
                        continue
                    
                    models.append(Model(
                        full_model_name=GoogleLLMClient.PROVIDER + ":" + model_name,
                        model=model_name,
                        provider=GoogleLLMClient.PROVIDER,
                        display_name=google_model.display_name,
                    ))
        return models


@inherited_decorator
def _error_handler_async(func: Callable[P, Awaitable[Response]]) -> Callable[P, Awaitable[Response]]:
    """
    Decorator to catch error from google.genai and transform it into unified one
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except RetryError as retry_error:
            e = retry_error.last_attempt._exception
            if e is None:
                raise APIError()
            code = e.code
            response_json = {
                "status": e.status,
                "message": e.message,
            }
            response = e.response
            raise APIError(code, response_json, response)
    return wrapper

class GoogleLLMClientAsync(BaseLLMClientAsync):
    PROVIDER: str = "google"
    
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        decorator_configs: DecoratorConfigs | None = None,
        default_thinking_config: ThinkingConfig | None = None,
        default_max_tokens: int | None = None,
        **kwargs,
    ):
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        if api_key is None or not isinstance(api_key, str):
            raise ValueError("To create a google llm client you need to either set the environment variable GOOGLE_API_KEY or pass the api_key in string format")
        super().__init__(GoogleLLMClientAsync.PROVIDER, model, decorator_configs=decorator_configs, default_thinking_config=default_thinking_config, default_max_tokens=default_max_tokens)
        self._api_key = api_key
        self.client = Client(api_key=api_key, **kwargs)

    @property
    def api_key(self) -> str:
        return self._api_key

    @_error_handler_async
    async def _create(
        self,
        messages: list[Content],
        result_type: ResultType = None,
        *,
        thinking_config: ThinkingConfig | None = None,
        system_message: str | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        tools: list[Tool] | None = None,
        tool_config: ToolConfig = ToolConfig(),
        without_cache: bool = False
    ) -> Response:
        messages = GoogleLLMClient._preprocess_messages(messages)
        # Convert custom types to google.genai.types
        genai_messages = _convert_messages_to_genai(messages)
        genai_tools = _convert_tools_to_genai(tools)
        genai_tool_config = _convert_tool_config_to_genai(tool_config)
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        config = types.GenerateContentConfig(
            system_instruction=system_message,
            max_output_tokens=max_tokens,
            tools=genai_tools,
            tool_config=genai_tool_config,
        )
        if timeout is not None:
            config.http_options = types.HttpOptions(timeout=int(timeout * 1_000))

        if thinking_config is None:
            thinking_config = self.default_thinking_config
        config.thinking_config = _convert_thinking_config_to_genai(thinking_config)

        if result_type is None:
            return await self.client.aio.models.generate_content(
                model=self.model,
                contents=genai_messages,
                config=config,
            )
        elif result_type == "json":
            config.response_mime_type = "application/json"
            return await self.client.aio.models.generate_content(
                model=self.model,
                contents=genai_messages,
                config=config,
            )
        elif isinstance(result_type, type(BaseModel)):
            config.response_mime_type = "application/json"
            config.response_schema = result_type
            return await self.client.aio.models.generate_content(
                model=self.model,
                contents=genai_messages,
                config=config,
            )
        else:
            raise ValueError(f"Unsupported result_type: {result_type}. Supported types are: None, 'json', or a Pydantic model.")
    
    @_error_handler_async
    async def _create_stream(
        self,
        messages: list[Content],
        result_type: ResultType = None,
        *,
        thinking_config: ThinkingConfig | None = None,
        system_message: str | None = None,
        max_tokens: int | None = None,
        without_cache: bool = False
    ) -> AsyncIterator[Response]:
        messages = GoogleLLMClient._preprocess_messages(messages)
        # Convert custom types to google.genai.types
        genai_messages = _convert_messages_to_genai(messages)
        
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        config = types.GenerateContentConfig(
            system_instruction=system_message,
            max_output_tokens=max_tokens,
        )
        
        if thinking_config is None:
            thinking_config = self.default_thinking_config
        config.thinking_config = _convert_thinking_config_to_genai(thinking_config)

        if result_type == "json":
            config.response_mime_type = "application/json"
        elif isinstance(result_type, type(BaseModel)):
            config.response_mime_type = "application/json"
            config.response_schema = result_type
        elif result_type is not None:
            raise ValueError(f"Unsupported result_type: {result_type}. Supported types are: None, 'json', or a Pydantic model.")
        
        response = await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=genai_messages,
            config=config,
        )
        return response

    @staticmethod
    def models_list() -> list[Model]:
        return GoogleLLMClient.models_list()
