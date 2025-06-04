import os
from typing import AsyncIterator, Iterator

from pydantic import BaseModel
from google.genai import Client, types

from promptbuilder.llm_client.base_client import BaseLLMClient, BaseLLMClientAsync, ResultType
from promptbuilder.llm_client.types import Response, Content, ThinkingConfig, Tool, ToolConfig
from promptbuilder.llm_client.config import DecoratorConfigs


class GoogleLLMClient(BaseLLMClient):
    PROVIDER: str = "google"
    
    def __init__(
        self,
        model: str,
        api_key: str = os.getenv("GOOGLE_API_KEY"),
        decorator_configs: DecoratorConfigs | None = None,
        default_max_tokens: int | None = None,
        **kwargs,
    ):
        if api_key is None or not isinstance(api_key, str):
            raise ValueError("To create a google llm client you need to either set the environment variable GOOGLE_API_KEY or pass the api_key in string format")
        super().__init__(GoogleLLMClient.PROVIDER, model, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        self._api_key = api_key
        self.client = Client(api_key=api_key)
    
    @property
    def api_key(self) -> str:
        return self._api_key
    
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
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        config = types.GenerateContentConfig(
            system_instruction=system_message,
            max_output_tokens=max_tokens,
            tools=tools,
            tool_config=tool_config,
        )
        
        if not thinking_config.include_thoughts:
            thinking_config = ThinkingConfig(include_thoughts=False, thinking_budget=0)
        if thinking_config.include_thoughts or "gemini-2.5" in self.model:
            config.thinking_config = thinking_config
        
        if result_type is None:
            return self.client.models.generate_content(
                model=self.model,
                contents=messages,
                config=config,
            )
        elif result_type == "json":
            response = self.client.models.generate_content(
                model=self.model,
                contents=messages,
                config=config,
            )
            response.parsed = self._as_json(response.text)
            return response
        elif isinstance(result_type, type(BaseModel)):
            config.response_mime_type = "application/json"
            config.response_schema = result_type
            return self.client.models.generate_content(
                model=self.model,
                contents=messages,
                config=config,
            )
        
    def create_stream(
        self,
        messages: list[Content],
        *,
        system_message: str | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[Response]:
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        config = types.GenerateContentConfig(
            system_instruction=system_message,
            max_output_tokens=max_tokens,
            thinking_config=ThinkingConfig(include_thoughts=False, thinking_budget=0),
        )
        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=messages,
            config=config,
        )
        return response


class GoogleLLMClientAsync(BaseLLMClientAsync):
    PROVIDER: str = "google"
    
    def __init__(
        self,
        model: str,
        api_key: str = os.getenv("GOOGLE_API_KEY"),
        decorator_configs: DecoratorConfigs | None = None,
        default_max_tokens: int | None = None,
        **kwargs,
    ):
        if api_key is None or not isinstance(api_key, str):
            raise ValueError("To create a google llm client you need to either set the environment variable GOOGLE_API_KEY or pass the api_key in string format")
        super().__init__(GoogleLLMClientAsync.PROVIDER, model, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        self._api_key = api_key
        self.client = Client(api_key=api_key)
    
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
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        config = types.GenerateContentConfig(
            system_instruction=system_message,
            max_output_tokens=max_tokens,
            tools=tools,
            tool_config=tool_config,
        )
        
        if not thinking_config.include_thoughts:
            thinking_config = ThinkingConfig(include_thoughts=False, thinking_budget=0)
        if thinking_config.include_thoughts or "gemini-2.5" in self.model:
            config.thinking_config = thinking_config
        
        if result_type is None:
            return await self.client.aio.models.generate_content(
                model=self.model,
                contents=messages,
                config=config,
            )
        elif result_type == "json":
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=messages,
                config=config,
            )
            response.parsed = self._as_json(response.text)
            return response
        elif isinstance(result_type, type(BaseModel)):
            config.response_mime_type = "application/json"
            config.response_schema = result_type
            return await self.client.aio.models.generate_content(
                model=self.model,
                contents=messages,
                config=config,
            )
        
    async def create_stream(
        self,
        messages: list[Content],
        *,
        system_message: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[Response]:
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        config = types.GenerateContentConfig(
            system_instruction=system_message,
            max_output_tokens=max_tokens,
            thinking_config=ThinkingConfig(include_thoughts=False, thinking_budget=0),
        )
        response = await self.client.aio.models.generate_content_stream(
            model=self.model,
            contents=messages,
            config=config,
        )
        return response
