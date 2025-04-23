import os
from typing import Awaitable, AsyncIterator, Iterator

from pydantic import BaseModel
from google.genai import Client, types

from promptbuilder.llm_client.base_client import BaseLLMClient, BaseLLMClientAsync, ResultType
from promptbuilder.llm_client.messages import Response, Content, PydanticStructure
from promptbuilder.llm_client.base_configs import DecoratorConfigs, base_decorator_configs


class GoogleLLMClient(BaseLLMClient):
    def __init__(self, model: str, api_key: str = os.getenv("GOOGLE_API_KEY"), decorator_configs: DecoratorConfigs | None = None, **kwargs):
        if decorator_configs is None:
            decorator_configs = base_decorator_configs["google:" + model]
        super().__init__(decorator_configs=decorator_configs)
        self.client = Client(api_key=api_key)
        self._model = model
    
    @property
    def model(self) -> str:
        return "google:" + self._model
    
    def generate_response(self, messages: list[Content], result_type: ResultType = None, use_thinking: bool = False, system_message: str | None = None, **kwargs) -> Response:
        config = types.GenerateContentConfig(system_instruction=system_message, **kwargs)
        if use_thinking:
            config.thinking_config=types.ThinkingConfig(include_thoughts=use_thinking)
        
        if result_type is None:
            return self.client.models.generate_content(
                model=self._model,
                contents=messages,
                config=config,
            )
        elif result_type == "json":
            response = self.client.models.generate_content(
                model=self._model,
                contents=messages,
                config=config,
            )
            response.parsed = self._as_json(response.text)
            return response
        elif isinstance(result_type, type(BaseModel)):
            config.response_mime_type = "application/json"
            config.response_schema = result_type
            return self.client.models.generate_content(
                model=self._model,
                contents=messages,
                config=config,
            )
        
    def create_stream(self, messages: list[Content], system_message: str | None = None, **kwargs) -> Iterator[Response]:
        config=types.GenerateContentConfig(system_instruction=system_message, **kwargs)
        response = self.client.models.generate_content_stream(
            model=self._model,
            contents=messages,
            config=config,
        )
        return response


class GoogleLLMClientAsync(BaseLLMClientAsync):
    def __init__(self, model: str, api_key: str = os.getenv("GOOGLE_API_KEY"), decorator_configs: DecoratorConfigs | None = None, **kwargs):
        if decorator_configs is None:
            decorator_configs = base_decorator_configs["google:" + model]
        super().__init__(decorator_configs=decorator_configs)
        self.client = Client(api_key=api_key)
        self._model = model
    
    @property
    def model(self) -> str:
        return "google:" + self._model
    
    async def generate_response(self, messages: list[Content], result_type: ResultType = None, use_thinking: bool = False, system_message: str | None = None, **kwargs) -> Response:
        config = types.GenerateContentConfig(system_instruction=system_message, **kwargs)
        if use_thinking:
            config.thinking_config=types.ThinkingConfig(include_thoughts=use_thinking)
        
        if result_type is None:
            return await self.client.aio.models.generate_content(
                model=self._model,
                contents=messages,
                config=config,
            )
        elif result_type == "json":
            response = await self.client.aio.models.generate_content(
                model=self._model,
                contents=messages,
                config=config,
            )
            response.parsed = self._as_json(response.text)
            return response
        elif isinstance(result_type, type(BaseModel)):
            config.response_mime_type = "application/json"
            config.response_schema = result_type
            return await self.client.aio.models.generate_content(
                model=self._model,
                contents=messages,
                config=config,
            )
        
    async def create_stream(self, messages: list[Content], system_message: str | None = None, **kwargs) -> Awaitable[AsyncIterator[Response]]:
        config=types.GenerateContentConfig(system_instruction=system_message, **kwargs)
        response = await self.client.aio.models.generate_content_stream(
            model=self._model,
            contents=messages,
            config=config,
        )
        return response
