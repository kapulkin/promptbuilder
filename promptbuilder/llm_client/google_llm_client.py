from promptbuilder.llm_client.llm_client import BaseLLMClient
from promptbuilder.llm_client.llm_client_async import BaseLLMClientAsync
from promptbuilder.llm_client.messages import Response, Content
from typing import List
import os
from google.genai import Client

class GoogleLLMClient(BaseLLMClient):
    def __init__(self, model: str = None, api_key: str = os.getenv('GOOGLE_API_KEY')):
        self.client = Client(api_key=api_key)
        self.aio = GoogleLLMClientAsync(client=self.client, model=model)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def create(self, messages: List[Content], **kwargs) -> Response:
        config=types.GenerateContentConfig(**kwargs)        
        return self.client.models.generate_content(
            model=self.model,
            contents=messages,
            config=config
        )
    
    def create_text(self, messages: List[Content], **kwargs) -> Response:
        response = self.create(messages, **kwargs)
        return response.text

class GoogleLLMClientAsync(BaseLLMClientAsync):
    def __init__(self, model: str = None, api_key: str = os.getenv('GOOGLE_API_KEY'), client: Client = None):
        if client is None:
            client = Client(api_key=api_key)
        self.client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def create(self, messages: List[Content], **kwargs) -> Response:
        config=types.GenerateContentConfig(**kwargs)        
        return await self.client.aio.models.generate_content(
            model=self.model,
            contents=messages,
            config=config
        )
    
    async def create_text(self, messages: List[Content], **kwargs) -> Response:
        response = await self.create(messages, **kwargs)
        return response.text

