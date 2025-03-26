from typing import Dict, Any, Optional, List, Awaitable, AsyncIterator
import hashlib
import json
import re
import os
import aisuite_async
import logging
from promptbuilder.llm_client.messages import Response, MessagesDict, Content, Candidate, UsageMetadata, Part

logger = logging.getLogger(__name__)

class BaseLLMClientAsync:
    @property
    def model(self) -> str:
        """Return the model identifier used by this LLM client."""
        raise NotImplementedError    

    async def from_text(self, prompt: str, **kwargs) -> str:
        return await self.create_text(
            messages=[Content(parts=[Part(text=prompt)], role='user')],
            **kwargs
        )

    async def from_text_structured(self, prompt: str, **kwargs) -> dict | list:
        response = await self.from_text(prompt, **kwargs)
        try:
            return self._as_json(response)
        except ValueError as e:
            raise ValueError(f"Failed to parse LLM response as JSON:\n{response}\nPrompt:\n{prompt}")
    
    def _as_json(self, text: str) -> dict | list:
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

    async def with_system_message(self, system_message: str, input: str, **kwargs) -> str:
        return await self.create_text(
            messages=[
                Content(parts=[Part(text=input)], role='user'),
            ],
            system_message=system_message,
            **kwargs
        )

    async def create(self, messages: List[Content], **kwargs) -> Response:
        raise NotImplementedError

    async def create_text(self, messages: List[Content], **kwargs) -> str:
        response = await self.create(messages, **kwargs)
        return response.candidates[0].content.parts[0].text

    async def create_structured(self, messages: List[Content], **kwargs) -> list | dict:
        content = await self.create_text(messages, **kwargs)
        try:
            return self._as_json(content)
        except ValueError as e:
            raise ValueError(f"Failed to parse LLM response as JSON:\n{content}\nMessages:\n{messages}")

    async def create_stream(self, messages: List[Content], **kwargs) -> Awaitable[AsyncIterator[Response]]:
        raise NotImplementedError

class AiSuiteLLMClientAsync(BaseLLMClientAsync):
    def __init__(self, model: str = None, api_key: str = None, timeout: int = 60):
        if model is None:
            model = os.getenv('DEFAULT_MODEL')
        self._model = model
        provider = model.split(':')[0]
        provider_configs = { provider: {} }
        if api_key is not None:
            provider_configs[provider]['api_key'] = api_key
        if timeout is not None:
            provider_configs[provider]['timeout'] = timeout
        self.client = aisuite_async.AsyncClient(provider_configs=provider_configs)
    
    @property
    def model(self) -> str:
        return self._model

    async def create(self, messages: List[Content], **kwargs) -> Response:
        messages = [{ 'role': message.role, 'content': message.parts[0].text } for message in messages]

        system_message = kwargs.get('system_message', None)
        if system_message is not None:
            messages.insert(0, { 'role': 'system', 'content': system_message })
            del kwargs['system_message']

        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return Response(
            candidates=[
                Candidate(
                    content=Content(
                        parts=[Part(text=choice.message.content)],
                        role=choice.message.role if hasattr(choice.message, 'role') else None
                    )
                )
                for choice in completion.choices
            ],
            usage_metadata=UsageMetadata(
                candidates_token_count=completion.usage.completion_tokens,
                prompt_token_count=completion.usage.prompt_tokens,
                total_token_count=completion.usage.total_tokens
            ) if hasattr(completion, 'usage') and completion.usage is not None else None
        )

LLMClientAsync = AiSuiteLLMClientAsync