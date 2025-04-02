from typing import Dict, Any, Optional, List, Awaitable, AsyncIterator
import hashlib
import json
import re
import os
import aisuite_async
import logging
from promptbuilder.llm_client.messages import Response, Content, Candidate, UsageMetadata, Part
from promptbuilder.llm_client.llm_client import AiSuiteLLMClient, BaseLLMClient

logger = logging.getLogger(__name__)

class BaseLLMClientAsync:
    @property
    def model(self) -> str:
        """Return the model identifier used by this LLM client."""
        raise NotImplementedError    

    def _internal_role(self, role: str) -> str:
        return "user" if role == BaseLLMClient.user_tag else "assistant"

    def _external_role(self, role: str) -> str:
        return BaseLLMClient.user_tag if role == "user" else BaseLLMClient.assistant_tag

    async def from_text(self, prompt: str, **kwargs) -> str:
        return await self.create_text(
            messages=[Content(parts=[Part(text=prompt)], role=BaseLLMClient.user_tag)],
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
                Content(parts=[Part(text=input)], role=BaseLLMClient.user_tag),
            ],
            system_message=system_message,
            **kwargs
        )

    async def create(self, messages: List[Content], system_message: str = None, **kwargs) -> Response:
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

    async def create(self, messages: List[Content], system_message: str = None, **kwargs) -> Response:
        messages = [{ 'role': self._internal_role(message.role), 'content': message.parts[0].text } for message in messages]

        if system_message is not None:
            messages.insert(0, { 'role': 'system', 'content': system_message })

        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )

        tool_calls = getattr(completion.choices[0].message, "tool_calls", None)
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        tool_call = tool_calls[0] if tool_calls else None

        return Response(
            candidates=[
                Candidate(
                    content=Content(
                        parts=[Part(text=choice.message.content, function_call=AiSuiteLLMClient.make_function_call(tool_call) if tool_call else None)],
                        role=self._external_role(choice.message.role) if hasattr(choice.message, 'role') else None
                    )
                )
                for choice in completion.choices
            ],
            usage_metadata = AiSuiteLLMClientAsync.make_usage_metadata(completion.usage) if hasattr(completion, 'usage') and completion.usage is not None else None
        )

LLMClientAsync = AiSuiteLLMClientAsync