import requests
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from dataclasses import dataclass
import hashlib
import json
import re
import os
import aisuite
import logging
from .llm_client import Completion, BaseLLMClient

logger = logging.getLogger(__name__)

class BaseLLMClientAsync:
    default_max_tokens = BaseLLMClient.default_max_tokens

    async def from_text(self, prompt: str, **kwargs) -> str:
        return await self.create_text(
            messages=[{
                'role': 'user',
                'content': prompt
            }],
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
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': input}
            ],
            **kwargs
        )

    async def create(self, messages: List[Dict[str, str]], **kwargs) -> Completion:
        raise NotImplementedError

    async def create_text(self, messages: List[Dict[str, str]], **kwargs) -> str:
        completion = await self.create(messages, **kwargs)
        return completion.choices[0].message.content

    async def create_structured(self, messages: List[Dict[str, str]], **kwargs) -> list | dict:
        content = await self.create_text(messages, **kwargs)
        try:
            return self._as_json(content)
        except ValueError as e:
            raise ValueError(f"Failed to parse LLM response as JSON:\n{content}\nMessages:\n{messages}")
