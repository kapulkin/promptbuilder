import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
import json
import re
import os
import aisuite
import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv('DEFAULT_MODEL')

class BaseLLMClient:
    default_max_tokens = 1536

    def make_request(self, prompt: str, temperature: float = 0.0, max_tokens: int = default_max_tokens, **kwargs) -> str:
        return self.create(
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

    def make_json_request(self, prompt: str, temperature: float = 0.0, max_tokens: int = default_max_tokens, **kwargs) -> dict | list:
        response = self.make_request(prompt, temperature, max_tokens, **kwargs)
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

    def with_system_message(self, system_message: str, input: str, temperature: float = 0.0, max_tokens: int = default_max_tokens, **kwargs) -> str:
        return self.create(
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': input}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

    def create_structured(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = default_max_tokens, **kwargs) -> dict | list:
        response = self.create(messages, temperature, max_tokens, **kwargs)
        try:
            return self._as_json(response)
        except ValueError as e:
            raise ValueError(f"Failed to parse LLM response as JSON:\n{response}\nMessages:\n{messages}")

    def create(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = default_max_tokens, **kwargs) -> str:
        raise NotImplementedError

class LLMClient(BaseLLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str = None, timeout: int = 60):
        self.model = model
        provider = model.split(':')[0]
        provider_configs = { provider: {} }
        if api_key is not None:
            provider_configs[provider]['api_key'] = api_key
        if timeout is not None:
            provider_configs[provider]['timeout'] = timeout
        self.client = aisuite.Client(provider_configs=provider_configs)
    
    def create(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = BaseLLMClient.default_max_tokens, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response.choices[0].message.content

class CachedLLMClient(BaseLLMClient):
    def __init__(self, llm_client: LLMClient, cache_dir: str = 'data/llm_cache'):
        self.llm_client = llm_client
        self.cache_dir = cache_dir
        self.cache = {}
    
    def create(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = BaseLLMClient.default_max_tokens, **kwargs) -> str:
        key = hashlib.sha256(
            json.dumps((self.llm_client.model, messages)).encode()
        ).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'rt') as f:
                cache_data = json.load(f)
                if cache_data['model'] == self.llm_client.model and json.dumps(cache_data['request']) == json.dumps(messages):
                    return cache_data['response']
                else:
                    logger.debug(f"Cache mismatch for {key}")
        response = self.llm_client.create(messages, temperature, max_tokens, **kwargs)
        with open(cache_path, 'wt') as f:
            json.dump({'model': self.llm_client.model, 'request': messages, 'response': response}, f, indent=4)
        return response
