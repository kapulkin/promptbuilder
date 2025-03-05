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

logger = logging.getLogger(__name__)

class Message(BaseModel):
    role: str
    content: str

class Choice(BaseModel):
    message: Message

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Completion(BaseModel):
    choices: List[Choice]
    usage: Optional[Usage] = None

class BaseLLMClient:
    default_max_tokens = 1536

    def from_text(self, prompt: str, **kwargs) -> str:
        return self.create_text(
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            **kwargs
        )

    def from_text_structured(self, prompt: str, **kwargs) -> dict | list:
        response = self.from_text(prompt, **kwargs)
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

    def with_system_message(self, system_message: str, input: str, **kwargs) -> str:
        return self.create_text(
            messages=[
                {'role': 'system', 'content': system_message},
                {'role': 'user', 'content': input}
            ],
            **kwargs
        )

    def create(self, messages: List[Dict[str, str]], **kwargs) -> Completion:
        raise NotImplementedError

    def create_text(self, messages: List[Dict[str, str]], **kwargs) -> str:
        completion = self.create(messages, **kwargs)
        return completion.choices[0].message.content

    def create_structured(self, messages: List[Dict[str, str]], **kwargs) -> list | dict:
        content = self.create_text(messages, **kwargs)
        try:
            return self._as_json(content)
        except ValueError as e:
            raise ValueError(f"Failed to parse LLM response as JSON:\n{content}\nMessages:\n{messages}")

class LLMClient(BaseLLMClient):
    def __init__(self, model: str = None, api_key: str = None, timeout: int = 60):
        if model is None:
            model = os.getenv('DEFAULT_MODEL')
        self.model = model
        provider = model.split(':')[0]
        provider_configs = { provider: {} }
        if api_key is not None:
            provider_configs[provider]['api_key'] = api_key
        if timeout is not None:
            provider_configs[provider]['timeout'] = timeout
        self.client = aisuite.Client(provider_configs=provider_configs)
    
    def create(self, messages: List[Dict[str, str]], **kwargs) -> Completion:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return completion

class CachedLLMClient(BaseLLMClient):
    def __init__(self, llm_client: LLMClient, cache_dir: str = 'data/llm_cache'):
        self.llm_client = llm_client
        self.cache_dir = cache_dir
        self.cache = {}
    
    def _completion_to_dict(self, completion: Completion) -> dict:
        return {
            "choices": [
                {
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content
                    }
                }
                for choice in completion.choices
            ],
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            }
        }

    def create(self, messages: List[Dict[str, str]], **kwargs) -> Completion:
        key = hashlib.sha256(
            json.dumps((self.llm_client.model, messages)).encode()
        ).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rt') as f:
                    cache_data = json.load(f)
                    if cache_data['model'] == self.llm_client.model and json.dumps(cache_data['request']) == json.dumps(messages):
                        return Completion(**cache_data['response'])
                    else:
                        logger.debug(f"Cache mismatch for {key}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Invalid cache file {cache_path}: {str(e)}")
                # Continue to make API call if cache is invalid
        
        completion = self.llm_client.create(messages, **kwargs)
        with open(cache_path, 'wt') as f:
            json.dump({'model': self.llm_client.model, 'request': messages, 'response': self._completion_to_dict(completion)}, f, indent=4)
        return completion
