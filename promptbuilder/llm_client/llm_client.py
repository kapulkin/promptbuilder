from typing import Dict, Any, Optional, List, Iterator
import hashlib
import json
import re
import os
import aisuite
import logging
from promptbuilder.llm_client.messages import Completion, Response, Content, Part, UsageMetadata, Candidate, FunctionCall

logger = logging.getLogger(__name__)

class BaseLLMClient:
    user_tag: str = 'user'
    assistant_tag: str = 'model'

    @property
    def model(self) -> str:
        """Return the model identifier used by this LLM client."""
        raise NotImplementedError    

    def from_text(self, prompt: str, **kwargs) -> str:
        return self.create_text(
            messages=[Content(parts=[Part(text=prompt)], role=BaseLLMClient.user_tag)],
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
                Content(parts=[Part(text=input)], role=BaseLLMClient.user_tag)
            ],
            system_message = system_message,
            **kwargs
        )

    def create(self, messages: List[Content], system_message: str = None, **kwargs) -> Response:
        raise NotImplementedError

    def create_text(self, messages: List[Content], **kwargs) -> str:
        response = self.create(messages, **kwargs)
        return response.candidates[0].content.parts[0].text

    def create_structured(self, messages: List[Content], **kwargs) -> list | dict:
        content = self.create_text(messages, **kwargs)
        try:
            return self._as_json(content)
        except ValueError as e:
            raise ValueError(f"Failed to parse LLM response as JSON:\n{content}\nMessages:\n{messages}")

    def create_stream(self, messages: List[Content], **kwargs) -> Iterator[Response]:
        raise NotImplementedError

class AiSuiteLLMClient(BaseLLMClient):
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
        self.client = aisuite.Client(provider_configs=provider_configs)
    
    @property
    def model(self) -> str:
        return self._model

    def _internal_role(self, role: str) -> str:
        return "user" if role == self.user_tag else "assistant"

    def _external_role(self, role: str) -> str:
        return self.user_tag if role == "user" else self.assistant_tag

    @staticmethod
    def make_function_call(tool_call) -> FunctionCall | None:
        if isinstance(tool_call, dict):
            tool_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            tool_call_id = tool_call["id"]
        else:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments
            tool_call_id = tool_call.id

        # Ensure arguments is a dict
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        return FunctionCall(id=tool_call_id, name=tool_name, args=arguments)

    def create(self, messages: List[Content], system_message: str = None, **kwargs) -> Response:
        messages = [{ 'role': self._internal_role(message.role), 'content': message.parts[0].text } for message in messages]

        if system_message is not None:
            messages.insert(0, { 'role': 'system', 'content': system_message })

        completion = self.client.chat.completions.create(
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
            usage_metadata=UsageMetadata(
                candidates_token_count=completion.usage.completion_tokens,
                prompt_token_count=completion.usage.prompt_tokens,
                total_token_count=completion.usage.total_tokens
            ) if hasattr(completion, 'usage') and completion.usage is not None else None
        )

LLMClient = AiSuiteLLMClient

class CachedLLMClient(BaseLLMClient):
    def __init__(self, llm_client: BaseLLMClient, cache_dir: str = 'data/llm_cache'):
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

    def create(self, messages: List[Content], **kwargs) -> Response:
        messages_dump = [message.model_dump() for message in messages]
        key = hashlib.sha256(
            json.dumps((self.llm_client.model, messages_dump)).encode()
        ).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rt') as f:
                    cache_data = json.load(f)
                    if cache_data['model'] == self.llm_client.model and json.dumps(cache_data['request']) == json.dumps(messages_dump):
                        return Response(**cache_data['response'])
                    else:
                        logger.debug(f"Cache mismatch for {key}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Invalid cache file {cache_path}: {str(e)}")
                # Continue to make API call if cache is invalid
        
        response = self.llm_client.create(messages, **kwargs)
        with open(cache_path, 'wt') as f:
            json.dump({'model': self.llm_client.model, 'request': messages_dump, 'response': response.model_dump()}, f, indent=4)
        return response
