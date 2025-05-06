from promptbuilder.llm_client.base_client import BaseLLMClient, BaseLLMClientAsync
from promptbuilder.llm_client.base_configs import base_decorator_configs
from promptbuilder.llm_client.utils import DecoratorConfigs
from promptbuilder.llm_client.google_client import GoogleLLMClient, GoogleLLMClientAsync
from promptbuilder.llm_client.anthropic_client import AnthropicLLMClient, AnthropicLLMClientAsync
from promptbuilder.llm_client.openai_client import OpenaiLLMClient, OpenaiLLMClientAsync
from promptbuilder.llm_client.aisuite_client import AiSuiteLLMClient, AiSuiteLLMClientAsync


_memory: dict[str, BaseLLMClient] = {}
_memory_async: dict[str, BaseLLMClientAsync] = {}


def get_client(model: str, api_key: str | None = None, decorator_configs: DecoratorConfigs | None = None, default_max_tokens: int | None = None) -> BaseLLMClient:
    global _memory
    
    if model not in _memory:
        if decorator_configs is None:
            decorator_configs = base_decorator_configs[model]
        provider_name, model_name = model.split(":")
        if provider_name == "google":
            if api_key is None:
                client = GoogleLLMClient(model_name, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            else:
                client = GoogleLLMClient(model_name, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            _memory[model] = client
            return _memory[model]
        elif provider_name == "anthropic":
            if api_key is None:
                client = AnthropicLLMClient(model_name, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            else:
                client = AnthropicLLMClient(model_name, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            _memory[model] = client
            return _memory[model]
        elif provider_name == "openai":
            if api_key is None:
                client = OpenaiLLMClient(model_name, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            else:
                client = OpenaiLLMClient(model_name, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            _memory[model] = client
            return _memory[model]
        else:
            if api_key is None:
                raise ValueError(f"You should directly provide api_key for this provider: {provider_name}")
            else:
                client = AiSuiteLLMClient(model, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            _memory[model] = client
            return _memory[model]
    else:
        client = _memory[model]
        if decorator_configs is not None:
            client._decorator_configs = decorator_configs
        if default_max_tokens is not None:
            client.default_max_tokens = default_max_tokens
        return client


def get_async_client(model: str, api_key: str | None = None, decorator_configs: DecoratorConfigs | None = None, default_max_tokens: int | None = None) -> BaseLLMClientAsync:
    global _memory_async
    
    if model not in _memory_async:
        if decorator_configs is None:
            decorator_configs = base_decorator_configs[model]
        provider_name, model_name = model.split(":")
        if provider_name == "google":
            if api_key is None:
                client = GoogleLLMClientAsync(model_name, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            else:
                client = GoogleLLMClientAsync(model_name, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            _memory_async[model] = client
            return _memory_async[model]
        elif provider_name == "anthropic":
            if api_key is None:
                client = AnthropicLLMClientAsync(model_name, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            else:
                client = AnthropicLLMClientAsync(model_name, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            _memory_async[model] = client
            return _memory_async[model]
        elif provider_name == "openai":
            if api_key is None:
                client = OpenaiLLMClientAsync(model_name, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            else:
                client = OpenaiLLMClientAsync(model_name, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            _memory_async[model] = client
            return _memory_async[model]
        else:
            if api_key is None:
                raise ValueError(f"You should directly provide api_key for this provider: {provider_name}")
            else:
                client = AiSuiteLLMClientAsync(model_name, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            _memory_async[model] = client
            return _memory_async[model]
    else:
        client = _memory_async[model]
        if decorator_configs is not None:
            client._decorator_configs = decorator_configs
        if default_max_tokens is not None:
            client.default_max_tokens = default_max_tokens
        return client
