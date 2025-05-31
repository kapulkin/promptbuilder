import warnings
from itertools import chain

from promptbuilder.llm_client.base_client import BaseLLMClient, BaseLLMClientAsync
from promptbuilder.llm_client.config import GLOBAL_CONFIG
from promptbuilder.llm_client.utils import DecoratorConfigs
from promptbuilder.llm_client.google_client import GoogleLLMClient, GoogleLLMClientAsync
from promptbuilder.llm_client.anthropic_client import AnthropicLLMClient, AnthropicLLMClientAsync
from promptbuilder.llm_client.openai_client import OpenaiLLMClient, OpenaiLLMClientAsync
from promptbuilder.llm_client.aisuite_client import AiSuiteLLMClient, AiSuiteLLMClientAsync


_memory: dict[str, BaseLLMClient] = {}
_memory_async: dict[str, BaseLLMClientAsync] = {}


def get_client(full_model_name: str, api_key: str | None = None, decorator_configs: DecoratorConfigs | None = None, default_max_tokens: int | None = None) -> BaseLLMClient:
    global _memory
    
    if full_model_name not in _memory:
        provider, model = full_model_name.split(":")
        if provider == "google":
            if api_key is None:
                client = GoogleLLMClient(model, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            else:
                client = GoogleLLMClient(model, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        elif provider == "anthropic":
            if api_key is None:
                client = AnthropicLLMClient(model, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            else:
                client = AnthropicLLMClient(model, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        elif provider == "openai":
            if api_key is None:
                client = OpenaiLLMClient(model, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            else:
                client = OpenaiLLMClient(model, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        else:
            if api_key is None:
                raise ValueError(f"You should directly provide api_key for this provider: {provider}")
            else:
                client = AiSuiteLLMClient(full_model_name, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
                
        _memory[full_model_name] = client
        return _memory[full_model_name]
    else:
        client = _memory[full_model_name]
        if decorator_configs is not None:
            client._decorator_configs = decorator_configs
        if default_max_tokens is not None:
            client.default_max_tokens = default_max_tokens
        return client


def get_async_client(full_model_name: str, api_key: str | None = None, decorator_configs: DecoratorConfigs | None = None, default_max_tokens: int | None = None) -> BaseLLMClientAsync:
    global _memory_async
    
    if full_model_name not in _memory_async:
        provider, model = full_model_name.split(":")
        if provider == "google":
            if api_key is None:
                client = GoogleLLMClientAsync(model, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            else:
                client = GoogleLLMClientAsync(model, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        elif provider == "anthropic":
            if api_key is None:
                client = AnthropicLLMClientAsync(model, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            else:
                client = AnthropicLLMClientAsync(model, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        elif provider == "openai":
            if api_key is None:
                client = OpenaiLLMClientAsync(model, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
            else:
                client = OpenaiLLMClientAsync(model, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        else:
            if api_key is None:
                raise ValueError(f"You should directly provide api_key for this provider: {provider}")
            else:
                client = AiSuiteLLMClientAsync(full_model_name, api_key, decorator_configs=decorator_configs, default_max_tokens=default_max_tokens)
        
        _memory_async[full_model_name] = client
        return _memory_async[full_model_name]
    else:
        client = _memory_async[full_model_name]
        if decorator_configs is not None:
            client._decorator_configs = decorator_configs
        if default_max_tokens is not None:
            client.default_max_tokens = default_max_tokens
        return client


def configure(
    *,
    decorator_configs: dict[str, DecoratorConfigs] | None = None,
    update_decorator_configs: dict[str, DecoratorConfigs] | None = None,
    max_tokens: dict[str, int] | None = None,
    update_max_tokens: dict[str, int] | None = None,
):
    if decorator_configs is not None and update_decorator_configs is not None:
        warnings.warn("Both 'decorator_configs' and 'update_decorator_configs' were provided. "
                      "'update_decorator_configs' will be ignored.", UserWarning)
        update_decorator_configs = None
    if max_tokens is not None and update_max_tokens is not None:
        warnings.warn("Both 'max_tokens' and 'update_max_tokens' were provided. "
                      "'update_max_tokens' will be ignored.", UserWarning)
        update_max_tokens = None
    
    if decorator_configs is not None:
        GLOBAL_CONFIG.default_decorator_configs = decorator_configs
    if update_decorator_configs is not None:
        GLOBAL_CONFIG.default_decorator_configs.update(update_decorator_configs)
    
    if max_tokens is not None:
        GLOBAL_CONFIG.default_max_tokens = max_tokens
    if update_max_tokens is not None:
        GLOBAL_CONFIG.default_max_tokens.update(update_max_tokens)

def sync_existing_clients_with_global_config():
    for full_model_name, llm_client in chain(_memory.items(), _memory_async.items()):
        if full_model_name in GLOBAL_CONFIG.default_decorator_configs:
            llm_client._decorator_configs = GLOBAL_CONFIG.default_decorator_configs[full_model_name]
        else:
            llm_client._decorator_configs = DecoratorConfigs()
        
        if full_model_name in GLOBAL_CONFIG.default_max_tokens:
            llm_client.default_max_tokens = GLOBAL_CONFIG.default_max_tokens[full_model_name]
        else:
            llm_client.default_max_tokens = None
