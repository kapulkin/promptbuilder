import warnings
from itertools import chain

from promptbuilder.llm_client.types import ApiKey, Model, ThinkingConfig
from promptbuilder.llm_client.base_client import BaseLLMClient, BaseLLMClientAsync
from promptbuilder.llm_client.config import GLOBAL_CONFIG
from promptbuilder.llm_client.utils import DecoratorConfigs
from promptbuilder.llm_client.google_client import GoogleLLMClient, GoogleLLMClientAsync
from promptbuilder.llm_client.anthropic_client import AnthropicLLMClient, AnthropicLLMClientAsync
from promptbuilder.llm_client.openai_client import OpenaiLLMClient, OpenaiLLMClientAsync
from promptbuilder.llm_client.bedrock_client import BedrockLLMClient, BedrockLLMClientAsync
from promptbuilder.llm_client.aisuite_client import AiSuiteLLMClient, AiSuiteLLMClientAsync



_memory: dict[tuple[str, ApiKey], BaseLLMClient] = {}
_memory_async: dict[tuple[str, ApiKey], BaseLLMClientAsync] = {}


def get_client(
    full_model_name: str,
    api_key: ApiKey | None = None,
    decorator_configs: DecoratorConfigs | None = None,
    default_thinking_config: ThinkingConfig | None = None,
    default_max_tokens: int | None = None,
) -> BaseLLMClient:
    global _memory
    
    kwargs = {
        "decorator_configs": decorator_configs,
        "default_thinking_config": default_thinking_config,
        "default_max_tokens": default_max_tokens,
    }
    provider_to_client_class: dict[str, type[BaseLLMClient]] = {
        "google": GoogleLLMClient,
        "anthropic": AnthropicLLMClient,
        "openai": OpenaiLLMClient,
        "bedrock": BedrockLLMClient,
    }
    provider, model = full_model_name.split(":", 1)
    if provider in provider_to_client_class:
        client_class = provider_to_client_class[provider]
        if api_key is None:
            client = client_class(model, **kwargs)
        else:
            client = client_class(model, api_key, **kwargs)
    else:
        if api_key is None:
            raise ValueError(f"You should directly provide api_key for this provider: {provider}")
        else:
            client = AiSuiteLLMClient(full_model_name, api_key, **kwargs)
    
    if (full_model_name, client.api_key) in _memory:
        client = _memory[(full_model_name, client.api_key)]
        if decorator_configs is not None:
            client._decorator_configs = decorator_configs
        if default_thinking_config is not None:
            client.default_thinking_config = default_thinking_config
        if default_max_tokens is not None:
            client.default_max_tokens = default_max_tokens
        return client
    else:
        _memory[(full_model_name, client.api_key)] = client
        return client


def get_async_client(
    full_model_name: str,
    api_key: ApiKey | None = None,
    decorator_configs: DecoratorConfigs | None = None,
    default_thinking_config: ThinkingConfig | None = None,
    default_max_tokens: int | None = None,
) -> BaseLLMClientAsync:
    global _memory_async
    
    kwargs = {
        "decorator_configs": decorator_configs,
        "default_thinking_config": default_thinking_config,
        "default_max_tokens": default_max_tokens,
    }
    provider_to_client_class: dict[str, type[BaseLLMClientAsync]] = {
        "google": GoogleLLMClientAsync,
        "anthropic": AnthropicLLMClientAsync,
        "openai": OpenaiLLMClientAsync,
        "bedrock": BedrockLLMClientAsync,
    }
    provider, model = full_model_name.split(":", 1)
    if provider in provider_to_client_class:
        client_class = provider_to_client_class[provider]
        if api_key is None:
            client = client_class(model, **kwargs)
        else:
            client = client_class(model, api_key, **kwargs)
    else:
        if api_key is None:
            raise ValueError(f"You should directly provide api_key for this provider: {provider}")
        else:
            client = AiSuiteLLMClientAsync(full_model_name, api_key, **kwargs)
        
    if (full_model_name, client.api_key) in _memory_async:
        client = _memory_async[(full_model_name, client.api_key)]
        if decorator_configs is not None:
            client._decorator_configs = decorator_configs
        if default_thinking_config is not None:
            client.default_thinking_config = default_thinking_config
        if default_max_tokens is not None:
            client.default_max_tokens = default_max_tokens
        return client
    else:
        _memory_async[(full_model_name, client.api_key)] = client
        return client


def get_models_list(provider: str | None = None) -> list[Model]:
    if provider is None:
        models_list: list[Model] = []
        models_list += GoogleLLMClient.models_list()
        models_list += AnthropicLLMClient.models_list()
        models_list += OpenaiLLMClient.models_list()
        models_list += BedrockLLMClient.models_list()
        return models_list
    
    match provider:
        case "google":
            return GoogleLLMClient.models_list()
        case "anthropic":
            return AnthropicLLMClient.models_list()
        case "openai":
            return OpenaiLLMClient.models_list()
        case "bedrock":
            return BedrockLLMClient.models_list()
        case _:
            return []


def configure(
    *,
    decorator_configs: dict[str, DecoratorConfigs] | None = None,
    update_decorator_configs: dict[str, DecoratorConfigs] | None = None,
    thinking_configs: dict[str, ThinkingConfig] | None = None,
    update_thinking_configs: dict[str, ThinkingConfig] | None = None,
    max_tokens: dict[str, int] | None = None,
    update_max_tokens: dict[str, int] | None = None,
    use_logfire: bool | None = None,
):
    if decorator_configs is not None and update_decorator_configs is not None:
        warnings.warn("Both 'decorator_configs' and 'update_decorator_configs' were provided. "
                      "'update_decorator_configs' will be ignored.", UserWarning)
        update_decorator_configs = None
    if thinking_configs is not None and update_thinking_configs is not None:
        warnings.warn("Both 'thinking_configs' and 'update_thinking_configs' were provided. "
                      "'update_thinking_configs' will be ignored.", UserWarning)
        update_thinking_configs = None
    if max_tokens is not None and update_max_tokens is not None:
        warnings.warn("Both 'max_tokens' and 'update_max_tokens' were provided. "
                      "'update_max_tokens' will be ignored.", UserWarning)
        update_max_tokens = None
    
    if decorator_configs is not None:
        GLOBAL_CONFIG.default_decorator_configs = decorator_configs
    if update_decorator_configs is not None:
        GLOBAL_CONFIG.default_decorator_configs.update(update_decorator_configs)
    
    if thinking_configs is not None:
        GLOBAL_CONFIG.default_thinking_configs = thinking_configs
    if update_thinking_configs is not None:
        GLOBAL_CONFIG.default_thinking_configs.update(update_thinking_configs)
    
    if max_tokens is not None:
        GLOBAL_CONFIG.default_max_tokens = max_tokens
    if update_max_tokens is not None:
        GLOBAL_CONFIG.default_max_tokens.update(update_max_tokens)
    
    if use_logfire is not None:
        GLOBAL_CONFIG.use_logfire = use_logfire

def sync_existing_clients_with_global_config():
    for full_model_name, llm_client in chain(_memory.items(), _memory_async.items()):
        if full_model_name in GLOBAL_CONFIG.default_decorator_configs:
            llm_client._decorator_configs = GLOBAL_CONFIG.default_decorator_configs[full_model_name]
        else:
            llm_client._decorator_configs = DecoratorConfigs()
        
        if full_model_name in GLOBAL_CONFIG.default_thinking_configs:
            llm_client.default_thinking_config = GLOBAL_CONFIG.default_thinking_configs[full_model_name]
        else:
            llm_client.default_thinking_config = None
        
        if full_model_name in GLOBAL_CONFIG.default_max_tokens:
            llm_client.default_max_tokens = GLOBAL_CONFIG.default_max_tokens[full_model_name]
        else:
            llm_client.default_max_tokens = None
