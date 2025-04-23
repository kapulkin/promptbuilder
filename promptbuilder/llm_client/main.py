from promptbuilder.llm_client.base_client import BaseLLMClient, BaseLLMClientAsync
from promptbuilder.llm_client.base_configs import base_decorator_configs
from promptbuilder.llm_client.utils import DecoratorConfigs
from promptbuilder.llm_client.google_llm_client import GoogleLLMClient, GoogleLLMClientAsync


_memory: dict[str, BaseLLMClient] = {}
_memory_async: dict[str, BaseLLMClientAsync] = {}


def get_client(model: str, decorator_configs: DecoratorConfigs | None = None) -> BaseLLMClient:
    global _memory
    
    if model not in _memory:
        if decorator_configs is None:
            decorator_configs = base_decorator_configs[model]
        provider_name, model_name = model.split(":")
        if provider_name == "google":
            client = GoogleLLMClient(model_name, decorator_configs=decorator_configs)
            _memory[model] = client
            return _memory[model]
        else:
            raise ValueError(f"Sorry, this library doesn't support {provider_name} models")
    else:
        client = _memory[model]
        if decorator_configs is not None:
            client._decorator_configs = decorator_configs
        return client


def get_async_client(model: str, decorator_configs: DecoratorConfigs | None = None) -> BaseLLMClientAsync:
    global _memory_async
    
    if model not in _memory_async:
        if decorator_configs is None:
            decorator_configs = base_decorator_configs[model]
        provider_name, model_name = model.split(":")
        if provider_name == "google":
            client = GoogleLLMClientAsync(model_name, decorator_configs=decorator_configs)
            _memory_async[model] = client
            return _memory_async[model]
        else:
            raise ValueError(f"Sorry, this library doesn't support {provider_name} models")
    else:
        client = _memory_async[model]
        if decorator_configs is not None:
            client._decorator_configs = decorator_configs
        return client
