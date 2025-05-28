import dataclasses

from promptbuilder.llm_client.utils import DecoratorConfigs


@dataclasses.dataclass
class LlmClientConfigs:
    
    default_decorator_configs: dict[str, DecoratorConfigs] = dataclasses.field(default_factory=dict)
    """Dictionary mapping a model name to the default decorator configs to be used for that model."""
    
    default_max_tokens: dict[str, int] = dataclasses.field(default_factory=dict)
    """Dictionary mapping a model name to the default max_tokens value to be used for that model."""


# The global config is the single global object in promptbuilder.llm_client
# It also does not initialize anything when it's created
GLOBAL_CONFIG = LlmClientConfigs()
