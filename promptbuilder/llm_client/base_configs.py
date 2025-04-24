from collections import defaultdict

from promptbuilder.llm_client.utils import RetryConfig, RpmLimitConfig, DecoratorConfigs


base_decorator_configs = defaultdict(lambda: DecoratorConfigs(), {
    "google:gemini-2.5-flash-preview-04-17": DecoratorConfigs(retry=RetryConfig(times=3, delay=1), rpm_limit=RpmLimitConfig(rpm_limit=1000)),
    "google:gemini-2.5-pro-preview-03-25": DecoratorConfigs(retry=RetryConfig(times=3, delay=1), rpm_limit=RpmLimitConfig(rpm_limit=150)),
    "google:gemini-2.0-flash": DecoratorConfigs(retry=RetryConfig(times=3, delay=1), rpm_limit=RpmLimitConfig(rpm_limit=2000)),
    "anthropic:claude-3-7-sonnet-latest": DecoratorConfigs(retry=RetryConfig(times=3, delay=1), rpm_limit=RpmLimitConfig(rpm_limit=50)),
    "anthropic:claude-3-5-haiku-latest": DecoratorConfigs(retry=RetryConfig(times=3, delay=1), rpm_limit=RpmLimitConfig(rpm_limit=50)),
    "anthropic:claude-3-5-sonnet-latest": DecoratorConfigs(retry=RetryConfig(times=3, delay=1), rpm_limit=RpmLimitConfig(rpm_limit=50)),
    "anthropic:claude-3-opus-latest": DecoratorConfigs(retry=RetryConfig(times=3, delay=1), rpm_limit=RpmLimitConfig(rpm_limit=50)),
})

base_default_max_tokens_configs = defaultdict(lambda: 8192)