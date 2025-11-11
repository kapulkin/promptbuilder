import time
import asyncio
import logging
import traceback
from functools import wraps
from typing import Callable, Awaitable, ParamSpec, TypeVar
import tiktoken
from collections import defaultdict

from pydantic import BaseModel

from promptbuilder.llm_client.types import Content



logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def inherited_decorator(decorator: Callable[[Callable[P, T]], Callable[P, T]]) -> Callable[[Callable[P, T]], Callable[P, T]]:
    @wraps(decorator)
    def decorator_with_inheritance(func: Callable) -> Callable:
        new_func = decorator(func)
        new_func._inherit_decorators = getattr(func, "_inherit_decorators", []) + [decorator_with_inheritance]
        return new_func
    return decorator_with_inheritance


class InheritDecoratorsMixin:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        decorators_registry = getattr(cls, "_decorators_registry", defaultdict(list)).copy()
        inherited_parent_decorators = decorators_registry.copy()
        cls._decorators_registry = decorators_registry
        # annotate all decorated methods in the current subclass
        for name, obj in cls.__dict__.items():
            if getattr(obj, "_inherit_decorators", False):
                decorators_registry[name] += obj._inherit_decorators
        # decorate all methods annotated in the registry using parent decorators
        for name, decorators in inherited_parent_decorators.items():
            if name in cls.__dict__:
                for decorator in decorators:
                    setattr(cls, name, decorator(cls.__dict__[name]))


class RetryConfig(BaseModel):
    times: int = 0
    delay: float = 0.

class RpmLimitConfig(BaseModel):
    rpm_limit: int = 0

class TpmLimitConfig(BaseModel):
    tpm_limit: int = 0
    fast: bool = False

class DecoratorConfigs(BaseModel):
    retry: RetryConfig | None = None
    rpm_limit: RpmLimitConfig | None = None
    tpm_limit: TpmLimitConfig | None = None


@inherited_decorator
def retry_cls(class_method: Callable[P, T]) -> Callable[P, T]:
    """
    Retry Decorator
    Retries the wrapped class method `times` times.
    Decorated method must have 'self' as it's first arg.
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param delay: Delay between repeated calls
    :type delay: Float
    """
    @wraps(class_method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_decorator_configs"):
            self._decorator_configs = DecoratorConfigs()
        if self._decorator_configs.retry is None:
            self._decorator_configs.retry = RetryConfig()
        
        attempt = 0
        while attempt < self._decorator_configs.retry.times:
            try:
                return class_method(self, *args, **kwargs)
            except Exception as e:
                logger.error(
                    "Exception thrown when attempting to run %s, attempt %d of %d\n" 
                    "Exception: %s\n%s" % (class_method, attempt, self._decorator_configs.retry.times, e, traceback.format_exc())
                )
                if self._decorator_configs.retry.delay > 0:
                    time.sleep(self._decorator_configs.retry.delay)
                attempt += 1
        return class_method(self, *args, **kwargs)
    return wrapper


@inherited_decorator
def retry_cls_async(class_method: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """
    Async Retry Decorator
    Retries the wrapped class method `times` times.
    Decorated method must have 'self' as it's first arg.
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param delay: Delay between repeated calls
    :type delay: Float
    """
    @wraps(class_method)
    async def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_decorator_configs"):
            self._decorator_configs = DecoratorConfigs()
        if self._decorator_configs.retry is None:
            self._decorator_configs.retry = RetryConfig()
        
        attempt = 0
        while attempt < self._decorator_configs.retry.times:
            try:
                return await class_method(self, *args, **kwargs)
            except Exception as e:
                logger.error(
                    "Exception thrown when attempting to run %s, attempt %d of %d\n" 
                    "Exception: %s\n%s" % (class_method, attempt, self._decorator_configs.retry.times, e, traceback.format_exc())
                )
                if self._decorator_configs.retry.delay > 0:
                    await asyncio.sleep(self._decorator_configs.retry.delay)
                attempt += 1
        return await class_method(self, *args, **kwargs)
    return wrapper

    
@inherited_decorator
def rpm_limit_cls(class_method: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator that limits the number of requests per minute to the decorated class methods
    Decorated methods must have 'self' as it's first arg.
    :param rpm_limit: maximum number of requests per minute. If <= 0, then no limit is imposed
    """
    @wraps(class_method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_decorator_configs"):
            self._decorator_configs = DecoratorConfigs()
        if self._decorator_configs.rpm_limit is None:
            self._decorator_configs.rpm_limit = RpmLimitConfig()
        
        if self._decorator_configs.rpm_limit.rpm_limit <= 0:
            return class_method(self, *args, **kwargs)
        
        if not hasattr(self, "_last_request_time"):
            self._last_request_time = time.time() - 60 / self._decorator_configs.rpm_limit.rpm_limit
        
        while True:
            if time.time() - self._last_request_time < 60 / self._decorator_configs.rpm_limit.rpm_limit:
                diff = 60 / self._decorator_configs.rpm_limit.rpm_limit - (time.time() - self._last_request_time)
                time.sleep(diff)
                continue
            
            self._last_request_time = time.time()
            return class_method(self, *args, **kwargs)
    return wrapper


@inherited_decorator
def rpm_limit_cls_async(class_method: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """
    Decorator that limits the number of requests per minute to the decorated class methods
    Decorated method must have 'self' as it's first arg.
    :param rpm_limit: maximum number of requests per minute. If <= 0, then no limit is imposed
    """
    @wraps(class_method)
    async def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_decorator_configs"):
            self._decorator_configs = DecoratorConfigs()
        if self._decorator_configs.rpm_limit is None:
            self._decorator_configs.rpm_limit = RpmLimitConfig()
        
        if self._decorator_configs.rpm_limit.rpm_limit <= 0:
            return await class_method(self, *args, **kwargs)
        
        if not hasattr(self, "_last_request_time"):
            self._last_request_time = time.time() - 60 / self._decorator_configs.rpm_limit.rpm_limit
        
        while True:
            if time.time() - self._last_request_time < 60 / self._decorator_configs.rpm_limit.rpm_limit:
                diff = 60 / self._decorator_configs.rpm_limit.rpm_limit - (time.time() - self._last_request_time)
                await asyncio.sleep(diff)
                continue
            
            self._last_request_time = time.time()
            return await class_method(self, *args, **kwargs)
    return wrapper


def _estimate_input_tokens_from_messages(self, messages: list[Content], fast: bool = False) -> int:
    """Estimate input tokens for a list[Content] using best available method.

    Priority:
    1) If provider == "google" and a google.genai client is available, use
       models.count_tokens for accurate counts.
    2) If tiktoken is installed, approximate with a BPE encoding.
    3) Fallback heuristic: ~4 characters per token across text parts.
    """
    if not messages:
        return 0

    # Collect text parts for non-Google fallback methods
    texts: list[str] = []
    for m in messages:
        parts = m.parts
        if not parts:
            continue
        for part in parts:
            text = part.text
            if text:
                texts.append(text)

    if not fast:
    # 1) Google Gemini accurate count via genai API (when provider == google)
        if self.provider == "google":
            genai_client = self.client
            contents_arg = "\n".join(texts)
            total_tokens = genai_client.models.count_tokens(
                model=self.model,
                contents=contents_arg,
            ).total_tokens
            return total_tokens

        # 2) tiktoken approximation
        # cl100k_base is a good default for many chat models
        enc = tiktoken.get_encoding("cl100k_base")
        return sum(len(enc.encode(t)) for t in texts)

    else:
        # 3) Heuristic fallback
        total_chars = sum(len(t) for t in texts)
        tokens = total_chars // 4
        return tokens if tokens > 0 else (1 if total_chars > 0 else 0)


@inherited_decorator
def tpm_limit_cls(class_method: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator that limits the number of input tokens per minute to the decorated class methods.
    Decorated methods must have 'self' as its first arg and accept a 'messages' argument
    either positionally (first arg) or by keyword.

    The decorator estimates tokens from input messages and ensures the total tokens
    sent within a 60-second window do not exceed the configured TPM limit. If the
    limit would be exceeded, it waits until the window resets.
    """
    @wraps(class_method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_decorator_configs"):
            self._decorator_configs = DecoratorConfigs()
        if getattr(self._decorator_configs, "tpm_limit", None) is None:
            self._decorator_configs.tpm_limit = TpmLimitConfig()

        limit = self._decorator_configs.tpm_limit.tpm_limit
        if limit <= 0:
            return class_method(self, *args, **kwargs)

        # Extract messages from either kwargs or positional args
        messages = kwargs.get("messages") if "messages" in kwargs else (args[0] if len(args) > 0 else None)
        tokens_needed = _estimate_input_tokens_from_messages(self, messages, self._decorator_configs.tpm_limit.fast)

        # Initialize sliding window state
        now = time.time()
        if not hasattr(self, "_tpm_window_start"):
            self._tpm_window_start = now
        if not hasattr(self, "_tpm_used_tokens"):
            self._tpm_used_tokens = 0

        while True:
            now = time.time()
            elapsed = now - self._tpm_window_start
            if elapsed >= 60:
                # Reset window
                self._tpm_window_start = now
                self._tpm_used_tokens = 0

            if self._tpm_used_tokens + tokens_needed <= limit:
                self._tpm_used_tokens += tokens_needed
                break
            # Need to wait until window resets
            sleep_for = max(0.0, 60 - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)
                continue
            # If sleep_for == 0, loop will reset on next iteration

        return class_method(self, *args, **kwargs)
    return wrapper


@inherited_decorator
def tpm_limit_cls_async(class_method: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """
    Async variant of TPM limiter.
    """
    @wraps(class_method)
    async def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_decorator_configs"):
            self._decorator_configs = DecoratorConfigs()
        if getattr(self._decorator_configs, "tpm_limit", None) is None:
            self._decorator_configs.tpm_limit = TpmLimitConfig()

        limit = self._decorator_configs.tpm_limit.tpm_limit
        if limit <= 0:
            return await class_method(self, *args, **kwargs)

        messages = kwargs.get("messages") if "messages" in kwargs else (args[0] if len(args) > 0 else None)
        tokens_needed = _estimate_input_tokens_from_messages(self, messages, self._decorator_configs.tpm_limit.fast)

        now = time.time()
        if not hasattr(self, "_tpm_window_start"):
            self._tpm_window_start = now
        if not hasattr(self, "_tpm_used_tokens"):
            self._tpm_used_tokens = 0

        while True:
            now = time.time()
            elapsed = now - self._tpm_window_start
            if elapsed >= 60:
                self._tpm_window_start = now
                self._tpm_used_tokens = 0

            if self._tpm_used_tokens + tokens_needed <= limit:
                self._tpm_used_tokens += tokens_needed
                break

            sleep_for = max(0.0, 60 - elapsed)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
                continue

        return await class_method(self, *args, **kwargs)
    return wrapper
