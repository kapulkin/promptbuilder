import time
import asyncio
import logging
import traceback
from functools import wraps
from typing import Callable, Awaitable, Any, ParamSpec, TypeVar
from collections import defaultdict

from pydantic import BaseModel


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
    last_request_time: float | None = None

class DecoratorConfigs(BaseModel):
    retry: RetryConfig | None = None
    rpm_limit: RpmLimitConfig | None = None


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
        
        if self._decorator_configs.rpm_limit.last_request_time is None:
            self._decorator_configs.rpm_limit.last_request_time = time.time() - 60 / self._decorator_configs.rpm_limit.rpm_limit
        
        while True:
            if time.time() - self._decorator_configs.rpm_limit.last_request_time < 60 / self._decorator_configs.rpm_limit.rpm_limit:
                diff = 60 / self._decorator_configs.rpm_limit.rpm_limit - (time.time() - self._decorator_configs.rpm_limit.last_request_time)
                time.sleep(diff)
                continue
            
            self._decorator_configs.rpm_limit.last_request_time = time.time()
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
        
        if self._decorator_configs.rpm_limit.last_request_time is None:
            self._decorator_configs.rpm_limit.last_request_time = time.time() - 60 / self._decorator_configs.rpm_limit.rpm_limit
        
        while True:
            if time.time() - self._decorator_configs.rpm_limit.last_request_time < 60 / self._decorator_configs.rpm_limit.rpm_limit:
                diff = 60 / self._decorator_configs.rpm_limit.rpm_limit - (time.time() - self._decorator_configs.rpm_limit.last_request_time)
                await asyncio.sleep(diff)
                continue
            
            self._decorator_configs.rpm_limit.last_request_time = time.time()
            return await class_method(self, *args, **kwargs)
    return wrapper
