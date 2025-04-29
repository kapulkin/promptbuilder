import inspect
from typing import Callable, ParamSpec, TypeVar, Awaitable


P = ParamSpec("P")
T = TypeVar("T")

async def run_async(f: Callable[P, Awaitable[T]] | Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    if inspect.iscoroutinefunction(f):
        return await f(*args, **kwargs)
    else:
        return f(*args, **kwargs)
