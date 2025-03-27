import inspect

async def run_async(f, **kwargs):
    if inspect.iscoroutinefunction(f):
        return await f(**kwargs)
    else:
        return f(**kwargs)
