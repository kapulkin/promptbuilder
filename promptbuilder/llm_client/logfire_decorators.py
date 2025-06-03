import time
from functools import wraps
from contextlib import contextmanager
from typing import Callable, Awaitable, ParamSpec, Any, Iterator, AsyncIterator, Iterable, AsyncIterable

import logfire

from promptbuilder.llm_client.utils import inherited_decorator
from promptbuilder.llm_client.config import GLOBAL_CONFIG
from promptbuilder.llm_client.types import Response, UsageMetadata


P = ParamSpec("P")


def extract_span_data(self, *args, **kwargs) -> dict[str, Any]:
    span_data: dict[str, Any] = {}
    
    messages = kwargs.pop("messages", None) or args[0]
    # by filling the attributes "request_data" and "response_data" we will get a human-readable format
    logfire_messages = []
    system_message = kwargs.get("system_message")
    if system_message is not None:
        logfire_messages.append({"role": "system", "content": system_message})
    for content in messages:
        if content.role == "user":
            logfire_messages.append({"role": "user", "content": content.as_str()})
        elif content.role == "model":
            logfire_messages.append({"role": "assistant", "content": content.as_str()})
    span_data["request_data"] = {"messages": logfire_messages}
    span_data["messages"] = messages
    
    result_type = kwargs.pop("result_type", None)
    if result_type is None and len(args) == 2:
        result_type = args[1]
    span_data["result_type"] = result_type
    
    span_data["provider"] = self.provider
    span_data["model_name"] = self.model
    span_data["full_model_name"] = self.full_model_name
    span_data["kwargs"] = kwargs
    
    return span_data


@inherited_decorator
def create(class_method: Callable[P, Response]) -> Callable[P, Response]:
    """
    Decorator to log llm client's create method using logfire
    """
    @wraps(class_method)
    def wrapper(self, *args, **kwargs):
        if not GLOBAL_CONFIG.use_logfire:
            return class_method(self, *args, **kwargs)
        
        logfire_llm = logfire.with_settings(tags=["LLM"])
        span_data = extract_span_data(self, *args, **kwargs)
        with logfire_llm.span(f"Create with {span_data["full_model_name"]}", **span_data) as span:
            start_time = time.time()
            response = class_method(self, *args, **kwargs)
            span.set_attribute("duration", time.time() - start_time)
            
            span.set_attribute("response_data", {"message": {"role": "assistant", "content": response.text}})
            span.set_attribute("candidates", response.candidates)
            span.set_attribute("parsed", response.parsed)
            span.set_attribute("response_text", response.text)
            if response.usage_metadata is not None:
                span.set_attribute("usage_metadata.cached_content_token_count", response.usage_metadata.cached_content_token_count)
                span.set_attribute("usage_metadata.candidates_token_count", response.usage_metadata.candidates_token_count)
                span.set_attribute("usage_metadata.prompt_token_count", response.usage_metadata.prompt_token_count)
                span.set_attribute("usage_metadata.total_token_count", response.usage_metadata.total_token_count)
            
            return response
    
    return wrapper


@inherited_decorator
def create_async(class_method: Callable[P, Awaitable[Response]]) -> Callable[P, Awaitable[Response]]:
    """
    Decorator to log llm client's async create method using logfire
    """    
    @wraps(class_method)
    async def wrapper(self, *args, **kwargs):
        if not GLOBAL_CONFIG.use_logfire:
            return await class_method(self, *args, **kwargs)
        
        logfire_llm = logfire.with_settings(tags=["LLM"])
        span_data = extract_span_data(self, *args, **kwargs)
        with logfire_llm.span(f"Async create with {span_data["full_model_name"]}", **span_data) as span:
            start_time = time.time()
            response = await class_method(self, *args, **kwargs)
            span.set_attribute("duration", time.time() - start_time)
            
            span.set_attribute("response_data", {"message": {"role": "assistant", "content": response.text}})
            span.set_attribute("candidates", response.candidates)
            span.set_attribute("parsed", response.parsed)
            span.set_attribute("response_text", response.text)
            if response.usage_metadata is not None:
                span.set_attribute("usage_metadata.cached_content_token_count", response.usage_metadata.cached_content_token_count)
                span.set_attribute("usage_metadata.candidates_token_count", response.usage_metadata.candidates_token_count)
                span.set_attribute("usage_metadata.prompt_token_count", response.usage_metadata.prompt_token_count)
                span.set_attribute("usage_metadata.total_token_count", response.usage_metadata.total_token_count)
            
            return response
            
    return wrapper


class LogfireStreamState:
    def __init__(self):
        self._content: list[str] = []
        self.last_usage_data = UsageMetadata()

    def record_chunk(self, chunk: Response):
        if chunk.text:
            self._content.append(chunk.text)
        if chunk.usage_metadata is not None:
            self.last_usage_data = chunk.usage_metadata

    def get_response_data(self) -> Any:
        return {"message": {"role": "assistant", "content": "".join(self._content)}}

@contextmanager
def record_streaming(span: logfire.LogfireSpan):
    stream_state = LogfireStreamState()

    def record_chunk(chunk: Response):
        if chunk:
            stream_state.record_chunk(chunk)

    start_time = time.time()
    try:
        yield record_chunk
    finally:
        span.set_attribute("duration", time.time() - start_time)
        
        span.set_attribute("response_data", stream_state.get_response_data())
        span.set_attribute("response_text", stream_state.get_response_data()["message"]["content"])
        span.set_attribute("usage_metadata.cached_content_token_count", stream_state.last_usage_data.cached_content_token_count)
        span.set_attribute("usage_metadata.candidates_token_count", stream_state.last_usage_data.candidates_token_count)
        span.set_attribute("usage_metadata.prompt_token_count", stream_state.last_usage_data.prompt_token_count)
        span.set_attribute("usage_metadata.total_token_count", stream_state.last_usage_data.total_token_count)


@inherited_decorator
def create_stream(class_method: Callable[P, Iterable[Response]]) -> Callable[P, Iterable[Response]]:
    """
    Decorator to log llm client's create_stream method using logfire
    """    
    @wraps(class_method)
    def wrapper(self, *args, **kwargs):
        if not GLOBAL_CONFIG.use_logfire:
            return class_method(self, *args, **kwargs)
        
        logfire_llm = logfire.with_settings(tags=["LLM"])        
        span_data = extract_span_data(self, *args, **kwargs)
        stream_response = class_method(self, *args, **kwargs)
        
        class LogfireStreamIterator:
            def __init__(self, stream_iterator: Iterator[Response]):
                self._stream_iterator = stream_iterator

            def __iter__(self) -> Iterator[Response]:
                with logfire_llm.span(f"Create stream with {span_data["full_model_name"]}", **span_data) as span:
                    with record_streaming(span) as record_chunk:
                        for chunk in self._stream_iterator:
                            record_chunk(chunk)
                            yield chunk
        
        return LogfireStreamIterator(stream_response)
    
    return wrapper

@inherited_decorator
def create_stream_async(class_method: Callable[P, AsyncIterable[Response]]) -> Callable[P, AsyncIterable[Response]]:
    """
    Decorator to log llm client's async create_stream method using logfire
    """    
    @wraps(class_method)
    async def wrapper(self, *args, **kwargs):
        if not GLOBAL_CONFIG.use_logfire:
            return await class_method(self, *args, **kwargs)
        
        logfire_llm = logfire.with_settings(tags=["LLM"])        
        span_data = extract_span_data(self, *args, **kwargs)
        stream_response = await class_method(self, *args, **kwargs)
        
        class LogfireAsyncStreamIterator:
            def __init__(self, async_stream_iterator: AsyncIterator[Response]):
                self._async_stream_iterator = async_stream_iterator

            async def __aiter__(self) -> AsyncIterator[Response]:
                with logfire_llm.span(f"Async create stream with {span_data["full_model_name"]}", **span_data) as span:
                    with record_streaming(span) as record_chunk:
                        async for chunk in self._async_stream_iterator:
                            record_chunk(chunk)
                            yield chunk
        
        return LogfireAsyncStreamIterator(stream_response)
    
    return wrapper
