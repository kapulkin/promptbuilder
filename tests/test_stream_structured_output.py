import pytest
from pydantic import BaseModel

import promptbuilder.llm_client.google_client as google_mod
import promptbuilder.llm_client.openai_client as openai_mod
from promptbuilder.llm_client.base_client import BaseLLMClient, BaseLLMClientAsync
from promptbuilder.llm_client.google_client import GoogleLLMClient, GoogleLLMClientAsync
from promptbuilder.llm_client.openai_client import OpenaiLLMClient, OpenaiLLMClientAsync
from promptbuilder.llm_client.types import Candidate, Content, Part, Response


class StructuredPayload(BaseModel):
    value: int


class _BaseStructuredStreamClient(BaseLLMClient):
    @property
    def api_key(self) -> str:
        return "test-key"

    def _create(self, messages, result_type=None, **kwargs):
        raise NotImplementedError

    def _create_stream(self, messages, result_type=None, **kwargs):
        yield Response(candidates=[Candidate(content=Content(parts=[Part(text='{"value":')], role="model"))])
        yield Response(candidates=[Candidate(content=Content(parts=[Part(text="1}")], role="model"))])


class _BaseStructuredStreamClientAsync(BaseLLMClientAsync):
    @property
    def api_key(self) -> str:
        return "test-key"

    async def _create(self, messages, result_type=None, **kwargs):
        raise NotImplementedError

    async def _create_stream(self, messages, result_type=None, **kwargs):
        async def _iterator():
            yield Response(candidates=[Candidate(content=Content(parts=[Part(text='{"value":')], role="model"))])
            yield Response(candidates=[Candidate(content=Content(parts=[Part(text="1}")], role="model"))])

        return _iterator()


def test_base_create_stream_parses_json_chunks():
    client = _BaseStructuredStreamClient(provider="test", model="model")

    chunks = list(client.create_stream(messages=[], result_type="json"))

    assert chunks[-1].parsed == {"value": 1}


def test_base_create_stream_parses_pydantic_chunks():
    client = _BaseStructuredStreamClient(provider="test", model="model")

    chunks = list(client.create_stream(messages=[], result_type=StructuredPayload))

    assert chunks[-1].parsed == StructuredPayload(value=1)


@pytest.mark.asyncio
async def test_base_create_stream_async_parses_pydantic_chunks():
    client = _BaseStructuredStreamClientAsync(provider="test", model="model")

    chunks = []
    async for chunk in client.create_stream(messages=[], result_type=StructuredPayload):
        chunks.append(chunk)

    assert chunks[-1].parsed == StructuredPayload(value=1)


class _FakeOpenAIUsage:
    output_tokens = 3
    input_tokens = 2
    total_tokens = 5


class _FakeOpenAICompletedResponse:
    usage = _FakeOpenAIUsage()


class _FakeOpenAIEvent:
    def __init__(self, event_type, delta=None):
        self.type = event_type
        self.delta = delta
        self.response = _FakeOpenAICompletedResponse()


class _FakeOpenAIStream:
    def __init__(self, parsed):
        self._parsed = parsed

    def __iter__(self):
        yield _FakeOpenAIEvent("response.output_text.delta", '{"value":')
        yield _FakeOpenAIEvent("response.output_text.delta", "1}")
        yield _FakeOpenAIEvent("response.completed")

    def get_final_response(self):
        class _FinalResponse:
            output_parsed = self._parsed

        return _FinalResponse()


class _FakeOpenAIStreamManager:
    def __init__(self, parsed):
        self._stream = _FakeOpenAIStream(parsed)

    def __enter__(self):
        return self._stream

    def __exit__(self, exc_type, exc, exc_tb):
        return None


class _FakeOpenAIResponses:
    def __init__(self, recorder, parsed):
        self.recorder = recorder
        self.parsed = parsed

    def stream(self, **kwargs):
        self.recorder["stream_kwargs"] = kwargs
        return _FakeOpenAIStreamManager(self.parsed)


class _FakeOpenAIClient:
    def __init__(self, recorder, parsed):
        self.responses = _FakeOpenAIResponses(recorder, parsed)


def test_openai_stream_json_configured(monkeypatch):
    rec = {}
    monkeypatch.setattr(openai_mod, "OpenAI", lambda api_key=None: _FakeOpenAIClient(rec, None))
    client = OpenaiLLMClient(model="dummy", api_key="test-key")

    chunks = list(client._create_stream(messages=[Content(parts=[Part(text="hi")], role="user")], result_type="json"))

    assert rec["stream_kwargs"]["text"] == {"format": {"type": "json_object"}}
    assert chunks[-1].parsed == {"value": 1}


def test_openai_stream_pydantic_configured(monkeypatch):
    rec = {}
    monkeypatch.setattr(openai_mod, "OpenAI", lambda api_key=None: _FakeOpenAIClient(rec, StructuredPayload(value=1)))
    client = OpenaiLLMClient(model="dummy", api_key="test-key")

    chunks = list(client._create_stream(messages=[Content(parts=[Part(text="hi")], role="user")], result_type=StructuredPayload))

    assert rec["stream_kwargs"]["text_format"] is StructuredPayload
    assert chunks[-1].parsed == StructuredPayload(value=1)


class _FakeAsyncOpenAIStream:
    def __init__(self, parsed):
        self._parsed = parsed

    async def __aiter__(self):
        yield _FakeOpenAIEvent("response.output_text.delta", '{"value":')
        yield _FakeOpenAIEvent("response.output_text.delta", "1}")
        yield _FakeOpenAIEvent("response.completed")

    async def get_final_response(self):
        class _FinalResponse:
            output_parsed = self._parsed

        return _FinalResponse()


class _FakeAsyncOpenAIStreamManager:
    def __init__(self, parsed):
        self._stream = _FakeAsyncOpenAIStream(parsed)

    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, exc_type, exc, exc_tb):
        return None


class _FakeAsyncOpenAIResponses:
    def __init__(self, recorder, parsed):
        self.recorder = recorder
        self.parsed = parsed

    def stream(self, **kwargs):
        self.recorder["stream_kwargs_async"] = kwargs
        return _FakeAsyncOpenAIStreamManager(self.parsed)


class _FakeAsyncOpenAIClient:
    def __init__(self, recorder, parsed):
        self.responses = _FakeAsyncOpenAIResponses(recorder, parsed)


@pytest.mark.asyncio
async def test_openai_stream_async_pydantic_configured(monkeypatch):
    rec = {}
    monkeypatch.setattr(openai_mod, "AsyncOpenAI", lambda api_key=None: _FakeAsyncOpenAIClient(rec, StructuredPayload(value=1)))
    client = OpenaiLLMClientAsync(model="dummy", api_key="test-key")

    chunks = []
    stream = await client._create_stream(messages=[Content(parts=[Part(text="hi")], role="user")], result_type=StructuredPayload)
    async for chunk in stream:
        chunks.append(chunk)

    assert rec["stream_kwargs_async"]["text_format"] is StructuredPayload
    assert chunks[-1].parsed == StructuredPayload(value=1)


class _FakeGoogleModels:
    def __init__(self, recorder):
        self.recorder = recorder

    def generate_content_stream(self, *, model, contents, config):
        self.recorder["last_config"] = config
        return iter(())


class _FakeGoogleClient:
    def __init__(self, recorder):
        self.models = _FakeGoogleModels(recorder)


def test_google_stream_json_configured(monkeypatch):
    rec = {}
    monkeypatch.setattr(google_mod, "Client", lambda api_key=None, **kwargs: _FakeGoogleClient(rec))
    client = GoogleLLMClient(model="gemini-test", api_key="k")

    list(client._create_stream(messages=[Content(parts=[Part(text="hi")], role="user")], result_type="json"))

    assert rec["last_config"].response_mime_type == "application/json"


def test_google_stream_pydantic_configured(monkeypatch):
    rec = {}
    monkeypatch.setattr(google_mod, "Client", lambda api_key=None, **kwargs: _FakeGoogleClient(rec))
    client = GoogleLLMClient(model="gemini-test", api_key="k")

    list(client._create_stream(messages=[Content(parts=[Part(text="hi")], role="user")], result_type=StructuredPayload))

    assert rec["last_config"].response_mime_type == "application/json"
    assert rec["last_config"].response_schema is StructuredPayload


class _FakeAioGoogleModels:
    def __init__(self, recorder):
        self.recorder = recorder

    async def generate_content_stream(self, *, model, contents, config):
        self.recorder["last_config_async"] = config

        async def _empty_stream():
            if False:
                yield None

        return _empty_stream()


class _FakeAioGoogleWrapper:
    def __init__(self, recorder):
        self.models = _FakeAioGoogleModels(recorder)


class _FakeGoogleClientAsync:
    def __init__(self, recorder):
        self.aio = _FakeAioGoogleWrapper(recorder)


@pytest.mark.asyncio
async def test_google_stream_async_pydantic_configured(monkeypatch):
    rec = {}
    monkeypatch.setattr(google_mod, "Client", lambda api_key=None, **kwargs: _FakeGoogleClientAsync(rec))
    client = GoogleLLMClientAsync(model="gemini-test", api_key="k")

    stream = await client._create_stream(messages=[Content(parts=[Part(text="hi")], role="user")], result_type=StructuredPayload)
    async for _ in stream:
        pass

    assert rec["last_config_async"].response_mime_type == "application/json"
    assert rec["last_config_async"].response_schema is StructuredPayload