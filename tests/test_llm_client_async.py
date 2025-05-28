import pytest
from unittest.mock import Mock, patch, AsyncMock
from promptbuilder.llm_client import CachedLLMClientAsync
from promptbuilder.llm_client import get_client, get_async_client
from promptbuilder.llm_client.aisuite_client import AiSuiteLLMClientAsync
from promptbuilder.llm_client.messages import Completion, Choice, Message, Usage, Response, Candidate, Content, Part, UsageMetadata
import json
import os
import tempfile
import shutil
import asyncio

@pytest.fixture
def mock_aisuite_client():
    with patch('aisuite_async.AsyncClient') as mock_client:
        # Create a mock completion response
        mock_completion = Completion(
            choices=[
                Choice(
                    message=Message(
                        role="assistant",
                        content="This is a test response"
                    )
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )
        )
        
        # Set up the mock client to return our mock completion
        mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_completion)
        yield mock_client

@pytest.fixture
def llm_client(mock_aisuite_client):
    return AiSuiteLLMClientAsync(model="test:model", api_key="test-key")

@pytest.mark.asyncio
async def test_create_output_format(llm_client):
    messages = [Content(parts=[Part(text="Test message")], role="user")]
    response = await llm_client.create(messages)
    
    assert isinstance(response, Response)
    assert len(response.candidates) == 1
    assert response.candidates[0].content.parts[0].text == "This is a test response"
    assert response.candidates[0].content.role == "model"
    assert response.usage_metadata.prompt_token_count == 10
    assert response.usage_metadata.candidates_token_count == 20
    assert response.usage_metadata.total_token_count == 30

@pytest.mark.asyncio
async def test_create_text_output_format(llm_client):
    messages = [Content(parts=[Part(text="Test message")], role="user")]
    response = await llm_client.create_value(messages)
    
    assert isinstance(response, str)
    assert response == "This is a test response"

@pytest.fixture
def mock_aisuite_client_json():
    with patch('aisuite_async.AsyncClient') as mock_client:
        # Create a mock completion with JSON response
        mock_completion = Completion(
            choices=[
                Choice(
                    message=Message(
                        role="assistant",
                        content='{"key": "value", "number": 42}'
                    )
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )
        )
        
        mock_client.return_value.chat.completions.create = AsyncMock(return_value=mock_completion)
        yield mock_client

@pytest.fixture
def llm_client_json(mock_aisuite_client_json):
    return AiSuiteLLMClientAsync(model="test:model", api_key="test-key")

@pytest.mark.asyncio
async def test_create_structured_output_format(llm_client_json):
    messages = [Content(parts=[Part(text="Test message")], role="user")]
    response = await llm_client_json.create_value(messages, result_type="json")
    
    assert isinstance(response, dict)
    assert response == {"key": "value", "number": 42}

@pytest.mark.asyncio
async def test_create_structured_with_markdown(llm_client_json):
    with patch.object(llm_client_json.client.chat.completions, 'create') as mock_create:
        mock_create.return_value = Completion(
            choices=[
                Choice(
                    message=Message(
                        role="assistant",
                        content='```json\n{"key": "value", "number": 42}\n```'
                    )
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )
        )
        
        messages = [Content(parts=[Part(text="Test message")], role="user")]
        response = await llm_client_json.create_value(messages, result_type="json")
        
        assert isinstance(response, dict)
        assert response == {"key": "value", "number": 42}

@pytest.mark.asyncio
async def test_create_invalid_json_raises_error(llm_client):
    with patch.object(llm_client.client.chat.completions, 'create') as mock_create:
        mock_create.return_value = Completion(
            choices=[
                Choice(
                    message=Message(
                        role="assistant",
                        content='Invalid JSON response'
                    )
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )
        )
        
        messages = [Content(parts=[Part(text="Test message")], role="user")]
        with pytest.raises(ValueError):
            await llm_client.create_value(messages, result_type="json")

@pytest.mark.asyncio
async def test_from_text(llm_client):
    response = await llm_client.from_text("Test prompt")
    assert isinstance(response, str)
    assert response == "This is a test response"

@pytest.mark.asyncio
async def test_from_text_structured(llm_client_json):
    response = await llm_client_json.from_text("Test prompt", result_type="json")
    assert isinstance(response, dict)
    assert response == {"key": "value", "number": 42}

@pytest.mark.asyncio
async def test_with_system_message(llm_client):
    response = await llm_client.create_value(
        messages=[Content(parts=[Part(text="Test message")], role="user")],
        system_message="You are a helpful assistant"
    )
    assert isinstance(response, str)
    assert response == "This is a test response"

@pytest.mark.asyncio
async def test_create_with_system_message(llm_client):
    messages = [Content(parts=[Part(text="Test message")], role="user")]
    response = await llm_client.create(
        messages,
        system_message="You are a helpful assistant"
    )
    assert isinstance(response, Response)
    assert len(response.candidates) == 1
    assert response.candidates[0].content.parts[0].text == "This is a test response" 

@pytest.fixture
def temp_cache_dir():
    # Create a temporary directory for cache
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after tests
    shutil.rmtree(temp_dir)


@pytest.fixture
def cached_llm_client(llm_client, temp_cache_dir):
    return CachedLLMClientAsync(llm_client, cache_dir=temp_cache_dir)

@pytest.mark.asyncio
async def test_cached_llm_client_first_call(cached_llm_client, mock_aisuite_client):
    """Test that first call to create() makes an actual API call and caches result"""
    messages = [Content(parts=[Part(text="Test message")], role="user")]
    
    # First call should make an API request
    response = await cached_llm_client.create(messages)
    
    # Verify the completion
    assert isinstance(response, Response)
    assert len(response.candidates) == 1
    assert response.candidates[0].content.parts[0].text == "This is a test response"
    
    # Verify that the mock was called once
    mock_client = mock_aisuite_client.return_value
    mock_client.chat.completions.create.assert_called_once()
    
    # Verify cache file was created
    cache_files = os.listdir(cached_llm_client.cache_dir)
    assert len(cache_files) == 1
    assert cache_files[0].endswith('.json')