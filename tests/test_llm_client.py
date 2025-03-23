import pytest
from unittest.mock import Mock, patch
from promptbuilder.llm_client import LLMClient, CachedLLMClient
from promptbuilder.llm_client.messages import Completion, Choice, Message, Usage, Response, Candidate, Content, Part, UsageMetadata
import json
import os
import tempfile
import shutil

@pytest.fixture
def mock_aisuite_client():
    with patch('aisuite.Client') as mock_client:
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
        mock_client.return_value.chat.completions.create.return_value = mock_completion
        yield mock_client

@pytest.fixture
def llm_client(mock_aisuite_client):
    return LLMClient(model="test:model", api_key="test-key")

def test_create_output_format(llm_client):
    messages = [Content(parts=[Part(text="Test message")], role="user")]
    response = llm_client.create(messages)
    
    assert isinstance(response, Response)
    assert len(response.candidates) == 1
    assert response.candidates[0].content.parts[0].text == "This is a test response"
    assert response.candidates[0].content.role == "assistant"
    assert response.usage_metadata.prompt_token_count == 10
    assert response.usage_metadata.candidates_token_count == 20
    assert response.usage_metadata.total_token_count == 30

def test_create_text_output_format(llm_client):
    messages = [Content(parts=[Part(text="Test message")], role="user")]
    response = llm_client.create_text(messages)
    
    assert isinstance(response, str)
    assert response == "This is a test response"

@pytest.fixture
def mock_aisuite_client_json():
    with patch('aisuite.Client') as mock_client:
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
        
        mock_client.return_value.chat.completions.create.return_value = mock_completion
        yield mock_client

@pytest.fixture
def llm_client_json(mock_aisuite_client_json):
    return LLMClient(model="test:model", api_key="test-key")

def test_create_structured_output_format(llm_client_json):
    messages = [Content(parts=[Part(text="Test message")], role="user")]
    response = llm_client_json.create_structured(messages)
    
    assert isinstance(response, dict)
    assert response == {"key": "value", "number": 42}

def test_create_structured_with_markdown(llm_client_json):
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
        response = llm_client_json.create_structured(messages)
        
        assert isinstance(response, dict)
        assert response == {"key": "value", "number": 42}

def test_create_invalid_json_raises_error(llm_client):
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
            llm_client.create_structured(messages)

@pytest.fixture
def temp_cache_dir():
    # Create a temporary directory for cache
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after tests
    shutil.rmtree(temp_dir)

@pytest.fixture
def cached_llm_client(llm_client, temp_cache_dir):
    return CachedLLMClient(llm_client, cache_dir=temp_cache_dir)

def test_cached_llm_client_first_call(cached_llm_client, mock_aisuite_client):
    """Test that first call to create() makes an actual API call and caches result"""
    messages = [Content(parts=[Part(text="Test message")], role="user")]
    
    # First call should make an API request
    response = cached_llm_client.create(messages)
    
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

def test_cached_llm_client_cache_hit(cached_llm_client, mock_aisuite_client):
    """Test that second call with same input uses cache"""
    messages = [Content(parts=[Part(text="Test message")], role="user")]
    
    # First call to create cache
    first_response = cached_llm_client.create(messages)
    
    # Reset mock to verify it's not called again
    mock_client = mock_aisuite_client.return_value
    mock_client.chat.completions.create.reset_mock()
    
    # Second call should use cache
    second_response = cached_llm_client.create(messages)
    
    # Verify completions are identical
    assert first_response.candidates[0].content.parts[0].text == second_response.candidates[0].content.parts[0].text
    assert first_response.usage_metadata.total_token_count == second_response.usage_metadata.total_token_count
    
    # Verify no new API call was made
    mock_client.chat.completions.create.assert_not_called()

def test_cached_llm_client_different_messages(cached_llm_client, mock_aisuite_client):
    """Test that different messages create new cache entries"""
    first_messages = [Content(parts=[Part(text="First message")], role="user")]
    second_messages = [Content(parts=[Part(text="Second message")], role="user")]
    
    # First call
    cached_llm_client.create(first_messages)
    
    # Second call with different message
    cached_llm_client.create(second_messages)
    
    # Verify two cache files were created
    cache_files = os.listdir(cached_llm_client.cache_dir)
    assert len(cache_files) == 2

def test_cached_llm_client_invalid_cache_file(cached_llm_client, mock_aisuite_client):
    """Test handling of corrupted cache file"""
    messages = [Content(parts=[Part(text="Test message")], role="user")]
    
    # First call to create cache file
    cached_llm_client.create(messages)
    
    # Corrupt the cache file
    cache_files = os.listdir(cached_llm_client.cache_dir)
    cache_path = os.path.join(cached_llm_client.cache_dir, cache_files[0])
    with open(cache_path, 'w') as f:
        f.write('invalid json')
    
    # Reset mock to verify new API call is made
    mock_client = mock_aisuite_client.return_value
    mock_client.chat.completions.create.reset_mock()
    
    # Next call should make new API request
    response = cached_llm_client.create(messages)
    
    # Verify new API call was made
    mock_client.chat.completions.create.assert_called_once()
    assert isinstance(response, Response)