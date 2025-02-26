import pytest
from unittest.mock import Mock, patch
from promptbuilder.llm_client import LLMClient, CachedLLMClient, Completion, Choice, Message, Usage
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
    messages = [{"role": "user", "content": "Test message"}]
    completion = llm_client.create(messages)
    
    assert isinstance(completion, Completion)
    assert len(completion.choices) == 1
    assert completion.choices[0].message.content == "This is a test response"
    assert completion.choices[0].message.role == "assistant"
    assert completion.usage.prompt_tokens == 10
    assert completion.usage.completion_tokens == 20
    assert completion.usage.total_tokens == 30

def test_create_text_output_format(llm_client):
    messages = [{"role": "user", "content": "Test message"}]
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
    messages = [{"role": "user", "content": "Test message"}]
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
        
        messages = [{"role": "user", "content": "Test message"}]
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
        
        messages = [{"role": "user", "content": "Test message"}]
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
    messages = [{"role": "user", "content": "Test message"}]
    
    # First call should make an API request
    completion = cached_llm_client.create(messages)
    
    # Verify the completion
    assert isinstance(completion, Completion)
    assert len(completion.choices) == 1
    assert completion.choices[0].message.content == "This is a test response"
    
    # Verify that the mock was called once
    mock_client = mock_aisuite_client.return_value
    mock_client.chat.completions.create.assert_called_once()
    
    # Verify cache file was created
    cache_files = os.listdir(cached_llm_client.cache_dir)
    assert len(cache_files) == 1
    assert cache_files[0].endswith('.json')

def test_cached_llm_client_cache_hit(cached_llm_client, mock_aisuite_client):
    """Test that second call with same input uses cache"""
    messages = [{"role": "user", "content": "Test message"}]
    
    # First call to create cache
    first_completion = cached_llm_client.create(messages)
    
    # Reset mock to verify it's not called again
    mock_client = mock_aisuite_client.return_value
    mock_client.chat.completions.create.reset_mock()
    
    # Second call should use cache
    second_completion = cached_llm_client.create(messages)
    
    # Verify completions are identical
    assert first_completion.choices[0].message.content == second_completion.choices[0].message.content
    assert first_completion.usage.total_tokens == second_completion.usage.total_tokens
    
    # Verify no new API call was made
    mock_client.chat.completions.create.assert_not_called()

def test_cached_llm_client_different_messages(cached_llm_client, mock_aisuite_client):
    """Test that different messages create new cache entries"""
    first_messages = [{"role": "user", "content": "First message"}]
    second_messages = [{"role": "user", "content": "Second message"}]
    
    # First call
    cached_llm_client.create(first_messages)
    
    # Second call with different message
    cached_llm_client.create(second_messages)
    
    # Verify two cache files were created
    cache_files = os.listdir(cached_llm_client.cache_dir)
    assert len(cache_files) == 2

def test_cached_llm_client_invalid_cache_file(cached_llm_client, mock_aisuite_client):
    """Test handling of corrupted cache file"""
    messages = [{"role": "user", "content": "Test message"}]
    
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
    completion = cached_llm_client.create(messages)
    
    # Verify new API call was made
    mock_client.chat.completions.create.assert_called_once()
    assert isinstance(completion, Completion)