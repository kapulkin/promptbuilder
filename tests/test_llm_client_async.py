import pytest
from unittest.mock import Mock, patch, AsyncMock
from promptbuilder.llm_client import LLMClientAsync, BaseLLMClient
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
    return LLMClientAsync(model="test:model", api_key="test-key")

@pytest.mark.asyncio
async def test_create_output_format(llm_client):
    messages = [Content(parts=[Part(text="Test message")], role=BaseLLMClient.user_tag)]
    response = await llm_client.create(messages)
    
    assert isinstance(response, Response)
    assert len(response.candidates) == 1
    assert response.candidates[0].content.parts[0].text == "This is a test response"
    assert response.candidates[0].content.role == BaseLLMClient.assistant_tag
    assert response.usage_metadata.prompt_token_count == 10
    assert response.usage_metadata.candidates_token_count == 20
    assert response.usage_metadata.total_token_count == 30

@pytest.mark.asyncio
async def test_create_text_output_format(llm_client):
    messages = [Content(parts=[Part(text="Test message")], role=BaseLLMClient.user_tag)]
    response = await llm_client.create_text(messages)
    
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
    return LLMClientAsync(model="test:model", api_key="test-key")

@pytest.mark.asyncio
async def test_create_structured_output_format(llm_client_json):
    messages = [Content(parts=[Part(text="Test message")], role=BaseLLMClient.user_tag)]
    response = await llm_client_json.create_structured(messages)
    
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
        
        messages = [Content(parts=[Part(text="Test message")], role=BaseLLMClient.user_tag)]
        response = await llm_client_json.create_structured(messages)
        
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
        
        messages = [Content(parts=[Part(text="Test message")], role=BaseLLMClient.user_tag)]
        with pytest.raises(ValueError):
            await llm_client.create_structured(messages)

@pytest.mark.asyncio
async def test_from_text(llm_client):
    response = await llm_client.from_text("Test prompt")
    assert isinstance(response, str)
    assert response == "This is a test response"

@pytest.mark.asyncio
async def test_from_text_structured(llm_client_json):
    response = await llm_client_json.from_text_structured("Test prompt")
    assert isinstance(response, dict)
    assert response == {"key": "value", "number": 42}

@pytest.mark.asyncio
async def test_with_system_message(llm_client):
    response = await llm_client.with_system_message(
        system_message="You are a helpful assistant",
        input="Test message"
    )
    assert isinstance(response, str)
    assert response == "This is a test response"

@pytest.mark.asyncio
async def test_create_with_system_message(llm_client):
    messages = [Content(parts=[Part(text="Test message")], role=BaseLLMClient.user_tag)]
    response = await llm_client.create(
        messages,
        system_message="You are a helpful assistant"
    )
    assert isinstance(response, Response)
    assert len(response.candidates) == 1
    assert response.candidates[0].content.parts[0].text == "This is a test response" 