import os
import asyncio
import dotenv
dotenv.load_dotenv(".env")
from promptbuilder.llm_client import get_async_client
from promptbuilder.llm_client.types import Content, Part

"""
Example: Using LiteLLM with Ollama via PromptBuilder

Prerequisites:
- Install litellm (pip install litellm)
- Run an Ollama server locally (default http://localhost:11434)
- Set environment variable OLLAMA_BASE_URL to your Ollama server, e.g.:
    Windows PowerShell:
        $env:OLLAMA_BASE_URL = "http://localhost:11434"
- Pull a model in Ollama, e.g.:
    ollama pull llama3.1

Usage:
- Choose a model as "ollama:<model_name>", e.g. "ollama:llama3.1"
- api_key is optional for Ollama; LiteLLM will route the request.
"""

# Configure Ollama endpoint (if not set already)
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")

full_model_name = os.environ.get("LITELLM_OLLAMA_MODEL", "ollama:gemma3:4b")

# Create a synchronous client via factory; this will use LiteLLM client for 'ollama:*'
client = get_async_client(full_model_name=full_model_name)

resp = asyncio.run(
    client.create(
        messages=[Content(role="user", parts=[Part(text="Write a haiku about autumn.")])]
    )
)

print("Model:", full_model_name)
print("Text:\n", resp.text)
if resp.usage_metadata:
    print("Usage -> prompt:", resp.usage_metadata.prompt_token_count,
          "completion:", resp.usage_metadata.candidates_token_count,
          "total:", resp.usage_metadata.total_token_count)
