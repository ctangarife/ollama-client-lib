"""
Integration tests for OllamaClient

These tests require a real API key and connection to Ollama Cloud.
They are skipped by default unless OLLAMA_API_KEY is set.
"""
import pytest
import os
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, use system env vars only

from ollama_client_lib import OllamaClient


# Skip integration tests if API key is not available
pytestmark = pytest.mark.skipif(
    not os.getenv("OLLAMA_API_KEY"),
    reason="Integration tests require OLLAMA_API_KEY environment variable (set in .env or system)"
)


@pytest.mark.integration
class TestOllamaClientIntegration:
    """Integration tests for OllamaClient."""
    
    @pytest.mark.asyncio
    async def test_list_available_models(self):
        """Test listing available models."""
        async with OllamaClient() as client:
            models = await client.list_available_models()
            assert isinstance(models, list)
            assert len(models) > 0
    
    @pytest.mark.asyncio
    async def test_check_model_available(self):
        """Test checking model availability."""
        async with OllamaClient() as client:
            # First, get available models
            models = await client.list_available_models()
            if models:
                # Check if first model is available
                is_available = await client.check_model_available(models[0])
                assert is_available is True
    
    @pytest.mark.asyncio
    async def test_generate_response_basic(self):
        """Test basic response generation."""
        # Use a model that exists in Ollama Cloud (kimi-k2:1t)
        async with OllamaClient(default_model="kimi-k2:1t") as client:
            response = await client.generate_response(
                prompt="Say hello in one word",
                max_tokens=10
            )
            assert isinstance(response, str)
            assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_with_context(self):
        """Test response generation with context."""
        # Use a model that exists in Ollama Cloud (kimi-k2:1t)
        async with OllamaClient(default_model="kimi-k2:1t") as client:
            response = await client.generate_response(
                prompt="What is the capital?",
                context=["France is a country. The capital of France is Paris."],
                max_tokens=20
            )
            assert isinstance(response, str)
            assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_generate_response_streaming(self):
        """Test streaming response generation."""
        # Use a model that exists in Ollama Cloud (kimi-k2:1t)
        async with OllamaClient(default_model="kimi-k2:1t") as client:
            chunks = []
            async for chunk in client.generate_response_streaming(
                prompt="Count to 3",
                max_tokens=20
            ):
                chunks.append(chunk)
            
            assert len(chunks) > 0
            full_response = "".join(chunks)
            assert len(full_response) > 0

