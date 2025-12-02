"""
Unit tests for OllamaClient
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from ollama_client_lib import OllamaClient


class TestOllamaClient:
    """Test cases for OllamaClient class."""
    
    def test_init_with_api_key(self):
        """Test client initialization with API key."""
        client = OllamaClient(api_key="test_key", default_model="test_model")
        assert client.api_key == "test_key"
        assert client.default_model == "test_model"
        assert client.base_url == "https://ollama.com"
        assert client.timeout == 120.0
    
    def test_init_without_api_key_raises(self):
        """Test that initialization without API key raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                OllamaClient()
    
    def test_init_with_env_vars(self):
        """Test client initialization with environment variables."""
        with patch.dict("os.environ", {
            "OLLAMA_API_KEY": "env_key",
            "OLLAMA_URL": "https://custom.url",
            "OLLAMA_MODEL": "env_model"
        }):
            client = OllamaClient()
            assert client.api_key == "env_key"
            assert client.base_url == "https://custom.url"
            assert client.default_model == "env_model"
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client as context manager."""
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test_key"}):
            async with OllamaClient(default_model="test") as client:
                assert client.api_key == "test_key"
            # Client should be closed after context exit
    
    @pytest.mark.asyncio
    async def test_close(self):
        """Test client close method."""
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test_key"}):
            client = OllamaClient(default_model="test")
            # Create a mock HTTP client
            mock_client = AsyncMock()
            client._http_client = mock_client
            
            await client.close()
            mock_client.aclose.assert_called_once()
            assert client._http_client is None
    
    @pytest.mark.asyncio
    async def test_generate_response_no_model_raises(self):
        """Test that generate_response raises ValueError if no model is specified."""
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test_key"}, clear=True):
            client = OllamaClient()
            # Clear default_model to ensure it's None
            client.default_model = None
            with pytest.raises(ValueError, match="Model must be specified"):
                await client.generate_response(prompt="test")
    
    @pytest.mark.asyncio
    async def test_build_rag_prompt(self):
        """Test RAG prompt building."""
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test_key"}):
            client = OllamaClient(default_model="test")
            
            prompt = client._build_rag_prompt(
                user_question="What is Python?",
                context_chunks=["Python is a language"],
                system_prompt="You are helpful"
            )
            
            assert "What is Python?" in prompt
            assert "Python is a language" in prompt
            assert "You are helpful" in prompt
    
    @pytest.mark.asyncio
    async def test_build_rag_prompt_no_context(self):
        """Test RAG prompt building without context."""
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test_key"}):
            client = OllamaClient(default_model="test")
            
            prompt = client._build_rag_prompt(
                user_question="What is Python?"
            )
            
            assert "What is Python?" in prompt
            assert "Question:" in prompt
    
    @pytest.mark.asyncio
    async def test_list_available_models_error_handling(self):
        """Test list_available_models error handling."""
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test_key"}):
            client = OllamaClient(default_model="test")
            
            # Mock HTTP client to raise exception
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=Exception("Connection error"))
            
            # Mock _get_http_client to return our mock
            client._get_http_client = AsyncMock(return_value=mock_client)
            
            models = await client.list_available_models()
            assert models == []

