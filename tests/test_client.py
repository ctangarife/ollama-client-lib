"""
Unit tests for OllamaClient
"""
import pytest
import json
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
    
    def test_ollama_client_initializes_with_api_key_and_default_model(self):
        """Test OllamaClient initializes correctly with API key and default model from constructor."""
        client = OllamaClient(api_key="my_api_key", default_model="llama2")
        
        assert client.api_key == "my_api_key"
        assert client.default_model == "llama2"
        assert client.base_url == "https://ollama.com"
        assert client.timeout == 120.0
        assert client.use_http2 is False
        assert client._http_client is None
    
    def test_ollama_client_raises_value_error_without_api_key(self):
        """Test OllamaClient raises ValueError if no API key is provided during initialization."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                OllamaClient()
            
            assert "API key required" in str(exc_info.value)
            assert "OLLAMA_API_KEY" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_generate_response_returns_valid_response(self):
        """Test OllamaClient.generate_response returns a valid response for a given prompt and model."""
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test_key"}):
            client = OllamaClient(default_model="llama2")
            
            # Mock HTTP response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "message": {
                    "content": "Python is a high-level programming language."
                }
            }
            mock_response.raise_for_status = MagicMock()
            
            # Mock HTTP client
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_response)
            
            # Mock _get_http_client method
            client._get_http_client = AsyncMock(return_value=mock_http_client)
            
            response = await client.generate_response(
                prompt="What is Python?",
                model="llama2"
            )
            
            assert response == "Python is a high-level programming language."
            mock_http_client.post.assert_called_once()
            call_args = mock_http_client.post.call_args
            assert "llama2-cloud" in call_args.kwargs["json"]["model"]
    
    @pytest.mark.asyncio
    async def test_generate_response_streaming_yields_chunks(self):
        """Test OllamaClient.generate_response_streaming yields chunks of text incrementally."""
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test_key"}):
            client = OllamaClient(default_model="llama2")
            
            # Mock streaming response chunks
            mock_chunks = [
                json.dumps({"message": {"content": "Python"}, "done": False}),
                json.dumps({"message": {"content": " is"}, "done": False}),
                json.dumps({"message": {"content": " a language"}, "done": False}),
                json.dumps({"message": {"content": "."}, "done": True}),
            ]
            
            # Create async generator for mock
            async def mock_aiter_lines():
                for chunk in mock_chunks:
                    yield chunk
            
            # Mock streaming context manager
            mock_stream_response = AsyncMock()
            mock_stream_response.raise_for_status = MagicMock()
            mock_stream_response.aiter_lines = mock_aiter_lines
            
            mock_stream_context = AsyncMock()
            mock_stream_context.__aenter__ = AsyncMock(return_value=mock_stream_response)
            mock_stream_context.__aexit__ = AsyncMock(return_value=None)
            
            # Mock HTTP client
            mock_http_client = AsyncMock()
            mock_http_client.stream = MagicMock(return_value=mock_stream_context)
            
            # Mock _get_http_client method
            client._get_http_client = AsyncMock(return_value=mock_http_client)
            
            # Collect streaming chunks
            chunks = []
            async for chunk in client.generate_response_streaming(
                prompt="What is Python?",
                model="llama2"
            ):
                chunks.append(chunk)
            
            assert chunks == ["Python", " is", " a language", "."]
            assert len(chunks) == 4
    
    def test_build_rag_prompt_formats_correctly(self):
        """Test OllamaClient._build_rag_prompt correctly formats the prompt with user question, context, and system prompt."""
        with patch.dict("os.environ", {"OLLAMA_API_KEY": "test_key"}):
            client = OllamaClient(default_model="llama2")
            
            user_question = "What is machine learning?"
            context_chunks = [
                "Machine learning is a subset of AI.",
                "It uses algorithms to learn from data."
            ]
            system_prompt = "You are an AI expert."
            
            prompt = client._build_rag_prompt(
                user_question=user_question,
                context_chunks=context_chunks,
                system_prompt=system_prompt
            )
            
            # Verify all components are in the prompt
            assert "You are an AI expert." in prompt
            assert "Machine learning is a subset of AI." in prompt
            assert "It uses algorithms to learn from data." in prompt
            assert "What is machine learning?" in prompt
            
            # Verify structure
            assert "System:" in prompt
            assert "Documents:" in prompt
            assert "Question:" in prompt
            assert "Answer:" in prompt
            
            # Verify context is numbered
            assert "[1]" in prompt
            assert "[2]" in prompt

