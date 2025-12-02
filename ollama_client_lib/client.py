"""
Ollama Cloud Client Library - Object-Oriented Version
Allows instantiating multiple clients with different configurations.
"""
import os
import logging
import json
from typing import Optional, List, AsyncIterator, Dict, Any
import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with Ollama Cloud.
    
    Allows creating multiple instances with different configurations.
    
    Example usage:
        # Basic client with configuration from environment variables
        client = OllamaClient()
        
        # Client with custom configuration
        client = OllamaClient(
            api_key="your_api_key",
            base_url="https://ollama.com",
            default_model="llama2",
            timeout=120.0
        )
        
        # Generate response
        response = await client.generate_response(
            prompt="What is Python?",
            context=["Python is a programming language..."]
        )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: float = 120.0,
        use_http2: bool = False,
        max_keepalive_connections: int = 10,
        max_connections: int = 20
    ):
        """
        Initialize the Ollama client.
        
        Args:
            api_key: Ollama Cloud API key. If not provided, uses OLLAMA_API_KEY from environment.
            base_url: Base URL for Ollama Cloud. Default: https://ollama.com
            default_model: Default model to use. If not provided, uses OLLAMA_MODEL from environment or None.
            timeout: Timeout for requests in seconds. Default: 120.0
            use_http2: Whether to use HTTP/2. Default: False (may cause DNS issues)
            max_keepalive_connections: Maximum keep-alive connections in the pool
            max_connections: Maximum total connections in the pool
        """
        # Configuration with default values from environment variables
        self.api_key = api_key or os.getenv("OLLAMA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Provide it in the constructor or configure "
                "OLLAMA_API_KEY in the environment. Get one at: https://ollama.com/settings/keys"
            )
        
        self.base_url = base_url or os.getenv("OLLAMA_URL", "https://ollama.com")
        self.default_model = default_model or os.getenv("OLLAMA_MODEL")
        self.timeout = timeout
        self.use_http2 = use_http2
        self.max_keepalive_connections = max_keepalive_connections
        self.max_connections = max_connections
        
        # HTTP client (created on demand)
        self._http_client: Optional[httpx.AsyncClient] = None
        
        logger.info(
            f"OllamaClient initialized: base_url={self.base_url}, "
            f"model={self.default_model}, timeout={self.timeout}s"
        )
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create the reusable HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            # Close previous client if it exists and is closed
            if self._http_client is not None and self._http_client.is_closed:
                try:
                    await self._http_client.aclose()
                except Exception:
                    pass
                self._http_client = None
            
            # Headers for authentication with Ollama Cloud
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            self._http_client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_keepalive_connections=self.max_keepalive_connections,
                    max_connections=self.max_connections
                ),
                http2=self.use_http2,
                headers=headers,
                follow_redirects=True
            )
        return self._http_client
    
    async def close(self):
        """Close the HTTP client and release resources."""
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None
    
    async def __aenter__(self):
        """Context manager: enter."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager: exit, closes the client."""
        await self.close()
    
    def _build_rag_prompt(
        self,
        user_question: str,
        context_chunks: Optional[List[str]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build the complete prompt for RAG with context and system instructions.
        
        This is a generic implementation. Override this method or provide a custom
        system_prompt to customize the prompt structure.
        
        Args:
            user_question: User's question
            context_chunks: List of relevant text chunks
            system_prompt: System instructions (optional, uses generic default if not provided)
        
        Returns:
            Complete formatted prompt
        """
        # Generic system prompt (can be overridden)
        default_system = """You are a helpful assistant that answers questions using ONLY the provided documents.

Instructions:
- Use natural language, avoid technical jargon.
- Be direct and to the point.
- If you don't find the information in the documents, explicitly state that you didn't find it.
- DO NOT invent, DO NOT complete with external knowledge."""
        
        system = system_prompt or default_system
        
        # Build the prompt
        prompt_parts = []
        prompt_parts.append(f"System: {system}\n")
        
        if context_chunks:
            prompt_parts.append("Documents:")
            for i, chunk in enumerate(context_chunks, 1):
                prompt_parts.append(f"[{i}] {chunk}")
        
        prompt_parts.append(f"\nQuestion: {user_question}\nAnswer:")
        
        full_prompt = "\n".join(prompt_parts)
        logger.debug(f"Prompt built: {len(full_prompt)} characters (~{len(full_prompt) // 4} tokens)")
        
        return full_prompt
    
    async def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        num_ctx: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response using Ollama Cloud.
        
        Args:
            prompt: User's question/prompt
            model: Model name to use (uses self.default_model if not provided, must be set)
            context: List of relevant text chunks for RAG context
            system_prompt: System prompt to guide behavior (optional)
            temperature: Controls creativity (0.0-1.0). Lower = more deterministic
            max_tokens: Maximum number of tokens to generate
            num_ctx: Context window size (default: 4096)
            options: Additional model options to pass directly to Ollama API
        
        Returns:
            Generated response from the model
        
        Raises:
            ValueError: If model is not specified
            Exception: If there's an error communicating with Ollama Cloud
        """
        model = model or self.default_model
        if not model:
            raise ValueError(
                "Model must be specified. Provide it as a parameter or set default_model "
                "in the constructor or OLLAMA_MODEL environment variable."
            )
        
        # Ollama Cloud models require -cloud suffix for API access
        # Add it automatically if not present
        if not model.endswith("-cloud"):
            model = f"{model}-cloud"
        
        # Build messages for Ollama Cloud API (uses /api/chat endpoint)
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        elif context:
            # If context provided but no system prompt, use default system prompt
            default_system = "You are a helpful assistant that answers questions using ONLY the provided documents."
            messages.append({
                "role": "system",
                "content": default_system
            })
        
        # Add context as system message or include in user message
        if context:
            context_text = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context)])
            if not system_prompt:
                # Append context to system message
                messages[-1]["content"] += f"\n\nDocuments:\n{context_text}"
            else:
                # Add context as separate system message
                messages.append({
                    "role": "system",
                    "content": f"Documents:\n{context_text}"
                })
        
        # Add user prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Build options
        request_options: Dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": num_ctx or 4096,
        }
        
        if max_tokens:
            request_options["num_predict"] = max_tokens
        
        # Merge with custom options if provided
        if options:
            request_options.update(options)
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": request_options
        }
        
        client = await self._get_http_client()
        chat_url = f"{self.base_url}/api/chat"
        logger.info(f"Sending request to Ollama Cloud (model: {model}, URL: {chat_url})")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:
            response = await client.post(
                chat_url,
                json=payload,
                timeout=self.timeout
            )
            # Log response status for debugging
            logger.debug(f"Response status: {response.status_code}")
            if response.status_code != 200:
                logger.debug(f"Response text: {response.text}")
            response.raise_for_status()
            
            result = response.json()
            # Ollama Cloud API returns response in message.content
            generated_text = result.get("message", {}).get("content", "")
            
            if not generated_text:
                raise Exception("The model returned an empty response")
            
            logger.info(f"Response generated by Ollama (model: {model}, length: {len(generated_text)} characters)")
            return generated_text.strip()
            
        except (httpx.ConnectError, httpx.NetworkError) as e:
            # Recreate client and retry
            logger.warning(f"Connection error: {e}. Recreating HTTP client...")
            await self.close()
            client = await self._get_http_client()
            
            try:
                response = await client.post(
                    chat_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                generated_text = result.get("message", {}).get("content", "")
                
                if not generated_text:
                    raise Exception("The model returned an empty response")
                
                return generated_text.strip()
            except Exception as retry_error:
                logger.error(f"Error on retry: {retry_error}")
                raise Exception(f"Connection error with Ollama Cloud: {str(e)}")
        
        except httpx.TimeoutException:
            raise Exception(f"The model took too long to respond (more than {self.timeout}s)")
        
        except httpx.HTTPStatusError as e:
            error_text = e.response.text if hasattr(e.response, 'text') else str(e.response)
            raise Exception(f"Error communicating with the model (HTTP {e.response.status_code}): {error_text}")
    
    async def generate_response_streaming(
        self,
        prompt: str,
        model: Optional[str] = None,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        num_ctx: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """
        Generate a response using Ollama Cloud with streaming.
        
        Args:
            prompt: User's question/prompt
            model: Model name to use (uses self.default_model if not provided, must be set)
            context: List of relevant text chunks
            system_prompt: System prompt (optional)
            temperature: Controls creativity (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            num_ctx: Context window size (default: 4096)
            options: Additional model options to pass directly to Ollama API
        
        Yields:
            Text chunks generated incrementally
        
        Raises:
            ValueError: If model is not specified
            Exception: If there's an error communicating with Ollama Cloud
        """
        model = model or self.default_model
        if not model:
            raise ValueError(
                "Model must be specified. Provide it as a parameter or set default_model "
                "in the constructor or OLLAMA_MODEL environment variable."
            )
        
        # Ollama Cloud models require -cloud suffix for API access
        # Add it automatically if not present
        if not model.endswith("-cloud"):
            model = f"{model}-cloud"
        
        # Build messages for Ollama Cloud API (uses /api/chat endpoint)
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        elif context:
            # If context provided but no system prompt, use default system prompt
            default_system = "You are a helpful assistant that answers questions using ONLY the provided documents."
            messages.append({
                "role": "system",
                "content": default_system
            })
        
        # Add context as system message or include in user message
        if context:
            context_text = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context)])
            if not system_prompt:
                # Append context to system message
                messages[-1]["content"] += f"\n\nDocuments:\n{context_text}"
            else:
                # Add context as separate system message
                messages.append({
                    "role": "system",
                    "content": f"Documents:\n{context_text}"
                })
        
        # Add user prompt
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Build options
        request_options: Dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": num_ctx or 4096,
        }
        
        if max_tokens:
            request_options["num_predict"] = max_tokens
        
        # Merge with custom options if provided
        if options:
            request_options.update(options)
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": request_options
        }
        
        client = await self._get_http_client()
        chat_url = f"{self.base_url}/api/chat"
        logger.info(f"Sending streaming request to Ollama Cloud (model: {model})")
        
        async with client.stream(
            "POST",
            chat_url,
            json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        chunk_data = json.loads(line)
                        # Ollama Cloud API returns content in message.content
                        if "message" in chunk_data and "content" in chunk_data["message"]:
                            yield chunk_data["message"]["content"]
                        if chunk_data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
    
    async def check_model_available(self, model: Optional[str] = None) -> bool:
        """
        Check if a model is available in Ollama Cloud.
        
        Args:
            model: Model name to check (uses self.default_model if not provided)
        
        Returns:
            True if the model is available, False otherwise
        """
        model = model or self.default_model
        if not model:
            logger.warning("No model specified for availability check")
            return False
        
        try:
            client = await self._get_http_client()
            logger.debug(f"Checking model {model} availability in Ollama Cloud")
            
            try:
                response = await client.get(f"{self.base_url}/api/tags", timeout=30.0)
                response.raise_for_status()
            except (httpx.ConnectError, httpx.ConnectTimeout, OSError) as conn_error:
                logger.error(f"Connection error while checking model: {conn_error}")
                return False
            except httpx.HTTPStatusError as http_error:
                logger.error(f"HTTP error: {http_error.response.status_code}")
                return False
            
            models_data = response.json()
            available_models = [m.get("name", "") for m in models_data.get("models", [])]
            
            model_base = model.split(":")[0]
            is_available = (
                model in available_models or
                any(m.startswith(model_base + ":") for m in available_models)
            )
            
            if is_available:
                logger.info(f"Model {model} is available")
            else:
                logger.warning(f"Model {model} is NOT available. Available models: {available_models}")
            
            return is_available
            
        except Exception as e:
            logger.error(f"Error checking model: {e}")
            return False
    
    async def list_available_models(self) -> List[str]:
        """
        List all available models in Ollama Cloud.
        
        Returns:
            List of available model names
        """
        try:
            client = await self._get_http_client()
            response = await client.get(f"{self.base_url}/api/tags", timeout=10.0)
            response.raise_for_status()
            
            models_data = response.json()
            models = [m.get("name", "") for m in models_data.get("models", [])]
            
            logger.info(f"Available models: {models}")
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

