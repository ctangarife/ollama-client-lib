# ollama-client-lib

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Object-oriented Python library for interacting with Ollama Cloud. Allows creating multiple client instances with different configurations, automatic resource management with context managers, and full support for RAG (Retrieval Augmented Generation).

## Features

- ✅ Multiple client instances with independent configurations
- ✅ Context manager for automatic resource management
- ✅ RAG support with context
- ✅ Streaming responses
- ✅ Robust error handling and automatic reconnection
- ✅ Reusable HTTP connection pool
- ✅ Model and prompt agnostic (no hardcoded defaults)

## Installation

### From PyPI (when published)

```bash
pip install ollama-client-lib
```

### From GitHub

```bash
pip install git+https://github.com/ctangarife/ollama-client-lib.git
```

### For local development

```bash
git clone https://github.com/ctangarife/ollama-client-lib.git
cd ollama-client-lib
pip install -e .
```

## Quick Start

```python
import asyncio
from ollama_client_lib import OllamaClient

async def main():
    async with OllamaClient(
        api_key="your_api_key",
        default_model="kimi-k2:1t"  # Model available in Ollama Cloud
    ) as client:
        response = await client.generate_response(
            prompt="What is Python?",
            context=["Python is a programming language..."]
        )
        print(response)

asyncio.run(main())
```

**Note**: This library is specifically designed for **Ollama Cloud** (requires API key), not for local Ollama installations. The `-cloud` suffix is automatically added to model names if not present.

## Configuration

The library supports configuration via environment variables or constructor parameters:

- `OLLAMA_API_KEY`: API key (required) - Get one at https://ollama.com/settings/keys
- `OLLAMA_URL`: Base URL (default: https://ollama.com)
- `OLLAMA_MODEL`: Default model (optional, must be specified per request if not set)

**Important**: 
- This library is for **Ollama Cloud** only (not local Ollama)
- Model names automatically get the `-cloud` suffix added if not present (e.g., `kimi-k2:1t` → `kimi-k2:1t-cloud`)
- Use `list_available_models()` to see which models are available in your account

### Setting up environment variables

1. Copy `env.example` to `.env`:
   ```bash
   cp env.example .env
   ```

2. Edit `.env` and add your Ollama Cloud API key:
   ```
   OLLAMA_API_KEY=your_actual_api_key_here
   ```

3. Get your API key at: https://ollama.com/settings/keys

**Note**: The `.env` file is already in `.gitignore` and won't be committed to version control.

## Usage Examples

### Basic Usage

```python
import asyncio
from ollama_client_lib import OllamaClient

async def main():
    async with OllamaClient(
        api_key="your_api_key",
        default_model="kimi-k2:1t"  # Use a model available in Ollama Cloud
    ) as client:
        response = await client.generate_response(
            prompt="Explain quantum computing"
        )
        print(response)

asyncio.run(main())
```

### Multiple Clients

```python
# Fast client
client_fast = OllamaClient(
    api_key="your_api_key",
    default_model="kimi-k2:1t",
    timeout=60.0
)

# Powerful client
client_powerful = OllamaClient(
    api_key="your_api_key",
    default_model="gpt-oss:120b",
    timeout=300.0
)
```

### RAG with Context

```python
import asyncio
from ollama_client_lib import OllamaClient

async def main():
    async with OllamaClient(
        api_key="your_api_key",
        default_model="kimi-k2:1t"
    ) as client:
        context = [
            "Python is a high-level programming language...",
            "It was created by Guido van Rossum...",
        ]
        
        response = await client.generate_response(
            prompt="Who created Python?",
            context=context
        )
        print(response)

asyncio.run(main())
```

### Streaming

```python
import asyncio
from ollama_client_lib import OllamaClient

async def main():
    async with OllamaClient(
        api_key="your_api_key",
        default_model="kimi-k2:1t"
    ) as client:
        async for chunk in client.generate_response_streaming(
            prompt="Tell me a story"
        ):
            print(chunk, end="", flush=True)

asyncio.run(main())
```

### Custom System Prompt

```python
import asyncio
from ollama_client_lib import OllamaClient

async def main():
    async with OllamaClient(
        api_key="your_api_key",
        default_model="kimi-k2:1t"
    ) as client:
        response = await client.generate_response(
            prompt="Explain this code",
            system_prompt="You are a code reviewer. Explain code clearly and concisely.",
            context=["def hello(): print('world')"]
        )
        print(response)

asyncio.run(main())
```

### Check Model Availability

```python
import asyncio
from ollama_client_lib import OllamaClient

async def main():
    async with OllamaClient(api_key="your_api_key") as client:
        # List all available models
        models = await client.list_available_models()
        print(f"Available models: {models}")
        
        # Check if a specific model is available
        available = await client.check_model_available("kimi-k2:1t")
        if available:
            print("Model is available!")

asyncio.run(main())
```

**Note**: Model names in the list don't include the `-cloud` suffix, but it's automatically added when making API calls.

## API Reference

For detailed API documentation, see [Ollama Cloud Documentation](https://docs.ollama.com/cloud).

### OllamaClient

#### Constructor

```python
OllamaClient(
    api_key: Optional[str] = None,          # API key (required, or set OLLAMA_API_KEY env var)
    base_url: Optional[str] = None,         # Base URL (default: https://ollama.com)
    default_model: Optional[str] = None,     # Default model name (without -cloud suffix)
    timeout: float = 120.0,                 # Request timeout in seconds
    use_http2: bool = False,                 # Enable HTTP/2 (may cause DNS issues)
    max_keepalive_connections: int = 10,    # Max keep-alive connections in pool
    max_connections: int = 20                # Max total connections in pool
)
```

#### Methods

##### `generate_response()`

Generate a complete response from the model.

```python
async def generate_response(
    prompt: str,                              # User prompt/question
    model: Optional[str] = None,              # Model name (uses default_model if not set)
    context: Optional[List[str]] = None,      # RAG context chunks
    system_prompt: Optional[str] = None,       # Custom system prompt
    temperature: float = 0.7,                 # Creativity (0.0-1.0, lower = more deterministic)
    max_tokens: Optional[int] = None,          # Maximum tokens to generate
    num_ctx: Optional[int] = None,            # Context window size (default: 4096)
    options: Optional[Dict[str, Any]] = None  # Additional model options
) -> str
```

##### `generate_response_streaming()`

Generate a streaming response (yields chunks incrementally).

```python
async def generate_response_streaming(
    prompt: str,
    model: Optional[str] = None,
    context: Optional[List[str]] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    num_ctx: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None
) -> AsyncIterator[str]
```

##### `check_model_available()`

Check if a model is available in Ollama Cloud.

```python
async def check_model_available(model: Optional[str] = None) -> bool
```

##### `list_available_models()`

List all models available in your Ollama Cloud account.

```python
async def list_available_models() -> List[str]
```

##### `close()`

Close the HTTP client and release resources. Automatically called when using context manager.

```python
async def close() -> None
```

## Examples

See the `examples/` directory for more complete examples:

- `basic_usage.py` - Basic client usage
- `multiple_clients.py` - Using multiple client instances
- `streaming.py` - Streaming responses
- `rag_example.py` - RAG with context
- `context_manager.py` - Using context manager
- `custom_prompt.py` - Custom system prompts
- `check_models.py` - Checking model availability

## Requirements

- Python 3.8+
- httpx >= 0.24.0
- python-dotenv (optional, for loading .env files)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Testing

Run the test suite:

```bash
# Unit tests only
pytest tests/test_client.py -v

# All tests (including integration tests - requires API key in .env)
pytest tests/ -v
```

Integration tests require a valid `OLLAMA_API_KEY` in your `.env` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a Pull Request

## Links

- **Repository**: https://github.com/ctangarife/ollama-client-lib
- **Issues**: https://github.com/ctangarife/ollama-client-lib/issues

