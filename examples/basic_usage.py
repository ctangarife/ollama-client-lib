"""
Basic usage example of ollama_client_lib

This example shows how to use the library with environment variables.
"""
import asyncio
import os
from ollama_client_lib import OllamaClient


async def main():
    """Basic example: Client with configuration from environment variables."""
    print("=" * 60)
    print("EXAMPLE 1: Basic client (configuration from environment)")
    print("=" * 60)
    
    # Make sure OLLAMA_API_KEY is set in your environment
    # You can also pass it directly: OllamaClient(api_key="your_key")
    
    client = OllamaClient(
        api_key=os.getenv("OLLAMA_API_KEY"),
        default_model="llama2"  # Replace with your preferred model
    )
    
    try:
        response = await client.generate_response(
            prompt="What is Python programming language?"
        )
        print(f"\nResponse: {response}\n")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

