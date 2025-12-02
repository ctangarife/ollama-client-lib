"""
Streaming example

This example shows how to use streaming responses.
"""
import asyncio
import os
from ollama_client_lib import OllamaClient


async def main():
    """Example: Streaming responses."""
    print("=" * 60)
    print("EXAMPLE 3: Streaming response")
    print("=" * 60)
    
    client = OllamaClient(
        api_key=os.getenv("OLLAMA_API_KEY"),
        default_model="llama2"
    )
    
    try:
        print("\nResponse (streaming): ", end="", flush=True)
        async for chunk in client.generate_response_streaming(
            prompt="Tell me a short story about a robot"
        ):
            print(chunk, end="", flush=True)
        print("\n")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

