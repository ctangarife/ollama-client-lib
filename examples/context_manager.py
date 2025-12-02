"""
Context manager example

This example shows how to use the client as a context manager for automatic cleanup.
"""
import asyncio
import os
from ollama_client_lib import OllamaClient


async def main():
    """Example: Using the client as a context manager."""
    print("=" * 60)
    print("EXAMPLE 5: Context Manager (automatic cleanup)")
    print("=" * 60)
    
    # The client automatically closes when exiting the block
    async with OllamaClient(
        api_key=os.getenv("OLLAMA_API_KEY"),
        default_model="llama2"
    ) as client:
        response = await client.generate_response(
            prompt="What is async/await in Python?"
        )
        print(f"\nResponse: {response}\n")
    
    # Client is already closed here


if __name__ == "__main__":
    asyncio.run(main())

