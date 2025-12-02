"""
Multiple clients example

This example shows how to use multiple client instances with different configurations.
"""
import asyncio
import os
from ollama_client_lib import OllamaClient


async def main():
    """Example: Multiple clients with different configurations."""
    print("=" * 60)
    print("EXAMPLE 2: Multiple clients with different configurations")
    print("=" * 60)
    
    api_key = os.getenv("OLLAMA_API_KEY")
    
    # Fast client for quick responses
    client_fast = OllamaClient(
        api_key=api_key,
        default_model="llama2",  # Smaller/faster model
        timeout=60.0
    )
    
    # Powerful client for complex tasks
    client_powerful = OllamaClient(
        api_key=api_key,
        default_model="llama2:70b",  # Larger model
        timeout=300.0
    )
    
    try:
        # Use fast client for simple question
        response1 = await client_fast.generate_response(
            prompt="What is 2+2?"
        )
        print(f"\nFast response: {response1[:100]}...")
        
        # Use powerful client for complex question
        response2 = await client_powerful.generate_response(
            prompt="Explain quantum entanglement in simple terms"
        )
        print(f"\nDetailed response: {response2[:100]}...\n")
    finally:
        await client_fast.close()
        await client_powerful.close()


if __name__ == "__main__":
    asyncio.run(main())

