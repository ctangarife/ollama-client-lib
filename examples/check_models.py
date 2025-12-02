"""
Check models example

This example shows how to check model availability and list available models.
"""
import asyncio
import os
from ollama_client_lib import OllamaClient


async def main():
    """Example: Check model availability."""
    print("=" * 60)
    print("EXAMPLE 7: Check model availability")
    print("=" * 60)
    
    client = OllamaClient(
        api_key=os.getenv("OLLAMA_API_KEY")
    )
    
    try:
        # List all available models
        models = await client.list_available_models()
        print(f"\nAvailable models: {models}\n")
        
        # Check if a specific model is available
        model_to_check = "llama2"
        is_available = await client.check_model_available(model_to_check)
        print(f"Is {model_to_check} available? {is_available}\n")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

