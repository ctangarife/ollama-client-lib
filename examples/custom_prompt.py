"""
Custom system prompt example

This example shows how to customize the system prompt for specific use cases.
"""
import asyncio
import os
from ollama_client_lib import OllamaClient


async def main():
    """Example: Custom system prompt."""
    print("=" * 60)
    print("EXAMPLE 6: Custom system prompt")
    print("=" * 60)
    
    client = OllamaClient(
        api_key=os.getenv("OLLAMA_API_KEY"),
        default_model="llama2"
    )
    
    try:
        # Custom system prompt for code explanation
        code_explainer_prompt = """You are a code reviewer and teacher. 
Explain code clearly and concisely. Focus on:
- What the code does
- Key concepts used
- Potential improvements"""
        
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        
        response = await client.generate_response(
            prompt="Explain this code",
            system_prompt=code_explainer_prompt,
            context=[code]
        )
        
        print(f"\nResponse: {response}\n")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

