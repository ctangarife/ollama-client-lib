"""
RAG (Retrieval Augmented Generation) example

This example shows how to use RAG with context chunks.
"""
import asyncio
import os
from ollama_client_lib import OllamaClient


async def main():
    """Example: RAG with context."""
    print("=" * 60)
    print("EXAMPLE 4: RAG with context")
    print("=" * 60)
    
    client = OllamaClient(
        api_key=os.getenv("OLLAMA_API_KEY"),
        default_model="llama2"
    )
    
    try:
        # Context chunks from your knowledge base
        context = [
            "Python is a high-level, interpreted programming language created by Guido van Rossum.",
            "Python emphasizes code readability and simplicity, making it popular for beginners.",
            "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        ]
        
        response = await client.generate_response(
            prompt="Who created Python and what are its main features?",
            context=context,
            temperature=0.5,  # More deterministic
            max_tokens=200
        )
        
        print(f"\nResponse: {response}\n")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

