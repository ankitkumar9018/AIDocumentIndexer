#!/usr/bin/env python3
"""
Check detected embedding dimension based on environment configuration.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Import the dimension detection function
from backend.db.models import get_embedding_dimension, EMBEDDING_DIMENSION

def main():
    """Check and display embedding dimension configuration."""

    provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    model = os.getenv("DEFAULT_EMBEDDING_MODEL", "not set")
    ollama_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "not set")

    detected_dim = get_embedding_dimension()
    cached_dim = EMBEDDING_DIMENSION

    print("=" * 60)
    print("Embedding Dimension Configuration")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"Default Embedding Model: {model}")
    print(f"Ollama Embedding Model: {ollama_model}")
    print("-" * 60)
    print(f"Detected Dimension: {detected_dim}D")
    print(f"Cached Dimension: {cached_dim}D")
    print("=" * 60)

    # Provider-specific guidance
    if provider.lower() == "openai":
        print("\n✅ OpenAI configuration detected")
        print("   - Expected dimension: 1536D")
        print("   - Compatible models: text-embedding-3-small, text-embedding-ada-002")
    elif provider.lower() == "ollama":
        print("\n✅ Ollama configuration detected")
        print("   - Expected dimension: 768D")
        print("   - Compatible models: nomic-embed-text, mxbai-embed-large")
    elif provider.lower() == "huggingface":
        print("\n✅ HuggingFace configuration detected")
        print("   - Expected dimension: 768D (most common)")
        print("   - Compatible models: sentence-transformers/all-MiniLM-L6-v2")
    elif provider.lower() == "cohere":
        print("\n✅ Cohere configuration detected")
        print("   - Expected dimension: 1024D")
        print("   - Compatible models: embed-english-v3.0")
    else:
        print(f"\n⚠️  Unknown provider: {provider}")
        print("   - Using safe default: 768D")

    print("\n📝 To change the dimension:")
    print("   1. Update DEFAULT_LLM_PROVIDER in .env file")
    print("   2. Restart the application")
    print("   3. Database schema will automatically use the new dimension")
    print("\n⚠️  WARNING: Switching providers requires re-indexing:")
    print("   - Run migration script to clear old embeddings")
    print("   - Re-index all documents")
    print("   - Run entity embedding backfill")

    return detected_dim

if __name__ == "__main__":
    main()
