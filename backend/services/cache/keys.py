"""
AIDocumentIndexer - Cache Key Generation
==========================================

Standardized cache key generation utilities to ensure consistency
across all cache implementations.
"""

import hashlib
from typing import Any, List, Optional, Union


def hash_content(content: str, algorithm: str = "sha256", length: int = 32) -> str:
    """
    Generate a hash of content for cache key generation.

    Args:
        content: Content to hash
        algorithm: Hash algorithm (sha256, md5)
        length: Length of hash to return (max 64 for sha256)

    Returns:
        Hexadecimal hash string
    """
    if algorithm == "md5":
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:length]
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:length]


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent hashing.

    Args:
        text: Text to normalize

    Returns:
        Normalized text (lowercase, stripped)
    """
    return text.strip().lower()


class CacheKeyGenerator:
    """
    Standardized cache key generator.

    Provides consistent key generation patterns used across all caches:
    - Content-based keys (for embedding deduplication)
    - Parameter-based keys (for search result caching)
    - Composite keys (combining multiple factors)

    Usage:
        keygen = CacheKeyGenerator(prefix="embed")

        # Content-based key
        key = keygen.content_key("Hello world")

        # Parameter-based key
        key = keygen.params_key(query="hello", model="gpt-4", temp=0.0)

        # Composite key
        key = keygen.composite_key(
            content="query text",
            params={"model": "gpt-4", "tier": 2},
        )
    """

    def __init__(
        self,
        prefix: str = "",
        hash_algorithm: str = "sha256",
        hash_length: int = 32,
        normalize: bool = True,
    ):
        """
        Initialize key generator.

        Args:
            prefix: Optional prefix for all keys
            hash_algorithm: Hash algorithm (sha256, md5)
            hash_length: Length of generated hashes
            normalize: Whether to normalize text before hashing
        """
        self.prefix = prefix
        self.hash_algorithm = hash_algorithm
        self.hash_length = hash_length
        self.normalize = normalize

    def content_key(self, content: str) -> str:
        """
        Generate a key based on content hash.

        Used for:
        - Embedding deduplication
        - Response caching by prompt

        Args:
            content: Content to hash

        Returns:
            Content-based cache key
        """
        if self.normalize:
            content = normalize_text(content)

        content_hash = hash_content(content, self.hash_algorithm, self.hash_length)

        if self.prefix:
            return f"{self.prefix}:{content_hash}"
        return content_hash

    def params_key(self, **params: Any) -> str:
        """
        Generate a key based on parameters.

        Used for:
        - Search result caching
        - API response caching

        Args:
            **params: Key-value parameters to include in key

        Returns:
            Parameter-based cache key
        """
        # Sort parameters for consistent ordering
        sorted_parts = []
        for key in sorted(params.keys()):
            value = params[key]
            if isinstance(value, list):
                value = ",".join(sorted(str(v) for v in value))
            elif value is None:
                value = "null"
            sorted_parts.append(f"{key}:{value}")

        params_str = "|".join(sorted_parts)
        params_hash = hash_content(params_str, self.hash_algorithm, self.hash_length)

        if self.prefix:
            return f"{self.prefix}:{params_hash}"
        return params_hash

    def composite_key(
        self,
        content: Optional[str] = None,
        params: Optional[dict] = None,
        extra: Optional[str] = None,
    ) -> str:
        """
        Generate a composite key from content and parameters.

        Used for:
        - Search caching (query + search params)
        - Response caching (prompt + model + temperature)

        Args:
            content: Optional content to hash
            params: Optional parameters to include
            extra: Optional extra string to append

        Returns:
            Composite cache key
        """
        parts = []

        if content:
            if self.normalize:
                content = normalize_text(content)
            parts.append(f"c:{content}")

        if params:
            sorted_parts = []
            for key in sorted(params.keys()):
                value = params[key]
                if isinstance(value, list):
                    value = ",".join(sorted(str(v) for v in value))
                elif value is None:
                    value = "null"
                sorted_parts.append(f"{key}={value}")
            parts.append("p:" + "&".join(sorted_parts))

        if extra:
            parts.append(f"x:{extra}")

        combined = "||".join(parts)
        combined_hash = hash_content(combined, self.hash_algorithm, self.hash_length)

        if self.prefix:
            return f"{self.prefix}:{combined_hash}"
        return combined_hash

    def prompt_key(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate a key for LLM response caching.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model: Model identifier
            temperature: Temperature setting

        Returns:
            Cache key for LLM response
        """
        combined = prompt
        if system_prompt:
            combined = f"{system_prompt}|||{prompt}"

        params = {}
        if model:
            params["model"] = model
        if temperature is not None:
            params["temp"] = f"{temperature:.2f}"

        return self.composite_key(content=combined, params=params if params else None)

    def search_key(
        self,
        query: str,
        search_type: str,
        access_tier: int,
        top_k: int,
        document_ids: Optional[List[str]] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        collection: Optional[str] = None,
    ) -> str:
        """
        Generate a key for search result caching.

        Args:
            query: Search query
            search_type: Type of search (vector, keyword, hybrid)
            access_tier: User's access tier level
            top_k: Number of results
            document_ids: Optional document filter
            vector_weight: Vector weight for hybrid search
            keyword_weight: Keyword weight for hybrid search
            collection: Optional collection filter

        Returns:
            Cache key for search results
        """
        params = {
            "type": search_type,
            "tier": access_tier,
            "k": top_k,
        }

        if document_ids:
            params["docs"] = document_ids

        if vector_weight is not None:
            params["vw"] = f"{vector_weight:.2f}"

        if keyword_weight is not None:
            params["kw"] = f"{keyword_weight:.2f}"

        if collection:
            params["col"] = collection

        return self.composite_key(content=query, params=params)

    def embedding_key(
        self,
        text: str,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> str:
        """
        Generate a key for embedding caching.

        Args:
            text: Text to embed
            model: Embedding model
            dimensions: Embedding dimensions

        Returns:
            Cache key for embedding
        """
        params = {}
        if model:
            params["model"] = model
        if dimensions:
            params["dims"] = dimensions

        return self.composite_key(content=text, params=params if params else None)


# Pre-configured key generators
embedding_keygen = CacheKeyGenerator(prefix="embed", normalize=True)
search_keygen = CacheKeyGenerator(prefix="search", normalize=True)
response_keygen = CacheKeyGenerator(prefix="response", normalize=False)
