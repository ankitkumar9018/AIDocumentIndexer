"""
AIDocumentIndexer - Multilingual Cross-Lingual Search
======================================================

Enables search across language barriers:
1. Multilingual embeddings that map similar concepts in different languages
2. Query translation for enhanced keyword matching
3. Response translation to user's preferred language
4. Language detection for automatic handling

Supports seamless search where queries and documents can be in different languages.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


# Common languages supported
class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"
    UNKNOWN = "unknown"


# Language detection patterns (simple heuristics)
LANGUAGE_PATTERNS = {
    "en": ["the", "is", "are", "was", "have", "has", "been", "will", "would", "could"],
    "es": ["el", "la", "los", "las", "es", "son", "está", "están", "que", "de"],
    "fr": ["le", "la", "les", "est", "sont", "de", "du", "des", "que", "qui"],
    "de": ["der", "die", "das", "ist", "sind", "und", "für", "mit", "von", "zu"],
    "it": ["il", "la", "le", "è", "sono", "che", "di", "per", "con", "non"],
    "pt": ["o", "a", "os", "as", "é", "são", "de", "para", "com", "que"],
    "nl": ["de", "het", "een", "is", "zijn", "van", "voor", "met", "op", "te"],
}


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    detected_language: Language
    confidence: float
    alternative_languages: List[Tuple[Language, float]]  # (lang, confidence)


@dataclass
class TranslationResult:
    """Result of translation."""
    original_text: str
    translated_text: str
    source_language: Language
    target_language: Language
    confidence: float


@dataclass
class CrossLingualSearchResult:
    """Result from cross-lingual search."""
    document_id: str
    chunk_id: str
    content: str
    similarity_score: float
    source_language: Language
    translated_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultilingualSearchConfig:
    """Configuration for multilingual search."""
    # Multilingual embedding model
    embedding_model: str = "intfloat/multilingual-e5-large"
    # Alternative: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    # Search settings
    enable_query_translation: bool = True
    enable_response_translation: bool = False
    default_language: Language = Language.ENGLISH
    supported_languages: List[Language] = field(default_factory=lambda: [
        Language.ENGLISH, Language.SPANISH, Language.FRENCH,
        Language.GERMAN, Language.ITALIAN, Language.PORTUGUESE,
    ])

    # Translation settings
    translation_model: str = "gpt-4o-mini"  # For high quality translation
    max_translation_length: int = 5000


class LanguageDetector:
    """
    Detect language of text.

    Uses multiple strategies:
    1. Character set analysis
    2. Common word patterns
    3. Optional: External library (langdetect)
    """

    def __init__(self, use_external_detector: bool = True):
        self.use_external = use_external_detector
        self._langdetect_available = False

        if use_external_detector:
            try:
                from langdetect import detect, detect_langs
                self._langdetect_available = True
            except ImportError:
                logger.info("langdetect not available, using heuristic detection")

    def detect(self, text: str) -> LanguageDetectionResult:
        """
        Detect language of text.

        Args:
            text: Text to analyze

        Returns:
            LanguageDetectionResult with detected language
        """
        if not text or len(text.strip()) < 10:
            return LanguageDetectionResult(
                detected_language=Language.UNKNOWN,
                confidence=0.0,
                alternative_languages=[],
            )

        # Try external detector first
        if self._langdetect_available:
            try:
                from langdetect import detect_langs

                results = detect_langs(text)
                if results:
                    # Convert to our Language enum
                    top_lang = self._map_language_code(results[0].lang)
                    alternatives = [
                        (self._map_language_code(r.lang), r.prob)
                        for r in results[1:4]
                    ]

                    return LanguageDetectionResult(
                        detected_language=top_lang,
                        confidence=results[0].prob,
                        alternative_languages=alternatives,
                    )
            except Exception as e:
                logger.debug("External language detection failed", error=str(e))

        # Fallback to heuristic detection
        return self._heuristic_detect(text)

    def _heuristic_detect(self, text: str) -> LanguageDetectionResult:
        """Heuristic language detection using word patterns."""
        text_lower = text.lower()
        words = set(text_lower.split())

        scores: Dict[str, float] = {}

        for lang_code, patterns in LANGUAGE_PATTERNS.items():
            matches = sum(1 for p in patterns if p in words)
            scores[lang_code] = matches / len(patterns)

        if not scores or max(scores.values()) == 0:
            return LanguageDetectionResult(
                detected_language=Language.ENGLISH,  # Default
                confidence=0.3,
                alternative_languages=[],
            )

        # Sort by score
        sorted_langs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        top_lang = self._map_language_code(sorted_langs[0][0])
        top_score = sorted_langs[0][1]

        alternatives = [
            (self._map_language_code(code), score)
            for code, score in sorted_langs[1:4]
            if score > 0
        ]

        return LanguageDetectionResult(
            detected_language=top_lang,
            confidence=min(top_score * 2, 1.0),  # Scale up
            alternative_languages=alternatives,
        )

    def _map_language_code(self, code: str) -> Language:
        """Map language code to Language enum."""
        code_lower = code.lower()[:2]

        for lang in Language:
            if lang.value == code_lower:
                return lang

        return Language.UNKNOWN


class QueryTranslator:
    """
    Translate queries between languages.

    Uses LLM for high-quality translation that preserves
    search intent and technical terms.
    """

    def __init__(self, llm=None):
        """
        Initialize query translator.

        Args:
            llm: LangChain LLM for translation
        """
        self.llm = llm

    async def translate(
        self,
        text: str,
        source_language: Language,
        target_language: Language,
    ) -> TranslationResult:
        """
        Translate text between languages.

        Args:
            text: Text to translate
            source_language: Source language
            target_language: Target language

        Returns:
            TranslationResult
        """
        if source_language == target_language:
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_language,
                confidence=1.0,
            )

        if not self.llm:
            logger.warning("No LLM available for translation")
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_language,
                confidence=0.0,
            )

        prompt = f"""Translate the following {source_language.value} text to {target_language.value}.
Preserve technical terms, proper nouns, and search intent.
Return only the translation, nothing else.

Text: {text}

Translation:"""

        try:
            response = await self.llm.ainvoke(prompt)
            translated = response.content if hasattr(response, 'content') else str(response)

            return TranslationResult(
                original_text=text,
                translated_text=translated.strip(),
                source_language=source_language,
                target_language=target_language,
                confidence=0.85,  # Estimated confidence for LLM translation
            )
        except Exception as e:
            logger.warning("Translation failed", error=str(e))
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_language=source_language,
                target_language=target_language,
                confidence=0.0,
            )

    async def translate_for_search(
        self,
        query: str,
        target_languages: List[Language],
    ) -> Dict[Language, str]:
        """
        Translate query to multiple languages for cross-lingual search.

        Args:
            query: Search query
            target_languages: Languages to translate to

        Returns:
            Dict mapping language to translated query
        """
        # Detect source language
        detector = LanguageDetector()
        detection = detector.detect(query)
        source_lang = detection.detected_language

        translations = {source_lang: query}

        for lang in target_languages:
            if lang != source_lang and lang != Language.UNKNOWN:
                result = await self.translate(query, source_lang, lang)
                if result.confidence > 0.5:
                    translations[lang] = result.translated_text

        return translations


class MultilingualEmbeddingService:
    """
    Embedding service that works across languages.

    Uses multilingual embedding models that map semantically similar
    content to nearby vectors regardless of language.
    """

    # Multilingual models and their dimensions
    MULTILINGUAL_MODELS = {
        "intfloat/multilingual-e5-large": 1024,
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    }

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        embedding_service=None,
    ):
        """
        Initialize multilingual embedding service.

        Args:
            model_name: Multilingual model to use
            embedding_service: Base embedding service (for fallback)
        """
        self.model_name = model_name
        self.dimensions = self.MULTILINGUAL_MODELS.get(model_name, 768)
        self.base_service = embedding_service
        self._model = None

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text.

        Works with any language in the model's training set.

        Args:
            text: Text to embed (any supported language)

        Returns:
            Embedding vector
        """
        # Try to use HuggingFace model directly
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.warning("sentence-transformers not available")
            except Exception as e:
                logger.warning(f"Failed to load multilingual model: {e}")

        if self._model is not None:
            try:
                embedding = self._model.encode(text, convert_to_numpy=True)
                return embedding.tolist()
            except Exception as e:
                logger.warning(f"Multilingual embedding failed: {e}")

        # Fallback to base service
        if self.base_service:
            return await self.base_service.embed_text_async(text)

        raise RuntimeError("No embedding service available")

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embed multiple texts.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Failed to load multilingual model: {e}")

        if self._model is not None:
            try:
                embeddings = self._model.encode(texts, convert_to_numpy=True)
                return embeddings.tolist()
            except Exception as e:
                logger.warning(f"Batch multilingual embedding failed: {e}")

        # Fallback to individual embedding
        if self.base_service:
            return await self.base_service.embed_texts(texts)

        raise RuntimeError("No embedding service available")


class CrossLingualRetriever:
    """
    Retrieve documents across language barriers.

    Combines:
    - Multilingual embeddings for semantic search
    - Query translation for keyword matching enhancement
    - Result language detection and optional translation
    """

    def __init__(
        self,
        embedding_service: MultilingualEmbeddingService,
        vectorstore,
        llm=None,
        config: Optional[MultilingualSearchConfig] = None,
    ):
        """
        Initialize cross-lingual retriever.

        Args:
            embedding_service: Multilingual embedding service
            vectorstore: Vector store for search
            llm: LLM for translation (optional)
            config: Search configuration
        """
        self.embedding_service = embedding_service
        self.vectorstore = vectorstore
        self.llm = llm
        self.config = config or MultilingualSearchConfig()

        self.language_detector = LanguageDetector()
        self.translator = QueryTranslator(llm)

    async def search(
        self,
        query: str,
        session,
        top_k: int = 10,
        target_languages: Optional[List[Language]] = None,
        translate_results: bool = False,
        user_language: Optional[Language] = None,
    ) -> List[CrossLingualSearchResult]:
        """
        Perform cross-lingual search.

        Args:
            query: Search query (any language)
            session: Database session
            top_k: Number of results
            target_languages: Languages to search in (None = all)
            translate_results: Whether to translate results
            user_language: User's preferred language for translations

        Returns:
            List of CrossLingualSearchResult
        """
        # Detect query language
        query_detection = self.language_detector.detect(query)
        query_language = query_detection.detected_language

        logger.info(
            "Cross-lingual search",
            query_preview=query[:50],
            detected_language=query_language.value,
            confidence=query_detection.confidence,
        )

        results: List[CrossLingualSearchResult] = []

        # Strategy 1: Direct multilingual embedding search
        # Multilingual embeddings handle language-agnostic similarity
        query_embedding = await self.embedding_service.embed_text(query)

        if hasattr(self.vectorstore, 'search_by_embedding'):
            direct_results = await self.vectorstore.search_by_embedding(
                embedding=query_embedding,
                top_k=top_k * 2,  # Get more for diversity
            )
        else:
            # Fallback to regular search
            direct_results = await self.vectorstore.search(
                query=query,
                top_k=top_k * 2,
            )

        for r in direct_results:
            result_lang = self.language_detector.detect(r.content if hasattr(r, 'content') else str(r))

            results.append(CrossLingualSearchResult(
                document_id=r.document_id if hasattr(r, 'document_id') else "",
                chunk_id=r.chunk_id if hasattr(r, 'chunk_id') else "",
                content=r.content if hasattr(r, 'content') else str(r),
                similarity_score=r.similarity_score if hasattr(r, 'similarity_score') else 0.5,
                source_language=result_lang.detected_language,
                metadata=r.metadata if hasattr(r, 'metadata') else {},
            ))

        # Strategy 2: Query translation for keyword matching (optional)
        if self.config.enable_query_translation and self.llm:
            search_languages = target_languages or self.config.supported_languages

            # Translate query to target languages
            translations = await self.translator.translate_for_search(
                query,
                search_languages,
            )

            # Search with translated queries
            for lang, translated_query in translations.items():
                if lang != query_language:
                    translated_results = await self.vectorstore.search(
                        query=translated_query,
                        top_k=top_k // 2,
                    )

                    for r in translated_results:
                        results.append(CrossLingualSearchResult(
                            document_id=r.document_id if hasattr(r, 'document_id') else "",
                            chunk_id=r.chunk_id if hasattr(r, 'chunk_id') else "",
                            content=r.content if hasattr(r, 'content') else str(r),
                            similarity_score=r.similarity_score * 0.9 if hasattr(r, 'similarity_score') else 0.45,  # Slight penalty
                            source_language=lang,
                            metadata={
                                **(r.metadata if hasattr(r, 'metadata') else {}),
                                "translated_query": translated_query,
                            },
                        ))

        # Deduplicate by chunk_id
        seen_chunks = set()
        unique_results = []
        for r in results:
            if r.chunk_id and r.chunk_id not in seen_chunks:
                seen_chunks.add(r.chunk_id)
                unique_results.append(r)
            elif not r.chunk_id:
                unique_results.append(r)

        # Sort by similarity score
        unique_results.sort(key=lambda x: x.similarity_score, reverse=True)

        # Take top_k
        final_results = unique_results[:top_k]

        # Optionally translate results to user language
        if translate_results and user_language and self.llm:
            for result in final_results:
                if result.source_language != user_language:
                    translation = await self.translator.translate(
                        result.content,
                        result.source_language,
                        user_language,
                    )
                    if translation.confidence > 0.5:
                        result.translated_content = translation.translated_text

        logger.info(
            "Cross-lingual search complete",
            total_results=len(final_results),
            languages_found=list(set(r.source_language.value for r in final_results)),
        )

        return final_results


# =============================================================================
# Convenience Functions
# =============================================================================

_retriever_instance: Optional[CrossLingualRetriever] = None


def get_cross_lingual_retriever(
    embedding_service=None,
    vectorstore=None,
    llm=None,
) -> CrossLingualRetriever:
    """
    Get or create cross-lingual retriever singleton.

    Args:
        embedding_service: Multilingual embedding service
        vectorstore: Vector store
        llm: LLM for translation

    Returns:
        CrossLingualRetriever instance
    """
    global _retriever_instance

    if _retriever_instance is None:
        ml_embedding = MultilingualEmbeddingService(
            embedding_service=embedding_service,
        )
        _retriever_instance = CrossLingualRetriever(
            embedding_service=ml_embedding,
            vectorstore=vectorstore,
            llm=llm,
        )

    return _retriever_instance


async def detect_language(text: str) -> Language:
    """
    Convenience function to detect text language.

    Args:
        text: Text to analyze

    Returns:
        Detected Language
    """
    detector = LanguageDetector()
    result = detector.detect(text)
    return result.detected_language


async def translate_text(
    text: str,
    target_language: Language,
    llm=None,
) -> str:
    """
    Convenience function to translate text.

    Args:
        text: Text to translate
        target_language: Target language
        llm: LLM for translation

    Returns:
        Translated text
    """
    detector = LanguageDetector()
    source = detector.detect(text)

    if source.detected_language == target_language:
        return text

    translator = QueryTranslator(llm)
    result = await translator.translate(
        text,
        source.detected_language,
        target_language,
    )

    return result.translated_text
