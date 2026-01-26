"""
AIDocumentIndexer - Spell Correction Service (Phase 65)
=======================================================

BK-tree based spell correction for search queries.

Features:
- O(log n) lookup using BK-tree (vs O(n) naive)
- Levenshtein edit distance
- Domain vocabulary learning from corpus
- Context-aware correction using n-gram probabilities

Research:
- BK-tree: Burkhard-Keller tree for metric space search
- SymSpell: Symmetric delete spelling correction (Wolfgarbe)
- Norvig Spell Corrector: Probabilistic approach

Usage:
    from backend.services.spell_correction import (
        SpellCorrector,
        get_spell_corrector,
    )

    corrector = get_spell_corrector()

    # Correct query
    corrected = await corrector.correct("machien lerning")
    # Returns: "machine learning"
"""

import asyncio
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SpellCorrectionConfig:
    """Configuration for spell correction."""
    # BK-tree settings
    max_edit_distance: int = 2  # Maximum edit distance to consider
    suggestion_limit: int = 5  # Max suggestions per word

    # Vocabulary settings
    min_word_freq: int = 2  # Minimum frequency to include word
    max_word_length: int = 50  # Ignore very long words

    # Corpus learning
    learn_from_corpus: bool = True
    corpus_vocab_file: str = "data/spell_vocab.json"

    # Default vocabulary
    include_english_words: bool = True

    # Context settings
    use_context: bool = True  # Use n-gram context
    ngram_weight: float = 0.3  # Weight for context in scoring


@dataclass
class CorrectionResult:
    """Result of spell correction."""
    original: str
    corrected: str
    corrections: List[Tuple[str, str, int]]  # (original, corrected, distance)
    confidence: float
    is_corrected: bool


# =============================================================================
# Edit Distance Functions
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.

    Operations: insert, delete, substitute (each costs 1).
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Damerau-Levenshtein distance (includes transpositions).

    Operations: insert, delete, substitute, transpose.
    """
    len1, len2 = len(s1), len(s2)

    # Create distance matrix
    d = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        d[i][0] = i
    for j in range(len2 + 1):
        d[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1

            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion
                d[i][j - 1] + 1,      # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )

            # Transposition
            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + cost)

    return d[len1][len2]


# =============================================================================
# BK-Tree Implementation
# =============================================================================

class BKTreeNode:
    """Node in a BK-tree."""

    def __init__(self, word: str, frequency: int = 1):
        self.word = word
        self.frequency = frequency
        self.children: Dict[int, "BKTreeNode"] = {}


class BKTree:
    """
    Burkhard-Keller tree for fast approximate string matching.

    BK-trees allow O(log n) spell checking by organizing words
    based on edit distance. Similar words are close in the tree.

    Reference:
    - Burkhard & Keller (1973): "Some approaches to best-match file searching"
    """

    def __init__(self, distance_fn=None):
        self.root: Optional[BKTreeNode] = None
        self.distance = distance_fn or levenshtein_distance
        self._size = 0

    def add(self, word: str, frequency: int = 1) -> None:
        """Add a word to the tree."""
        word = word.lower().strip()
        if not word:
            return

        if self.root is None:
            self.root = BKTreeNode(word, frequency)
            self._size = 1
            return

        node = self.root
        while True:
            dist = self.distance(word, node.word)

            if dist == 0:
                # Word already exists, update frequency
                node.frequency += frequency
                return

            if dist in node.children:
                node = node.children[dist]
            else:
                node.children[dist] = BKTreeNode(word, frequency)
                self._size += 1
                return

    def search(
        self,
        word: str,
        max_distance: int,
    ) -> List[Tuple[str, int, int]]:
        """
        Find all words within max_distance of the query word.

        Args:
            word: Query word
            max_distance: Maximum edit distance

        Returns:
            List of (word, distance, frequency) tuples
        """
        word = word.lower().strip()

        if self.root is None:
            return []

        results = []
        candidates = [self.root]

        while candidates:
            node = candidates.pop()
            dist = self.distance(word, node.word)

            if dist <= max_distance:
                results.append((node.word, dist, node.frequency))

            # Add children within possible range
            for child_dist, child in node.children.items():
                if dist - max_distance <= child_dist <= dist + max_distance:
                    candidates.append(child)

        # Sort by distance, then by frequency (descending)
        results.sort(key=lambda x: (x[1], -x[2]))

        return results

    def __len__(self) -> int:
        return self._size


# =============================================================================
# Spell Corrector
# =============================================================================

class SpellCorrector:
    """
    Spell correction using BK-tree and n-gram context.

    Combines:
    - BK-tree for fast candidate retrieval
    - Word frequency for probability ranking
    - N-gram context for disambiguation
    """

    def __init__(self, config: Optional[SpellCorrectionConfig] = None):
        self.config = config or SpellCorrectionConfig()
        self._tree = BKTree(damerau_levenshtein_distance)
        self._word_freq: Counter = Counter()
        self._bigrams: Counter = Counter()
        self._total_words: int = 0
        self._initialized: bool = False

        # Load vocabulary
        self._load_vocabulary()

    # =========================================================================
    # Vocabulary Management
    # =========================================================================

    def _load_vocabulary(self) -> None:
        """Load vocabulary from file or defaults."""
        # Load from corpus file if exists
        if os.path.exists(self.config.corpus_vocab_file):
            self._load_corpus_vocabulary()

        # Add default English words if enabled
        if self.config.include_english_words and len(self._tree) < 1000:
            self._add_default_vocabulary()

        self._initialized = True
        logger.info("Spell corrector initialized", vocab_size=len(self._tree))

    def _load_corpus_vocabulary(self) -> None:
        """Load vocabulary learned from corpus."""
        try:
            with open(self.config.corpus_vocab_file, "r") as f:
                data = json.load(f)

            for word, freq in data.get("words", {}).items():
                self._tree.add(word, freq)
                self._word_freq[word] = freq

            for bigram, freq in data.get("bigrams", {}).items():
                self._bigrams[bigram] = freq

            self._total_words = data.get("total_words", sum(self._word_freq.values()))

            logger.info("Loaded corpus vocabulary", vocab_size=len(self._tree))

        except Exception as e:
            logger.warning("Failed to load corpus vocabulary", error=str(e))

    def _add_default_vocabulary(self) -> None:
        """Add common English words and tech terms."""
        # Common words (high frequency)
        common_words = [
            # English
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
            "her", "was", "one", "our", "out", "has", "his", "how", "its", "may",
            "new", "now", "old", "see", "way", "who", "did", "get", "let", "say",
            "she", "too", "use", "will", "with", "what", "when", "where", "which",
            # Tech terms
            "api", "app", "code", "data", "file", "html", "http", "json", "link",
            "page", "server", "user", "web", "query", "search", "index", "document",
            "machine", "learning", "model", "neural", "network", "python", "java",
            "database", "function", "class", "method", "variable", "string", "number",
            "array", "object", "interface", "module", "package", "import", "export",
            # RAG/AI terms
            "embedding", "vector", "chunk", "token", "prompt", "context", "retrieval",
            "generation", "language", "transformer", "attention", "encoder", "decoder",
            "training", "inference", "dataset", "annotation", "labeling", "accuracy",
        ]

        for word in common_words:
            self._tree.add(word, 1000)  # High frequency
            self._word_freq[word] = 1000

    async def learn_from_text(self, text: str) -> int:
        """
        Learn vocabulary from text.

        Args:
            text: Text to learn from

        Returns:
            Number of words learned
        """
        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        words = [w for w in words if 2 <= len(w) <= self.config.max_word_length]

        # Count frequencies
        word_counts = Counter(words)
        n_learned = 0

        for word, freq in word_counts.items():
            if freq >= self.config.min_word_freq:
                self._tree.add(word, freq)
                self._word_freq[word] += freq
                n_learned += 1

        # Learn bigrams for context
        if self.config.use_context:
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i + 1]}"
                self._bigrams[bigram] += 1

        self._total_words += len(words)

        return n_learned

    async def save_vocabulary(self) -> None:
        """Save learned vocabulary to file."""
        os.makedirs(os.path.dirname(self.config.corpus_vocab_file), exist_ok=True)

        data = {
            "words": dict(self._word_freq),
            "bigrams": dict(self._bigrams.most_common(10000)),
            "total_words": self._total_words,
        }

        with open(self.config.corpus_vocab_file, "w") as f:
            json.dump(data, f)

        logger.info("Saved vocabulary", n_words=len(self._word_freq))

    # =========================================================================
    # Correction
    # =========================================================================

    async def correct(self, query: str) -> CorrectionResult:
        """
        Correct spelling in a query.

        Args:
            query: Input query

        Returns:
            CorrectionResult with corrected query and details
        """
        if not query or not query.strip():
            return CorrectionResult(
                original=query,
                corrected=query,
                corrections=[],
                confidence=1.0,
                is_corrected=False,
            )

        # Tokenize
        words = query.split()
        corrected_words = []
        corrections = []
        total_confidence = 0.0

        for i, word in enumerate(words):
            # Skip non-alphabetic tokens
            if not re.match(r'^[a-zA-Z]+$', word):
                corrected_words.append(word)
                total_confidence += 1.0
                continue

            # Check if word is known
            word_lower = word.lower()
            if word_lower in self._word_freq:
                corrected_words.append(word)
                total_confidence += 1.0
                continue

            # Get candidates
            context = words[i - 1] if i > 0 else None
            correction, distance, confidence = await self._correct_word(
                word_lower, context
            )

            if correction != word_lower:
                # Preserve capitalization
                if word[0].isupper():
                    correction = correction.capitalize()
                if word.isupper():
                    correction = correction.upper()

                corrections.append((word, correction, distance))
                corrected_words.append(correction)
            else:
                corrected_words.append(word)

            total_confidence += confidence

        avg_confidence = total_confidence / len(words) if words else 1.0
        corrected_query = " ".join(corrected_words)

        return CorrectionResult(
            original=query,
            corrected=corrected_query,
            corrections=corrections,
            confidence=avg_confidence,
            is_corrected=len(corrections) > 0,
        )

    async def _correct_word(
        self,
        word: str,
        context: Optional[str] = None,
    ) -> Tuple[str, int, float]:
        """
        Correct a single word.

        Args:
            word: Word to correct
            context: Previous word for context

        Returns:
            (corrected_word, edit_distance, confidence)
        """
        # Run search in thread pool
        loop = asyncio.get_running_loop()
        candidates = await loop.run_in_executor(
            None,
            self._tree.search,
            word,
            self.config.max_edit_distance,
        )

        if not candidates:
            return (word, 0, 0.5)  # Unknown word, low confidence

        # Score candidates
        scored = []
        for candidate, distance, freq in candidates[:self.config.suggestion_limit]:
            # Base score from frequency
            freq_score = freq / (self._total_words + 1)

            # Distance penalty
            distance_penalty = 1.0 / (1.0 + distance)

            # Context bonus
            context_score = 0.0
            if self.config.use_context and context:
                bigram = f"{context.lower()} {candidate}"
                if bigram in self._bigrams:
                    context_score = self._bigrams[bigram] / (self._total_words + 1)

            # Combined score
            score = (
                (1 - self.config.ngram_weight) * (freq_score * distance_penalty)
                + self.config.ngram_weight * context_score
            )

            scored.append((candidate, distance, score))

        # Sort by score (descending)
        scored.sort(key=lambda x: x[2], reverse=True)

        best = scored[0]
        confidence = min(best[2] * 100, 1.0)  # Normalize to 0-1

        return (best[0], best[1], confidence)

    async def suggest(
        self,
        word: str,
        limit: int = 5,
    ) -> List[Tuple[str, int]]:
        """
        Get spelling suggestions for a word.

        Args:
            word: Word to get suggestions for
            limit: Maximum number of suggestions

        Returns:
            List of (suggestion, distance) tuples
        """
        loop = asyncio.get_running_loop()
        candidates = await loop.run_in_executor(
            None,
            self._tree.search,
            word.lower(),
            self.config.max_edit_distance,
        )

        # Return top suggestions
        return [(c[0], c[1]) for c in candidates[:limit]]

    def get_stats(self) -> Dict[str, Any]:
        """Get spell corrector statistics."""
        return {
            "vocabulary_size": len(self._tree),
            "total_words_seen": self._total_words,
            "bigrams_learned": len(self._bigrams),
            "initialized": self._initialized,
            "max_edit_distance": self.config.max_edit_distance,
        }


# =============================================================================
# Singleton
# =============================================================================

_spell_corrector: Optional[SpellCorrector] = None


def get_spell_corrector(
    config: Optional[SpellCorrectionConfig] = None,
) -> SpellCorrector:
    """Get or create spell corrector singleton."""
    global _spell_corrector

    if _spell_corrector is None or config is not None:
        _spell_corrector = SpellCorrector(config)

    return _spell_corrector


# =============================================================================
# Convenience Functions
# =============================================================================

async def correct_query(query: str) -> str:
    """Convenience function to correct a query."""
    result = await get_spell_corrector().correct(query)
    return result.corrected
