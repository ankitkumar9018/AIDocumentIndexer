"""
AIDocumentIndexer - Schema Linker
==================================

Intelligent schema pruning for Text-to-SQL queries.
Reduces schema context by selecting only relevant tables based on
keyword matching and optional embedding similarity.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

from backend.services.connectors.database.base import DatabaseSchema, TableSchema

logger = structlog.get_logger(__name__)

# Common English stop words to ignore during keyword matching
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "because", "but", "and", "or",
    "if", "while", "about", "up", "it", "its", "this", "that", "what",
    "which", "who", "whom", "these", "those", "i", "me", "my", "we",
    "our", "you", "your", "he", "him", "his", "she", "her", "they",
    "them", "their", "many", "much", "show", "get", "find", "give",
    "tell", "list", "display",
}

# SQL-related keywords that map to aggregation/query patterns (not table names)
SQL_INTENT_WORDS = {
    "count", "total", "sum", "average", "avg", "maximum", "max",
    "minimum", "min", "top", "bottom", "highest", "lowest", "most",
    "least", "group", "per", "each", "every", "number",
}


class SchemaLinker:
    """
    Links natural language questions to relevant database tables.

    Uses keyword matching and optional embedding similarity to prune
    the schema to only tables relevant to the user's question.
    """

    def __init__(self, schema: DatabaseSchema, annotations: Optional[Dict[str, Any]] = None):
        self.schema = schema
        self.annotations = annotations or {}
        self._table_keywords: Dict[str, Set[str]] = {}
        self._build_table_keywords()

    def _build_table_keywords(self):
        """Build keyword index for each table from names, columns, and descriptions."""
        table_annotations = self.annotations.get("tables", {})

        for table in self.schema.tables:
            keywords = set()

            # Table name tokens (split on underscore, camelCase)
            keywords.update(self._tokenize(table.name))

            # Table description
            desc = table_annotations.get(table.name, {}).get("description") or table.description
            if desc:
                keywords.update(self._tokenize(desc))

            # Column names and descriptions
            col_annotations = table_annotations.get(table.name, {}).get("columns", {})
            for col in table.columns:
                keywords.update(self._tokenize(col.name))
                col_desc = col_annotations.get(col.name) or col.description
                if col_desc:
                    keywords.update(self._tokenize(col_desc))

            # Remove stop words
            keywords -= STOP_WORDS
            keywords -= SQL_INTENT_WORDS

            self._table_keywords[table.name] = keywords

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into lowercase words, splitting on common delimiters."""
        # Split camelCase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # Split on underscores, hyphens, spaces, and other non-alphanumeric
        tokens = re.split(r'[_\-\s,./]+', text.lower())
        return {t for t in tokens if len(t) > 1}

    def link_by_keywords(self, question: str) -> Dict[str, float]:
        """
        Score tables by keyword overlap with the question.

        Returns:
            Dict of table_name -> relevance_score (0-1)
        """
        question_tokens = self._tokenize(question) - STOP_WORDS - SQL_INTENT_WORDS
        if not question_tokens:
            return {}

        scores: Dict[str, float] = {}
        for table_name, table_keywords in self._table_keywords.items():
            if not table_keywords:
                continue

            # Count matching tokens
            overlap = question_tokens & table_keywords
            if overlap:
                # Jaccard-like score weighted toward question coverage
                score = len(overlap) / len(question_tokens)
                scores[table_name] = min(score, 1.0)

        return scores

    async def link_by_embeddings(self, question: str) -> Dict[str, float]:
        """
        Score tables by embedding similarity between question and table descriptions.

        Returns:
            Dict of table_name -> similarity_score (0-1)
        """
        try:
            from backend.services.embeddings import get_embedding_service
            embed_svc = get_embedding_service()

            # Build description text for each table
            table_descriptions = []
            table_names = []
            table_annotations = self.annotations.get("tables", {})

            for table in self.schema.tables:
                desc = table_annotations.get(table.name, {}).get("description") or table.description or ""
                col_names = ", ".join(c.name for c in table.columns[:10])
                text = f"Table {table.name}: {desc}. Columns: {col_names}"
                table_descriptions.append(text)
                table_names.append(table.name)

            if not table_descriptions:
                return {}

            # Embed question and descriptions
            q_embedding = await embed_svc.embed_text(question)
            desc_embeddings = await embed_svc.embed_texts(table_descriptions)

            # Cosine similarity
            import numpy as np
            q_vec = np.array(q_embedding)
            scores = {}
            for i, d_vec in enumerate(desc_embeddings):
                d_arr = np.array(d_vec)
                cos_sim = float(np.dot(q_vec, d_arr) / (np.linalg.norm(q_vec) * np.linalg.norm(d_arr) + 1e-8))
                scores[table_names[i]] = max(0.0, cos_sim)

            return scores

        except Exception as e:
            logger.warning("Embedding-based schema linking failed, falling back to keywords", error=str(e))
            return {}

    def expand_foreign_keys(self, table_names: Set[str]) -> Set[str]:
        """Add FK-connected tables to the selected set."""
        expanded = set(table_names)

        for table in self.schema.tables:
            if table.name not in table_names:
                continue
            for col in table.columns:
                if col.is_foreign_key and col.foreign_key_table:
                    expanded.add(col.foreign_key_table)

        # Also check reverse FK (tables that reference our selected tables)
        for table in self.schema.tables:
            if table.name in expanded:
                continue
            for col in table.columns:
                if col.is_foreign_key and col.foreign_key_table in table_names:
                    expanded.add(table.name)

        return expanded

    async def link(
        self,
        question: str,
        use_embeddings: bool = False,
        max_tables: int = 8,
    ) -> List[str]:
        """
        Select relevant tables for a question.

        Merges keyword and optional embedding scores, expands FK relations,
        and returns up to max_tables table names.
        """
        keyword_scores = self.link_by_keywords(question)

        if use_embeddings:
            embed_scores = await self.link_by_embeddings(question)
            # Merge: keyword 0.4 + embedding 0.6
            merged: Dict[str, float] = {}
            all_tables = set(keyword_scores.keys()) | set(embed_scores.keys())
            for t in all_tables:
                kw = keyword_scores.get(t, 0.0)
                em = embed_scores.get(t, 0.0)
                merged[t] = 0.4 * kw + 0.6 * em
        else:
            merged = keyword_scores

        if not merged:
            # Fallback: return all tables (up to max)
            return [t.name for t in self.schema.tables[:max_tables]]

        # Sort by score descending, take top candidates
        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        top_tables = {t for t, s in ranked[:max_tables] if s > 0.05}

        if not top_tables:
            top_tables = {ranked[0][0]} if ranked else set()

        # Expand FK relations
        expanded = self.expand_foreign_keys(top_tables)

        # Cap at max_tables
        if len(expanded) > max_tables:
            # Prioritize originally-scored tables
            result = []
            for t, _ in ranked:
                if t in expanded:
                    result.append(t)
                if len(result) >= max_tables:
                    break
            # Add FK tables if room
            for t in expanded:
                if t not in result and len(result) < max_tables:
                    result.append(t)
            return result

        return list(expanded)
