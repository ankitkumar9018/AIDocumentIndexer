"""
AIDocumentIndexer - System Settings Service
============================================

Manages system-wide configuration settings stored in the database.
Provides default values and type conversion for settings.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import structlog
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.database import async_session_context
from backend.db.models import SystemSettings

logger = structlog.get_logger(__name__)


# =============================================================================
# Types
# =============================================================================

class SettingCategory(str, Enum):
    """Setting categories for organization."""
    LLM = "llm"
    DATABASE = "database"
    SECURITY = "security"
    NOTIFICATIONS = "notifications"
    GENERAL = "general"
    RAG = "rag"  # Advanced RAG features
    OCR = "ocr"  # OCR configuration


class ValueType(str, Enum):
    """Setting value types."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    JSON = "json"


@dataclass
class SettingDefinition:
    """Definition for a setting including default value."""
    key: str
    category: SettingCategory
    default_value: Any
    value_type: ValueType
    description: str


# =============================================================================
# Default Settings Definitions
# =============================================================================

DEFAULT_SETTINGS: List[SettingDefinition] = [
    # LLM Configuration
    SettingDefinition(
        key="llm.default_model",
        category=SettingCategory.LLM,
        default_value="gpt-4",
        value_type=ValueType.STRING,
        description="Default LLM model for chat and generation"
    ),
    SettingDefinition(
        key="llm.embedding_model",
        category=SettingCategory.LLM,
        default_value="text-embedding-3-small",
        value_type=ValueType.STRING,
        description="Model used for generating embeddings"
    ),
    SettingDefinition(
        key="llm.temperature",
        category=SettingCategory.LLM,
        default_value=0.7,
        value_type=ValueType.NUMBER,
        description="Default temperature for LLM responses"
    ),
    SettingDefinition(
        key="llm.max_tokens",
        category=SettingCategory.LLM,
        default_value=4096,
        value_type=ValueType.NUMBER,
        description="Maximum tokens for LLM responses"
    ),

    # Database Configuration
    SettingDefinition(
        key="database.vector_dimensions",
        category=SettingCategory.DATABASE,
        default_value=1536,
        value_type=ValueType.NUMBER,
        description="Dimensions for vector embeddings"
    ),
    SettingDefinition(
        key="database.index_type",
        category=SettingCategory.DATABASE,
        default_value="hnsw",
        value_type=ValueType.STRING,
        description="Vector index type (hnsw, ivfflat)"
    ),
    SettingDefinition(
        key="database.max_results_per_query",
        category=SettingCategory.DATABASE,
        default_value=10,
        value_type=ValueType.NUMBER,
        description="Maximum results returned per search query"
    ),

    # Security Configuration
    SettingDefinition(
        key="security.require_email_verification",
        category=SettingCategory.SECURITY,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Require email verification for new users"
    ),
    SettingDefinition(
        key="security.enable_2fa",
        category=SettingCategory.SECURITY,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable two-factor authentication for admin accounts"
    ),
    SettingDefinition(
        key="security.enable_audit_logging",
        category=SettingCategory.SECURITY,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Log all user actions for compliance"
    ),
    SettingDefinition(
        key="security.session_timeout_minutes",
        category=SettingCategory.SECURITY,
        default_value=60,
        value_type=ValueType.NUMBER,
        description="Session timeout in minutes"
    ),

    # Notification Configuration
    SettingDefinition(
        key="notifications.processing_completed",
        category=SettingCategory.NOTIFICATIONS,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Notify when document processing completes"
    ),
    SettingDefinition(
        key="notifications.processing_failed",
        category=SettingCategory.NOTIFICATIONS,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Notify when document processing fails"
    ),
    SettingDefinition(
        key="notifications.cost_alerts",
        category=SettingCategory.NOTIFICATIONS,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Notify when API costs exceed threshold"
    ),

    # ==========================================================================
    # Advanced RAG Configuration
    # ==========================================================================

    # GraphRAG Settings
    SettingDefinition(
        key="rag.graphrag_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable GraphRAG for knowledge graph-based retrieval"
    ),
    SettingDefinition(
        key="rag.graph_max_hops",
        category=SettingCategory.RAG,
        default_value=2,
        value_type=ValueType.NUMBER,
        description="Maximum hops for graph traversal (1-3)"
    ),
    SettingDefinition(
        key="rag.entity_extraction_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Extract entities during document processing"
    ),

    # Agentic RAG Settings
    SettingDefinition(
        key="rag.agentic_enabled",
        category=SettingCategory.RAG,
        default_value=False,
        value_type=ValueType.BOOLEAN,
        description="Enable Agentic RAG for complex multi-step queries"
    ),
    SettingDefinition(
        key="rag.agentic_max_iterations",
        category=SettingCategory.RAG,
        default_value=5,
        value_type=ValueType.NUMBER,
        description="Maximum ReAct loop iterations (1-10)"
    ),
    SettingDefinition(
        key="rag.auto_detect_complexity",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Auto-detect complex queries for agentic mode"
    ),

    # Multimodal RAG Settings
    SettingDefinition(
        key="rag.multimodal_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable multimodal RAG for images/tables/charts"
    ),
    SettingDefinition(
        key="rag.vision_provider",
        category=SettingCategory.RAG,
        default_value="auto",
        value_type=ValueType.STRING,
        description="Vision model provider (auto, ollama, openai, anthropic)"
    ),
    SettingDefinition(
        key="rag.ollama_vision_model",
        category=SettingCategory.RAG,
        default_value="llava",
        value_type=ValueType.STRING,
        description="Ollama vision model for free multimodal (llava, bakllava)"
    ),
    SettingDefinition(
        key="rag.extract_tables",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Extract and index tables from documents"
    ),
    SettingDefinition(
        key="rag.caption_images",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Generate captions for images during indexing"
    ),

    # Real-Time Updates Settings
    SettingDefinition(
        key="rag.incremental_indexing",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable incremental indexing (update only changed chunks)"
    ),
    SettingDefinition(
        key="rag.freshness_tracking",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Track document freshness and flag stale content"
    ),
    SettingDefinition(
        key="rag.freshness_threshold_days",
        category=SettingCategory.RAG,
        default_value=30,
        value_type=ValueType.NUMBER,
        description="Days until content is considered aging"
    ),
    SettingDefinition(
        key="rag.stale_threshold_days",
        category=SettingCategory.RAG,
        default_value=90,
        value_type=ValueType.NUMBER,
        description="Days until content is considered stale"
    ),

    # Query Suggestions
    SettingDefinition(
        key="rag.suggested_questions_enabled",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Generate follow-up question suggestions after answers"
    ),
    SettingDefinition(
        key="rag.suggestions_count",
        category=SettingCategory.RAG,
        default_value=3,
        value_type=ValueType.NUMBER,
        description="Number of suggested questions to generate (1-5)"
    ),

    # Hybrid Search Settings
    SettingDefinition(
        key="rag.graph_weight",
        category=SettingCategory.RAG,
        default_value=0.3,
        value_type=ValueType.NUMBER,
        description="Weight for graph results in hybrid search (0-1)"
    ),
    SettingDefinition(
        key="rag.vector_weight",
        category=SettingCategory.RAG,
        default_value=0.7,
        value_type=ValueType.NUMBER,
        description="Weight for vector results in hybrid search (0-1)"
    ),

    # Retrieval Settings
    SettingDefinition(
        key="rag.top_k",
        category=SettingCategory.RAG,
        default_value=10,
        value_type=ValueType.NUMBER,
        description="Number of documents to retrieve per query (higher = broader search)"
    ),
    SettingDefinition(
        key="rag.rerank_results",
        category=SettingCategory.RAG,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Enable cross-encoder reranking for better relevance ordering"
    ),
    SettingDefinition(
        key="rag.query_expansion_count",
        category=SettingCategory.RAG,
        default_value=3,
        value_type=ValueType.NUMBER,
        description="Number of query variations to generate (improves recall)"
    ),
    SettingDefinition(
        key="rag.similarity_threshold",
        category=SettingCategory.RAG,
        default_value=0.4,
        value_type=ValueType.NUMBER,
        description="Minimum similarity score for document retrieval (0.0-1.0)"
    ),

    # OCR Configuration
    SettingDefinition(
        key="ocr.provider",
        category=SettingCategory.OCR,
        default_value="paddleocr",
        value_type=ValueType.STRING,
        description="OCR provider (paddleocr, easyocr, tesseract, auto)"
    ),
    SettingDefinition(
        key="ocr.paddle.variant",
        category=SettingCategory.OCR,
        default_value="server",
        value_type=ValueType.STRING,
        description="PaddleOCR model variant (server=accurate, mobile=fast)"
    ),
    SettingDefinition(
        key="ocr.paddle.languages",
        category=SettingCategory.OCR,
        default_value=["en", "de"],
        value_type=ValueType.JSON,
        description="List of language codes for OCR"
    ),
    SettingDefinition(
        key="ocr.paddle.model_dir",
        category=SettingCategory.OCR,
        default_value="./data/paddle_models",
        value_type=ValueType.STRING,
        description="Directory for PaddleOCR model cache"
    ),
    SettingDefinition(
        key="ocr.paddle.auto_download",
        category=SettingCategory.OCR,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Auto-download models on startup"
    ),
    SettingDefinition(
        key="ocr.tesseract.fallback_enabled",
        category=SettingCategory.OCR,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Fall back to Tesseract if PaddleOCR fails"
    ),
    # EasyOCR Settings
    SettingDefinition(
        key="ocr.easyocr.languages",
        category=SettingCategory.OCR,
        default_value=["en"],
        value_type=ValueType.JSON,
        description="List of language codes for EasyOCR"
    ),
    SettingDefinition(
        key="ocr.easyocr.use_gpu",
        category=SettingCategory.OCR,
        default_value=True,
        value_type=ValueType.BOOLEAN,
        description="Use GPU acceleration for EasyOCR (if available)"
    ),
]

# Build lookup dictionary
SETTINGS_DEFINITIONS: Dict[str, SettingDefinition] = {
    sd.key: sd for sd in DEFAULT_SETTINGS
}


# =============================================================================
# Settings Service
# =============================================================================

class SettingsService:
    """
    Service for managing system settings.

    Features:
    - Get/set individual settings
    - Get all settings by category
    - Type conversion and validation
    - Default values for unset settings
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_loaded = False

    def _convert_value(self, value: str, value_type: ValueType) -> Any:
        """Convert string value to appropriate type."""
        if value is None:
            return None

        try:
            if value_type == ValueType.NUMBER:
                # Try int first, then float
                try:
                    return int(value)
                except ValueError:
                    return float(value)
            elif value_type == ValueType.BOOLEAN:
                return value.lower() in ("true", "1", "yes", "on")
            elif value_type == ValueType.JSON:
                return json.loads(value)
            else:
                return value
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning("Failed to convert setting value", error=str(e))
            return value

    def _serialize_value(self, value: Any, value_type: ValueType) -> str:
        """Serialize value to string for storage."""
        if value is None:
            return ""

        if value_type == ValueType.BOOLEAN:
            return "true" if value else "false"
        elif value_type == ValueType.JSON:
            return json.dumps(value)
        else:
            return str(value)

    async def get_setting(
        self,
        key: str,
        session: Optional[AsyncSession] = None,
    ) -> Any:
        """
        Get a single setting value.

        Returns the stored value or default if not set.
        """
        definition = SETTINGS_DEFINITIONS.get(key)
        default_value = definition.default_value if definition else None
        value_type = definition.value_type if definition else ValueType.STRING

        async def _get(db: AsyncSession) -> Any:
            query = select(SystemSettings).where(SystemSettings.key == key)
            result = await db.execute(query)
            setting = result.scalar_one_or_none()

            if setting is None:
                return default_value

            return self._convert_value(setting.value, value_type)

        if session:
            return await _get(session)

        async with async_session_context() as db:
            return await _get(db)

    async def set_setting(
        self,
        key: str,
        value: Any,
        session: Optional[AsyncSession] = None,
    ) -> None:
        """
        Set a single setting value.

        Creates the setting if it doesn't exist.
        """
        definition = SETTINGS_DEFINITIONS.get(key)
        value_type = definition.value_type if definition else ValueType.STRING
        category = definition.category.value if definition else SettingCategory.GENERAL.value
        description = definition.description if definition else None

        serialized = self._serialize_value(value, value_type)

        async def _set(db: AsyncSession) -> None:
            query = select(SystemSettings).where(SystemSettings.key == key)
            result = await db.execute(query)
            setting = result.scalar_one_or_none()

            if setting:
                setting.value = serialized
            else:
                setting = SystemSettings(
                    key=key,
                    value=serialized,
                    category=category,
                    description=description,
                    value_type=value_type.value,
                )
                db.add(setting)

            await db.commit()

        if session:
            await _set(session)
        else:
            async with async_session_context() as db:
                await _set(db)

        # Invalidate cache
        self._cache_loaded = False

    async def get_all_settings(
        self,
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        Get all settings including defaults.

        Returns a dictionary with all settings, using defaults
        for any that haven't been explicitly set.
        """
        async def _get_all(db: AsyncSession) -> Dict[str, Any]:
            # Start with defaults
            result = {
                sd.key: sd.default_value
                for sd in DEFAULT_SETTINGS
            }

            # Get stored settings
            query = select(SystemSettings)
            db_result = await db.execute(query)
            stored = db_result.scalars().all()

            # Override with stored values
            for setting in stored:
                definition = SETTINGS_DEFINITIONS.get(setting.key)
                value_type = ValueType(setting.value_type) if setting.value_type else ValueType.STRING

                if definition:
                    value_type = definition.value_type

                result[setting.key] = self._convert_value(setting.value, value_type)

            return result

        if session:
            return await _get_all(session)

        async with async_session_context() as db:
            return await _get_all(db)

    async def get_settings_by_category(
        self,
        category: SettingCategory,
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        Get all settings for a specific category.
        """
        all_settings = await self.get_all_settings(session)

        return {
            key: value
            for key, value in all_settings.items()
            if key.startswith(f"{category.value}.")
        }

    async def update_settings(
        self,
        settings: Dict[str, Any],
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        Update multiple settings at once.

        Returns the updated settings.
        """
        async def _update(db: AsyncSession) -> Dict[str, Any]:
            for key, value in settings.items():
                definition = SETTINGS_DEFINITIONS.get(key)
                value_type = definition.value_type if definition else ValueType.STRING
                category = definition.category.value if definition else SettingCategory.GENERAL.value
                description = definition.description if definition else None

                serialized = self._serialize_value(value, value_type)

                query = select(SystemSettings).where(SystemSettings.key == key)
                result = await db.execute(query)
                setting = result.scalar_one_or_none()

                if setting:
                    setting.value = serialized
                else:
                    setting = SystemSettings(
                        key=key,
                        value=serialized,
                        category=category,
                        description=description,
                        value_type=value_type.value,
                    )
                    db.add(setting)

            await db.commit()

            # Return all settings after update
            return await self.get_all_settings(db)

        if session:
            result = await _update(session)
        else:
            async with async_session_context() as db:
                result = await _update(db)

        # Invalidate cache
        self._cache_loaded = False

        return result

    async def reset_to_defaults(
        self,
        session: Optional[AsyncSession] = None,
    ) -> Dict[str, Any]:
        """
        Reset all settings to their default values.
        """
        async def _reset(db: AsyncSession) -> Dict[str, Any]:
            # Delete all stored settings
            query = select(SystemSettings)
            result = await db.execute(query)
            for setting in result.scalars().all():
                await db.delete(setting)

            await db.commit()

            # Return defaults
            return {
                sd.key: sd.default_value
                for sd in DEFAULT_SETTINGS
            }

        if session:
            return await _reset(session)

        async with async_session_context() as db:
            return await _reset(db)

    def get_default_value(self, key: str) -> Any:
        """
        Get the default value for a setting (synchronous).

        This doesn't read from the database - just returns the hardcoded default.
        Useful for initialization when async DB access isn't available.
        """
        definition = SETTINGS_DEFINITIONS.get(key)
        return definition.default_value if definition else None

    def get_setting_definitions(self) -> List[Dict[str, Any]]:
        """
        Get all setting definitions with metadata.

        Useful for building settings UI.
        """
        return [
            {
                "key": sd.key,
                "category": sd.category.value,
                "default_value": sd.default_value,
                "value_type": sd.value_type.value,
                "description": sd.description,
            }
            for sd in DEFAULT_SETTINGS
        ]


# =============================================================================
# Singleton Instance
# =============================================================================

_settings_service: Optional[SettingsService] = None


def get_settings_service() -> SettingsService:
    """Get the settings service singleton instance."""
    global _settings_service
    if _settings_service is None:
        _settings_service = SettingsService()
    return _settings_service
