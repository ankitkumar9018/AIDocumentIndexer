"""
AIDocumentIndexer - Constants Module
=====================================

Centralized constants for the application.
Extracted for better maintainability and consistency.
"""

from enum import Enum, auto
from typing import Dict


# =============================================================================
# LLM Provider Constants
# =============================================================================

class LLMProvider(str, Enum):
    """Available LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class LLMOperation(str, Enum):
    """Operations that use LLM."""
    RAG = "rag"
    CHAT = "chat"
    EMBEDDING = "embedding"
    SUMMARIZATION = "summarization"
    CONTENT_GENERATION = "content_generation"
    QUERY_EXPANSION = "query_expansion"
    VERIFICATION = "verification"
    TAGGING = "tagging"


# Default models by provider
DEFAULT_CHAT_MODELS: Dict[str, str] = {
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    LLMProvider.OLLAMA: "llama3.2",
}

DEFAULT_EMBEDDING_MODELS: Dict[str, str] = {
    LLMProvider.OPENAI: "text-embedding-3-small",
    LLMProvider.OLLAMA: "nomic-embed-text",
}

# Embedding dimensions by model
EMBEDDING_DIMENSIONS: Dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "mxbai-embed-large": 1024,
}


# =============================================================================
# Document Processing Constants
# =============================================================================

class ProcessingMode(str, Enum):
    """Document processing modes."""
    BASIC = "basic"           # Text extraction only (fastest)
    OCR_ENABLED = "ocr"       # Text + OCR for scanned documents
    FULL = "full"             # Text + OCR + AI image analysis (most thorough)


class DocumentStatus(str, Enum):
    """Document lifecycle status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    QUEUED = "queued"


# Default chunking parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 0.15
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 2000


# =============================================================================
# Agent Types
# =============================================================================

class AgentType(str, Enum):
    """Multi-LLM agent types for collaboration."""
    ANALYST = "analyst"
    RESEARCHER = "researcher"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    FACT_CHECKER = "fact_checker"


# Agent configurations
AGENT_CONFIGS = {
    AgentType.ANALYST: {
        "name": "Analyst",
        "description": "Analyzes data and provides insights",
        "preferred_model": "gpt-4o",
    },
    AgentType.RESEARCHER: {
        "name": "Researcher",
        "description": "Searches and retrieves relevant information",
        "preferred_model": "gpt-4o",
    },
    AgentType.CRITIC: {
        "name": "Critic",
        "description": "Critically evaluates responses and identifies issues",
        "preferred_model": "claude-3-5-sonnet-20241022",
    },
    AgentType.SYNTHESIZER: {
        "name": "Synthesizer",
        "description": "Combines multiple viewpoints into coherent response",
        "preferred_model": "gpt-4o",
    },
    AgentType.FACT_CHECKER: {
        "name": "Fact Checker",
        "description": "Verifies claims against source documents",
        "preferred_model": "gpt-4o-mini",
    },
}


# =============================================================================
# Access Tier Constants
# =============================================================================

class AccessTier(int, Enum):
    """User access tier levels."""
    VIEWER = 10
    USER = 30
    EDITOR = 50
    MANAGER = 90
    ADMIN = 100


ACCESS_TIER_NAMES: Dict[int, str] = {
    AccessTier.VIEWER: "Viewer",
    AccessTier.USER: "User",
    AccessTier.EDITOR: "Editor",
    AccessTier.MANAGER: "Manager",
    AccessTier.ADMIN: "Admin",
}


# =============================================================================
# RAG Constants
# =============================================================================

class SearchType(str, Enum):
    """Vector search types."""
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class VerificationLevel(str, Enum):
    """Self-RAG verification levels."""
    NONE = "none"
    QUICK = "quick"
    STANDARD = "standard"
    THOROUGH = "thorough"


# Default RAG parameters
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.4
DEFAULT_HYBRID_ALPHA = 0.5


# =============================================================================
# File Type Constants
# =============================================================================

SUPPORTED_FILE_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "txt": "text/plain",
    "md": "text/markdown",
    "html": "text/html",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv",
    "json": "application/json",
    "xml": "application/xml",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}

IMAGE_FILE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"}
DOCUMENT_FILE_EXTENSIONS = {"pdf", "docx", "doc", "txt", "md", "html"}
SPREADSHEET_FILE_EXTENSIONS = {"xlsx", "xls", "csv"}
PRESENTATION_FILE_EXTENSIONS = {"pptx", "ppt"}


# =============================================================================
# HTTP Status Messages
# =============================================================================

ERROR_MESSAGES = {
    "unauthorized": "Authentication required",
    "forbidden": "You don't have permission to access this resource",
    "not_found": "Resource not found",
    "validation_error": "Invalid request data",
    "internal_error": "An internal error occurred",
    "rate_limited": "Too many requests, please try again later",
    "service_unavailable": "Service temporarily unavailable",
}


# =============================================================================
# Cache TTL Constants (in seconds)
# =============================================================================

class CacheTTL(int, Enum):
    """Cache time-to-live values."""
    SHORT = 60  # 1 minute
    MEDIUM = 300  # 5 minutes
    LONG = 3600  # 1 hour
    DAY = 86400  # 24 hours


# =============================================================================
# Generation Constants
# =============================================================================

class OutputFormat(str, Enum):
    """Document generation output formats."""
    PPTX = "pptx"
    DOCX = "docx"
    PDF = "pdf"
    XLSX = "xlsx"
    MARKDOWN = "markdown"
    HTML = "html"
    TXT = "txt"


class GenerationStatus(str, Enum):
    """Document generation workflow status."""
    DRAFT = "draft"
    OUTLINE_PENDING = "outline_pending"
    OUTLINE_APPROVED = "outline_approved"
    GENERATING = "generating"
    SECTION_REVIEW = "section_review"
    REVISION = "revision"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


# =============================================================================
# Feature Flags (can be overridden by environment)
# =============================================================================

DEFAULT_FEATURE_FLAGS = {
    "query_expansion": True,
    "summarization": True,
    "verification": True,
    "multi_llm": True,
    "image_generation": False,
    "web_scraping": True,
    "file_watcher": False,
}
