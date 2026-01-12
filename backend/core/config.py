"""
AIDocumentIndexer - Core Configuration
========================================

Centralized configuration using Pydantic settings.
All environment variables and configuration options are defined here.

Usage:
    from backend.core.config import settings

    api_key = settings.OPENAI_API_KEY
    debug = settings.DEBUG
"""

import os
from functools import lru_cache
from typing import Optional, List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==========================================================================
    # Application
    # ==========================================================================
    APP_NAME: str = Field(default="AIDocumentIndexer", description="Application name")
    DEBUG: bool = Field(default=False, description="Debug mode")
    ENVIRONMENT: str = Field(default="development", description="Environment (development, staging, production)")

    # ==========================================================================
    # API Keys - LLM Providers
    # ==========================================================================
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API key")
    GOOGLE_API_KEY: str = Field(default="", description="Google AI API key")
    COHERE_API_KEY: str = Field(default="", description="Cohere API key")
    MISTRAL_API_KEY: str = Field(default="", description="Mistral AI API key")
    GROQ_API_KEY: str = Field(default="", description="Groq API key")
    TOGETHER_API_KEY: str = Field(default="", description="Together AI API key")
    FIREWORKS_API_KEY: str = Field(default="", description="Fireworks AI API key")
    DEEPSEEK_API_KEY: str = Field(default="", description="DeepSeek API key")

    # ==========================================================================
    # API Keys - Other Services
    # ==========================================================================
    ELEVENLABS_API_KEY: str = Field(default="", description="ElevenLabs TTS API key")
    SLACK_BOT_TOKEN: str = Field(default="", description="Slack Bot OAuth token")
    SLACK_SIGNING_SECRET: str = Field(default="", description="Slack signing secret")

    # ==========================================================================
    # Database
    # ==========================================================================
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/aidocindexer",
        description="PostgreSQL connection URL"
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )

    # ==========================================================================
    # Storage
    # ==========================================================================
    UPLOAD_DIR: str = Field(default="./uploads", description="Directory for uploaded files")
    AUDIO_OUTPUT_DIR: str = Field(default="./audio_output", description="Directory for generated audio files")
    TEMP_DIR: str = Field(default="./tmp", description="Temporary files directory")

    # ==========================================================================
    # LLM Configuration
    # ==========================================================================
    DEFAULT_LLM_PROVIDER: str = Field(default="openai", description="Default LLM provider")
    DEFAULT_CHAT_MODEL: str = Field(default="gpt-4o", description="Default chat model")
    DEFAULT_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="Default embedding model")

    # ==========================================================================
    # RAG Configuration
    # ==========================================================================
    RAG_TOP_K: int = Field(default=10, description="Number of documents to retrieve")
    RAG_SIMILARITY_THRESHOLD: float = Field(default=0.55, description="Minimum similarity score")
    ENABLE_QUERY_EXPANSION: bool = Field(default=True, description="Enable query expansion")
    ENABLE_VERIFICATION: bool = Field(default=True, description="Enable response verification")
    ENABLE_HYDE: bool = Field(default=True, description="Enable HyDE for retrieval")
    ENABLE_CRAG: bool = Field(default=True, description="Enable Corrective RAG")

    # ==========================================================================
    # Audio Configuration
    # ==========================================================================
    TTS_DEFAULT_PROVIDER: str = Field(default="openai", description="Default TTS provider (openai, elevenlabs, local)")
    TTS_DEFAULT_VOICE: str = Field(default="alloy", description="Default TTS voice")
    AUDIO_FORMAT: str = Field(default="mp3", description="Default audio format")
    AUDIO_SAMPLE_RATE: int = Field(default=24000, description="Audio sample rate in Hz")

    # ==========================================================================
    # Security
    # ==========================================================================
    SECRET_KEY: str = Field(default="change-me-in-production", description="Secret key for JWT tokens")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiration time")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token expiration time")

    # ==========================================================================
    # External Integrations
    # ==========================================================================
    GOOGLE_OAUTH_CLIENT_ID: str = Field(default="", description="Google OAuth client ID")
    GOOGLE_OAUTH_CLIENT_SECRET: str = Field(default="", description="Google OAuth client secret")
    NOTION_INTEGRATION_TOKEN: str = Field(default="", description="Notion integration token")

    # ==========================================================================
    # Feature Flags
    # ==========================================================================
    ENABLE_AUDIO_OVERVIEWS: bool = Field(default=True, description="Enable audio overview generation")
    ENABLE_WORKFLOW_ENGINE: bool = Field(default=True, description="Enable workflow automation")
    ENABLE_CONNECTORS: bool = Field(default=True, description="Enable external data connectors")
    ENABLE_LLM_GATEWAY: bool = Field(default=True, description="Enable LLM gateway with budget control")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra env vars


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
