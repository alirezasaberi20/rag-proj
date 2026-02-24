"""
Application configuration management.

Uses pydantic-settings for environment-based configuration
with validation and type safety.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "RAG Chatbot API"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # API
    api_prefix: str = "/api/v1"
    allowed_origins: list[str] = Field(default=["*"])

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "tinyllama"
    ollama_timeout: int = 120

    # Embedding Configuration
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector Store Configuration
    vector_store_path: str = "./data/vectorstore"
    collection_name: str = "documents"

    # Document Processing
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_documents: int = 1000

    # RAG Configuration
    retrieval_top_k: int = 3
    max_context_length: int = 2000

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
