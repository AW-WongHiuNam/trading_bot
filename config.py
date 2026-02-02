from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Settings:
    alphavantage_api_key: str = ""
    qdrant_url: Optional[str] = "http://localhost:7500"
    qdrant_path: str = "qdrant_db"
    qdrant_collection: str = "api_calls"
    qdrant_force_mock_embed: bool = False
    ollama_embed_url: Optional[str] = "http://127.0.0.1:11434/api/embeddings"
    ollama_embed_model: str = "nomic-embed-text:latest"
    ollama_completion_url: str = "http://127.0.0.1:11434/completion"
    ollama_chat_model: str = "qwen2.5:14b"
    completion_max_tokens: int = 512
    chunk_size: int = 2000
    chunk_overlap: int = 400
    vector_top_k: int = 5
    data_dir: str = "data"
    default_output_path: Optional[str] = None


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_config(dotenv_path: str = ".env") -> Settings:
    """Load settings from .env (if present) and environment variables."""
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)

    return Settings(
        alphavantage_api_key=os.getenv("ALPHAVANTAGE_API_KEY", ""),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:7500"),
        qdrant_path=os.getenv("QDRANT_PATH", "qdrant_db"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "api_calls"),
        qdrant_force_mock_embed=_get_bool("QDRANT_FORCE_MOCK_EMBED", False),
        ollama_embed_url=os.getenv("OLLAMA_EMBED_URL", "http://127.0.0.1:11434/api/embeddings"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest"),
        ollama_completion_url=os.getenv("OLLAMA_COMPLETION_URL", "http://127.0.0.1:11434/completion"),
        ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:14b"),
        completion_max_tokens=_get_int("COMPLETION_MAX_TOKENS", 512),
        chunk_size=_get_int("CHUNK_SIZE", 2000),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 400),
        vector_top_k=_get_int("VECTOR_TOP_K", 5),
        data_dir=os.getenv("DATA_DIR", "data"),
        default_output_path=os.getenv("DEFAULT_OUTPUT_PATH"),
    )


# Provide a cached singleton for convenience.
_settings_cache: Optional[Settings] = None


def get_settings(dotenv_path: str = ".env") -> Settings:
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = load_config(dotenv_path)
    return _settings_cache
