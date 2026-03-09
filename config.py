from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Settings:
    alphavantage_api_key: str = ""
    alphavantage_api_keys: tuple[str, ...] = ()
    alphavantage_key_daily_limit: int = 25
    sqlite_path: str = "vector_store.sqlite"
    sqlite_table: str = "api_calls"
    vector_index_path: str = "vector_index.bin"
    ann_index_space: str = "ip"
    ann_ef: int = 200
    ann_m: int = 16
    vector_dim: int = 768
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

    raw_multi_keys = os.getenv("ALPHAVANTAGE_API_KEYS", "")
    parsed_multi_keys = tuple([k.strip() for k in raw_multi_keys.split(",") if k.strip()])

    return Settings(
        alphavantage_api_key=os.getenv("ALPHAVANTAGE_API_KEY", ""),
        alphavantage_api_keys=parsed_multi_keys,
        alphavantage_key_daily_limit=_get_int("ALPHAVANTAGE_KEY_DAILY_LIMIT", 25),
        sqlite_path=os.getenv("SQLITE_PATH", "vector_store.sqlite"),
        sqlite_table=os.getenv("SQLITE_TABLE", "api_calls"),
        vector_index_path=os.getenv("VECTOR_INDEX_PATH", "vector_index.bin"),
        ann_index_space=os.getenv("ANN_INDEX_SPACE", "ip"),
        ann_ef=_get_int("ANN_EF", 200),
        ann_m=_get_int("ANN_M", 16),
        vector_dim=_get_int("VECTOR_DIM", 768),
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
