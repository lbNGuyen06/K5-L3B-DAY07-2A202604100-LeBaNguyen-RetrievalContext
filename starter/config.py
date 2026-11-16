"""Load settings from the Day 7 root .env file.

Values are returned to the caller only; never print or log the API key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# starter/config.py -> starter/ -> project root (where .env lives)
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "strategies.json"


@dataclass(frozen=True)
class Settings:
    database_url: str
    gemini_api_key: str
    embedding_model: str

    def describe(self) -> str:
        """Safe summary for logs: shows presence, never the secret values."""
        return (f"database_url={'set' if self.database_url else 'MISSING'}, "
                f"gemini_api_key={'set' if self.gemini_api_key else 'MISSING'}, "
                f"embedding_model={self.embedding_model}")


def load_settings(env_path: Path = ENV_PATH) -> Settings:
    if not env_path.is_file():
        raise FileNotFoundError(f"không tìm thấy .env tại {env_path}")
    load_dotenv(env_path, override=False)

    database_url = os.environ.get("DATABASE_URL", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not database_url:
        raise RuntimeError("thiếu DATABASE_URL trong .env")
    if not gemini_key:
        raise RuntimeError("thiếu GEMINI_API_KEY trong .env")

    embedding_model = os.environ.get("EMBEDDING_MODEL", "").strip()
    if not embedding_model:
        import json
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        embedding_model = cfg["embedding_model"]

    return Settings(database_url=database_url, gemini_api_key=gemini_key,
                    embedding_model=embedding_model)


if __name__ == "__main__":
    print(load_settings().describe())
