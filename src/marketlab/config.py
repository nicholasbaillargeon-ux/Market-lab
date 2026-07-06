"""Runtime configuration, loaded from environment / .env via pydantic-settings."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="", extra="ignore", case_sensitive=False
    )

    # Providers
    polygon_api_key: str | None = Field(default=None, alias="POLYGON_API_KEY")
    provider: str = Field(default="yfinance", alias="MARKETLAB_PROVIDER")

    # Storage
    parquet_root: Path = Field(default=Path("./data/lake"), alias="MARKETLAB_PARQUET_ROOT")
    pg_dsn: str | None = Field(default=None, alias="MARKETLAB_PG_DSN")

    @property
    def postgres_enabled(self) -> bool:
        return bool(self.pg_dsn)


def get_settings() -> Settings:
    return Settings()
