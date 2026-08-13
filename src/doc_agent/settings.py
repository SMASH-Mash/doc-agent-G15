"""FIXED — typed settings from environment (secrets live here, never in code/config)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_api_key: str = ""
    wandb_api_key: str = ""


settings = Settings()  # import this; do not read os.environ elsewhere
