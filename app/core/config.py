
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LLM Engineering Lab"
    app_version: str = "0.1.0"
    environment: str = "development"

    groq_api_key: str = ""
    groq_model: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def validate_groq_settings(self) -> None:
        if not self.groq_api_key.strip():
            raise ValueError("GROQ_API_KEY is not configured")

        if not self.groq_model.strip():
            raise ValueError("GROQ_MODEL is not configured")


@lru_cache
def get_settings() -> Settings:
    return Settings()
