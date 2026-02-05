from __future__ import annotations
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    LLM_ENABLED: bool = False
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TIMEOUT_S: float = 12.0

    class Config:
        env_prefix = ""
        case_sensitive = False

settings = Settings()
