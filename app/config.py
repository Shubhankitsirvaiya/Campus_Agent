"""Environment-backed settings shared by FastAPI and the teaching notebook."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Validate configuration loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "CampusPath Academic Planning and Enrollment Agent"
    environment: str = "development"

    openai_api_key: str = Field(default="", repr=False)
    openai_model: str = "gpt-5-mini"

    langsmith_tracing: bool = False
    langsmith_api_key: str = Field(default="", repr=False)
    langsmith_project: str = "project-13-campuspath-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    academic_db_path: Path = Path("data/academic_operations.db")
    checkpoint_db_path: Path = Path("data/agent_checkpoints.db")
    audit_log_path: Path = Path("data/audit/agent_events.jsonl")

    max_agent_steps: int = Field(default=28, ge=6, le=100)
    max_tool_retries: int = Field(default=3, ge=1, le=10)
    standard_max_credits: int = Field(default=22, ge=12, le=30)

    def absolute_path(self, path: Path) -> Path:
        """Resolve relative storage paths from the repository root."""
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    """Create one settings object per process."""
    return Settings()

