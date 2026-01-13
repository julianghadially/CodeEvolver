"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "codeevolver"
    workspace_root: str = "/tmp/codeevolver/workspaces"

    model_config = {"env_prefix": "CODEEVOLVER_"}


settings = Settings()
