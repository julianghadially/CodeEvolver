"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "codeevolver"
    
    # Workspace (overridden to /workspaces when running on Modal)
    workspace_root: str = "/tmp/codeevolver/workspaces"
    
    # Modal settings
    modal_app_name: str = "codeevolver-agents"
    sandbox_timeout: int = 600  # 10 minutes
    sandbox_cpu: int = 2
    sandbox_memory: int = 4096  # MB
    
    # API Keys (for LLM providers - injected into sandbox)
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    model_config = {"env_prefix": "CODEEVOLVER_"}


settings = Settings()
