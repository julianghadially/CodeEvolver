"""Application configuration using pydantic-settings."""

import base64
from typing import Optional, Literal
from pydantic import field_validator
from pydantic_settings import BaseSettings
import os

def determine_environment() -> Literal['local', 'dev', 'prod']:
    """Determine the current environment."""
    print(f"[INFO] Determining environment from APP_MODE: {os.getenv('APP_MODE','APP_MODE not found')}")
    app_mode: Literal['local', 'dev', 'prod'] = str(os.getenv('APP_MODE', 'prod')).lower()
    if os.getenv('LOCATION', '') == 'local':
        app_mode = 'local'
    return app_mode

def determine_api_url() -> str:
    """Determine the API URL based on the environment."""
    app_mode = determine_environment()
    if app_mode in ['dev', 'local']:
        return 'https://julianghadially--codeevolver-fastapi-app-dev.modal.run'
    elif app_mode == 'prod':
        return 'https://julianghadially--codeevolver-fastapi-app.modal.run'
    else:
        print(f"[WARNING] Invalid app mode: {app_mode}")
        return 'https://julianghadially--codeevolver-fastapi-app-dev.modal.run'

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    # Priority: CODEEVOLVER_MONGODB_URL > MONGO_KEY > localhost fallback
    mongo_key: Optional[str] = os.environ.get("MONGO_KEY")
    mongodb_url: str = f"mongodb+srv://julianghadially:{mongo_key}@ce1.lbnamk9.mongodb.net/?appName=ce1" if os.environ.get("MONGO_KEY") else ""
    database_name: str = "codeevolver"
    
    # Workspace (overridden to /workspaces when running on Modal)
    workspace_root: str = "/tmp/codeevolver/workspaces"
    
    # Modal settings
    modal_app_name: str = "codeevolver"
    sandbox_timeout: int = 600  # 10 minutes
    sandbox_cpu: int = 2
    sandbox_memory: int = 4096  # MB
    
    # API Keys (for LLM providers - injected into sandbox)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # GitHub App authentication (for private repositories)
    github_app_id: str = "2671751"
    github_app_private_key: Optional[str] = None  # PEM format, can be base64 encoded or raw

    # GEPA optimization settings
    # This timeout is used for: Modal function, sandbox, and JWT TTL
    gepa_job_timeout: int = 43200  # 12 hours (in seconds)
    gepa_optimization_cpu: int = 4
    gepa_optimization_memory: int = 8192  # MB

    # JWT / callback settings (for GEPA sandbox → FastAPI communication)
    jwt_secret: Optional[str] = None       # CODEEVOLVER_JWT_SECRET (HS256 key)
    app_mode: Literal['local', 'dev', 'prod'] = determine_environment()
    callback_url: str = determine_api_url()                # CODEEVOLVER_CALLBACK_URL (Modal web endpoint)

    model_config = {"env_prefix": "CODEEVOLVER_"}
    
    @field_validator('github_app_private_key')
    @classmethod
    def decode_private_key_if_base64(cls, v: Optional[str]) -> Optional[str]:
        """Decode base64 encoded private key if needed."""
        if not v:
            return v
        
        # Check if already in PEM format
        if 'BEGIN' in v and ('PRIVATE KEY' in v or 'RSA PRIVATE KEY' in v):
            return v
        
        # Try to decode as base64 (common for environment variables)
        try:
            decoded = base64.b64decode(v).decode('utf-8')
            # Verify decoded value is PEM format
            if 'BEGIN' in decoded and ('PRIVATE KEY' in decoded or 'RSA PRIVATE KEY' in decoded):
                return decoded
            # If decoded but not PEM, it might be a different encoding or invalid
            # Return original and let jwt library handle the error
        except (base64.binascii.Error, UnicodeDecodeError):
            # If base64 decoding fails, assume it's already in PEM format
            pass
        except Exception:
            # Other exceptions - return original
            pass
        
        return v


settings = Settings()
