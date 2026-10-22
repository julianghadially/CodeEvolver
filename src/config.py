"""Application configuration using pydantic-settings."""

import base64
import secrets
from pathlib import Path
from typing import Optional, Literal
from pydantic import field_validator
from pydantic_settings import BaseSettings
import os


_cached_environment: Literal['local', 'dev', 'prod'] | None = None

def determine_environment() -> Literal['local', 'dev', 'prod']:
    """Determine the current environment. Result is cached after first call."""
    global _cached_environment
    if _cached_environment is not None:
        return _cached_environment
    app_mode: Literal['local', 'dev', 'prod'] = str(os.getenv('APP_MODE', 'prod')).lower()
    if os.getenv('LOCATION', '') == 'local':
        app_mode = 'local'
    _cached_environment = app_mode
    print(f"[CONFIG] Environment: {app_mode}")
    return app_mode


def env_var(name: str, dev_suffix: str = "_DEV") -> str | None:
    """Load the right env var based on APP_MODE.

    In dev/local mode: tries NAME_DEV first, falls back to NAME.
    In prod mode: uses NAME directly.
    """
    if determine_environment() in ('dev', 'local'):
        val = os.environ.get(f"{name}{dev_suffix}")
        if val is not None:
            return val
    return os.environ.get(name)


def determine_database_name() -> str:
    """Return 'codeevolver' for prod, 'codeevolver_dev' for dev/local."""
    if determine_environment() in ('dev', 'local'):
        return 'codeevolver_dev'
    return 'codeevolver'


def determine_sandbox_mode() -> Literal['cloud', 'local']:
    """Return 'local' for a local deploy, else 'cloud' (the hosted portal API).

    Only a local deploy launches optimizations; the cloud API serves the
    portal (Clerk auth, Atlas) and imported run results.

    Resolution order:
      1. Explicit SANDBOX_MODE env var ('local', or 'cloud'; the legacy
         value 'modal' is read as 'cloud').
      2. APP_MODE=local (or LOCATION=local) implies sandbox_mode='local'.
      3. Default 'cloud'.
    """
    explicit = os.getenv('SANDBOX_MODE', '').lower()
    if explicit == 'local':
        return 'local'
    if explicit in ('cloud', 'modal'):
        return 'cloud'
    if determine_environment() == 'local':
        return 'local'
    return 'cloud'


def determine_strategy() -> str:
    """Return the optimizer strategy (method) to run.

    Read from the `STRATEGY` env var; defaults to 'freeform' when unset. Case is
    normalized to lower. Unknown values are passed through as-is — the
    launcher validates against the known method set and raises a clear error
    (see `src/launchers/local_launcher.py::_ENTRY_SCRIPT_REL`)."""
    return os.getenv('STRATEGY', 'freeform').strip().lower() or 'freeform'


def _default_mongo_url() -> str:
    """Return the Mongo connection string for the active sandbox mode.

    Local deploys connect to a Mongo Community instance on the host
    (default mongodb://localhost:27017). The cloud deploy uses the Atlas
    connection assembled from MONGO_USER, MONGO_KEY and MONGO_HOST (each read
    through `env_var`, so dev uses the `_DEV` variants).
    """
    if determine_sandbox_mode() == 'local':
        return os.getenv('LOCAL_MONGO_URI', 'mongodb://localhost:27017')
    key, user, host = env_var("MONGO_KEY"), env_var("MONGO_USER"), env_var("MONGO_HOST")
    if not (key and user and host):
        return ""
    return f"mongodb+srv://{user}:{key}@{host}/"


API_URLS = {
    "local": "http://localhost:8080",
    "dev": "https://dev-api.codeevolver.ai",
    "prod": "https://api.codeevolver.ai",
}

FRONTEND_URLS = {
    "local": "http://localhost:3000",
    "dev": "https://dev.codeevolver.ai",
    "prod": "https://codeevolver.ai",
}


def get_api_base_url() -> str:
    """Get the API base URL for the current environment (including localhost for local)."""
    return API_URLS[determine_environment()]


def determine_api_url() -> str:
    """Determine the API URL based on the environment.

    Reads from the central `API_URLS` dict so local mode hits the local
    uvicorn instance (used by the optimizer's `JobCallbacks` HTTP client),
    and dev/prod hit their respective cloud URLs.
    """
    app_mode = determine_environment()
    url = API_URLS.get(app_mode)
    if url is None:
        print(f"[WARNING] Invalid app mode: {app_mode}")
        return API_URLS['dev']
    return url


# Minimum Python version allowed for the client sandbox image. Floor is
# set by claude-agent-sdk (>=3.10); anything at or above this tuple is
# accepted, so we do not need to edit this when 3.13/3.14 ship.
MIN_PYTHON_SANDBOX_VERSION: tuple[int, int] = (3, 10)


def validate_python_sandbox_version(value: str) -> str:
    """Validate a sandbox `python_version` string against the floor.

    Accepts only MAJOR.MINOR format (e.g. "3.10", "3.12"). Returns the
    value unchanged on success; raises `ValueError` on any failure.
    Used by the `/optimize` request schema.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"python_version must be a string, got {type(value).__name__}"
        )
    parts = value.split(".")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(
            f"python_version must be in MAJOR.MINOR format (e.g. '3.12'), got {value!r}"
        )
    major, minor = int(parts[0]), int(parts[1])
    if (major, minor) < MIN_PYTHON_SANDBOX_VERSION:
        floor_str = ".".join(str(x) for x in MIN_PYTHON_SANDBOX_VERSION)
        raise ValueError(
            f"python_version {value!r} is below the minimum supported "
            f"version {floor_str} (required by claude-agent-sdk)"
        )
    return value


def _default_jwt_secret() -> Optional[str]:
    """Return the JWT secret for the current environment.

    In dev/prod we require `CODEEVOLVER_JWT_SECRET` to be set explicitly
    (cross-process: Cloud Run mints, the sandbox sends, Cloud Run validates,
    so a stable shared value is required). In local mode the same uvicorn
    process both mints and validates, so we auto-generate an ephemeral
    in-memory secret when the env var is unset — keeps callbacks working
    without forcing the operator to set anything in their `.env`.
    """
    configured = env_var("CODEEVOLVER_JWT_SECRET", dev_suffix="")
    if configured:
        return configured
    if determine_sandbox_mode() == 'local':
        return secrets.token_urlsafe(32)
    return configured


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Environment-specific secrets use the env_var() helper which appends
    _DEV suffix in dev/local mode. Both prod and dev vars coexist in the
    same environment; APP_MODE selects which set to use.
    """
    # PARAMETER SETTINGS
    
    # Fixed comparison-set size default (engine 3): drawn from the trainset
    # with seed 42; counted against the eval-row budget. Overridable per run
    # via the /optimize request's `comparison_set_size` field.
    comparison_set_size: int = 50
    
    # Soft wall-clock deadline for a job, used for the optimizer deadline and
    # the callback JWT TTL.
    #
    # This is a ceiling on TOTAL wall clock, not on the optimization loop: the
    # loop stops `TIMEOUT_BUFFER_SECONDS` early (optimizer_loop.py) so that
    # finalization — the seed + final-primary held-out evals, then the
    # meta-reflection and run-integrity agents — completes inside this number.
    local_job_timeout: int = 432000  # 120 hours (in seconds)

    # CONFIG SETTINGS
    # Database
    mongo_key: Optional[str] = env_var("MONGO_KEY")
    mongodb_url: str = _default_mongo_url()
    database_name: str = determine_database_name()

    # Deployment mode — single source of truth for cloud-vs-local. Local mode
    # launches optimizations, skips Clerk (single operator) and points Mongo at
    # localhost; cloud mode is the hosted portal API and does not launch runs.
    sandbox_mode: Literal["cloud", "local"] = determine_sandbox_mode()

    # Optimizer strategy (method) — selected by the STRATEGY env var, default
    # 'freeform'. Resolved once here and injected as `method` in the launcher
    # dispatch (src/launchers/__init__.py), so no /optimize request field or
    # per-call threading is needed. An explicit `method` kwarg still wins.
    strategy: str = determine_strategy()

    # Local-deploy paths. Only consulted when sandbox_mode == "local".
    # The engine source root is NOT a setting — it's auto-detected from
    # `src/engine/paths.py`'s own `__file__`, so the operator can clone
    # CodeEvolver anywhere on disk without configuring this. Use
    # `src.engine.paths.engine_root()` to read it.
    local_history_root: Path = Path.home() / ".codeevolver" / "history"
    local_log_root: Path = Path.home() / ".codeevolver" / "logs"


    # API Keys (for LLM providers - injected into sandbox)
    anthropic_api_key: Optional[str] = env_var("CODEEVOLVER_ANTHROPIC_API_KEY", dev_suffix="")
    openai_api_key: Optional[str] = env_var("CODEEVOLVER_OPENAI_API_KEY", dev_suffix="")

    # GitHub App authentication (for private repositories)
    github_app_id: Optional[str] = env_var("CODEEVOLVER_GITHUB_APP_ID")
    github_app_private_key: Optional[str] = env_var("CODEEVOLVER_GITHUB_APP_PRIVATE_KEY")


    # JWT / callback settings (for optimizer sandbox → FastAPI communication).
    # Local mode auto-generates an ephemeral secret — see `_default_jwt_secret`.
    jwt_secret: Optional[str] = _default_jwt_secret()
    app_mode: Literal['local', 'dev', 'prod'] = determine_environment()
    callback_url: str = determine_api_url()

    # Clerk authentication
    clerk_secret: Optional[str] = env_var("CODEEVOLVER_CLERK_SECRET")
    clerk_wh_signature: Optional[str] = env_var("CODEEVOLVER_WH_SIGNATURE")
    clerk_jwks_url: Optional[str] = env_var("CODEEVOLVER_CLERK_JWKS_URL")
    clerk_issuer: Optional[str] = env_var("CODEEVOLVER_CLERK_ISSUER")

    # API key encryption (Fernet symmetric key)
    snoopy_key: Optional[str] = env_var("CODEEVOLVER_SNOOPY_KEY")

    model_config = {}

    @field_validator('sandbox_mode', mode='before')
    @classmethod
    def normalize_sandbox_mode(cls, v):
        """Read the legacy SANDBOX_MODE value 'modal' as 'cloud'."""
        if isinstance(v, str) and v.strip().lower() == 'modal':
            return 'cloud'
        return v.strip().lower() if isinstance(v, str) else v

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
        except (base64.binascii.Error, UnicodeDecodeError):
            pass
        except Exception:
            pass

        return v


settings = Settings()
