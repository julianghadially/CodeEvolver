"""Test configuration and fixtures."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app
from src.config import settings

def get_modal_app_url() -> str:
    """Return the Modal app URL for tests.

    Precedence:
    1. MODAL_APP_URL env var (explicit override)
    2. APP_MODE=prod → production URL
    3. Otherwise → dev URL
    """
    env_url = os.getenv("MODAL_APP_URL")
    if env_url:
        return env_url

    app_mode = os.getenv("APP_MODE", "").lower()
    if app_mode == "prod":
        return "https://julianghadially--codeevolver-fastapi-app.modal.run"

    return "https://julianghadially--codeevolver-fastapi-app-dev.modal.run"


@pytest.fixture
def modal_url() -> str:
    """Fixture that provides the Modal app URL for integration tests."""
    return get_modal_app_url()


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory for tests."""
    workspace = tmp_path / "workspaces"
    workspace.mkdir()
    original_root = settings.workspace_root
    settings.workspace_root = str(workspace)
    yield workspace
    settings.workspace_root = original_root


@pytest.fixture
def mock_mongodb():
    """Mock MongoDB for tests that don't need real database."""
    mock_db = MagicMock()
    mock_db.clients = AsyncMock()
    mock_db.programs = AsyncMock()
    return mock_db


@pytest.fixture
async def async_client():
    """Create an async test client for FastAPI."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
