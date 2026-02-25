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

# ---------------------------------------------------------------------------
# Modal app URL — single source of truth for all integration tests
# Override with MODAL_APP_URL env var when pointing at a different deployment.
# ---------------------------------------------------------------------------
DEFAULT_MODAL_URL = "https://julianghadially--codeevolver-fastapi-app-dev.modal.run"


def get_modal_app_url() -> str:
    """Return the Modal app URL from env or the default."""
    return os.getenv("MODAL_APP_URL", DEFAULT_MODAL_URL)


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
