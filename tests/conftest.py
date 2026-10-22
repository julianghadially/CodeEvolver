"""Test configuration and fixtures."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app

@pytest.fixture
def mock_mongodb():
    """Mock MongoDB for tests that don't need real database."""
    mock_db = MagicMock()
    mock_db.repos = AsyncMock()
    mock_db.programs = AsyncMock()
    return mock_db


@pytest.fixture
async def async_client():
    """Create an async test client for FastAPI."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
