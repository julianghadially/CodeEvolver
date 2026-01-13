"""MongoDB connection management using Motor (async driver)."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from ..config import settings

_client: AsyncIOMotorClient | None = None
_database: AsyncIOMotorDatabase | None = None


def get_database() -> AsyncIOMotorDatabase:
    """Get the database instance. Must be called after lifespan startup."""
    if _database is None:
        raise RuntimeError("Database not initialized. Ensure lifespan is used.")
    return _database


@asynccontextmanager
async def lifespan(app) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager for database connection."""
    global _client, _database

    _client = AsyncIOMotorClient(settings.mongodb_url)
    _database = _client[settings.database_name]

    yield

    _client.close()
    _client = None
    _database = None
