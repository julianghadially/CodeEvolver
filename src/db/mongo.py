"""MongoDB connection management for both sync (pymongo) and async (motor) drivers.

All database driver imports are lazy so this module can be loaded in
environments without pymongo or motor installed (e.g. the GEPA sandbox).
"""

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator

from ..config import settings, determine_environment

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
    from pymongo import MongoClient
    from pymongo.database import Database

# Async client for FastAPI (managed via lifespan)
_client: "AsyncIOMotorClient | None" = None
_database: "AsyncIOMotorDatabase | None" = None

# Sync clients for Modal sandboxes (lazy initialization)
_sync_client: "MongoClient | None" = None
_async_client: "AsyncIOMotorClient | None" = None


def get_mongo_db(db_name: str = 'ce1', force_prod: bool = False) -> "Database":
    """Get synchronous MongoDB database instance.

    Uses lazy initialization with a global client. Safe for Modal sandboxes
    since each function invocation runs in its own container.

    Args:
        db_name: Database name
        force_prod: If True, skip test suffix even in dev/local

    Returns:
        MongoDB Database instance
    """
    from pymongo import MongoClient

    global _sync_client
    if (determine_environment() in ['local', 'dev'] and '_test' not in db_name) and not force_prod:
        db_name = f"{db_name}_test"
    if _sync_client is None:
        _sync_client = MongoClient(settings.mongodb_url)
    return _sync_client[db_name]


def get_async_mongo_db(db_name: str = 'db_pllm', force_prod: bool = False) -> "AsyncIOMotorDatabase":
    """Get asynchronous MongoDB database instance.

    Uses lazy initialization with a global client. Safe for Modal sandboxes
    since each function invocation runs in its own container.

    Args:
        db_name: Database name
        force_prod: If True, skip test suffix even in dev/local

    Returns:
        MongoDB AsyncIOMotorDatabase instance
    """
    from motor.motor_asyncio import AsyncIOMotorClient

    global _async_client
    if (determine_environment() in ['local', 'dev'] and '_test' not in db_name) and not force_prod:
        db_name = f"{db_name}_test"
    if _async_client is None:
        _async_client = AsyncIOMotorClient(settings.mongodb_url)
    return _async_client[db_name]


def get_database() -> "AsyncIOMotorDatabase":
    """Get the database instance. Must be called after lifespan startup."""
    if _database is None:
        raise RuntimeError("Database not initialized. Ensure lifespan is used.")
    return _database


@asynccontextmanager
async def lifespan(app) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager for database connection."""
    from motor.motor_asyncio import AsyncIOMotorClient

    global _client, _database

    _client = AsyncIOMotorClient(settings.mongodb_url)
    _database = _client[settings.database_name]

    yield

    _client.close()
    _client = None
    _database = None
