"""MongoDB connection management for both sync (pymongo) and async (motor) drivers.

All database driver imports are lazy so this module can be loaded in
environments without pymongo or motor installed (e.g. the optimizer sandbox).

Database naming:
  - prod: 'codeevolver'
  - dev/local: 'codeevolver_dev'
  This is controlled by settings.database_name (set in config.py via determine_database_name).
"""

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncGenerator

from ..config import settings

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
    from pymongo import MongoClient
    from pymongo.database import Database

# Async client for FastAPI (managed via lifespan)
_client: "AsyncIOMotorClient | None" = None
_database: "AsyncIOMotorDatabase | None" = None

# Sync clients for subprocesses (lazy initialization)
_sync_client: "MongoClient | None" = None
_async_client: "AsyncIOMotorClient | None" = None


def get_mongo_db(db_name: str | None = None, **kwargs) -> "Database":
    """Get synchronous MongoDB database instance.

    Uses lazy initialization with a global client.

    Args:
        db_name: Database name override. Defaults to settings.database_name
                 (codeevolver in prod, codeevolver_dev in dev).

    Returns:
        MongoDB Database instance
    """
    from pymongo import MongoClient

    global _sync_client
    if db_name is None:
        db_name = settings.database_name
    if _sync_client is None:
        _sync_client = MongoClient(settings.mongodb_url)
    return _sync_client[db_name]


def get_async_mongo_db(db_name: str | None = None, **kwargs) -> "AsyncIOMotorDatabase":
    """Get asynchronous MongoDB database instance.

    Uses lazy initialization with a global client.

    Args:
        db_name: Database name override. Defaults to settings.database_name
                 (codeevolver in prod, codeevolver_dev in dev).

    Returns:
        MongoDB AsyncIOMotorDatabase instance
    """
    from motor.motor_asyncio import AsyncIOMotorClient

    global _async_client
    if db_name is None:
        db_name = settings.database_name
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

    if not settings.mongodb_url:
        print("[DB] WARNING: No MongoDB URL configured. Database operations will fail.")
        yield
        return

    _client = AsyncIOMotorClient(settings.mongodb_url)
    _database = _client[settings.database_name]

    print(f"[DB] Connected to database: {settings.database_name}")

    # GET /jobs filters by clerk_id and sorts by created_at. Without this index
    # Mongo sorts whole job documents in memory and fails past its 32 MB sort
    # limit once jobs carry large histories. Idempotent; never blocks startup.
    try:
        await _database.jobs.create_index([("clerk_id", 1), ("created_at", -1)])
    except Exception as exc:
        print(f"[DB] WARNING: could not ensure jobs(clerk_id, created_at) index: {exc}")

    yield

    if _client:
        _client.close()
    _client = None
    _database = None
