"""API key generation, hashing, and encryption utilities."""

import hashlib
import secrets

from cryptography.fernet import Fernet


def generate_api_key() -> str:
    """Generate a new raw API key (~43 URL-safe characters)."""
    return secrets.token_urlsafe(32)


def hash_api_key(raw_key: str) -> str:
    """SHA-256 hash of the raw API key (used for database lookup)."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def encrypt_api_key(raw_key: str, fernet_key: str) -> str:
    """Fernet-encrypt the raw API key (for recovery/retrieval)."""
    f = Fernet(fernet_key.encode())
    return f.encrypt(raw_key.encode()).decode()


def decrypt_api_key(encrypted: str, fernet_key: str) -> str:
    """Decrypt a Fernet-encrypted API key back to the raw key."""
    f = Fernet(fernet_key.encode())
    return f.decrypt(encrypted.encode()).decode()
