"""
AIDocumentIndexer - Encryption Service
=======================================

Provides encryption utilities for securely storing API keys
and other sensitive configuration data.

Uses Fernet symmetric encryption with a key derived from
an environment variable.
"""

import os
import base64
import hashlib
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
import structlog

logger = structlog.get_logger(__name__)


def _get_encryption_key() -> bytes:
    """
    Get or generate the encryption key.

    Uses ENCRYPTION_KEY from environment, or derives one from SECRET_KEY.
    Falls back to a default key for development (NOT SECURE FOR PRODUCTION).
    """
    # First, try explicit encryption key
    encryption_key = os.getenv("ENCRYPTION_KEY")
    if encryption_key:
        # If it's already a valid Fernet key (44 chars base64), use it
        if len(encryption_key) == 44:
            return encryption_key.encode()
        # Otherwise, derive a key from it
        return base64.urlsafe_b64encode(
            hashlib.sha256(encryption_key.encode()).digest()
        )

    # Fall back to SECRET_KEY
    secret_key = os.getenv("SECRET_KEY")
    if secret_key:
        return base64.urlsafe_b64encode(
            hashlib.sha256(secret_key.encode()).digest()
        )

    # Development fallback (NOT SECURE)
    logger.warning(
        "No ENCRYPTION_KEY or SECRET_KEY set. Using development default. "
        "SET ENCRYPTION_KEY IN PRODUCTION!"
    )
    return base64.urlsafe_b64encode(
        hashlib.sha256(b"development-only-key-change-in-production").digest()
    )


# Global Fernet instance
_fernet: Optional[Fernet] = None


def _get_fernet() -> Fernet:
    """Get or create the Fernet instance."""
    global _fernet
    if _fernet is None:
        _fernet = Fernet(_get_encryption_key())
    return _fernet


def encrypt_value(plaintext: str) -> str:
    """
    Encrypt a plaintext string.

    Args:
        plaintext: The string to encrypt

    Returns:
        Base64-encoded encrypted string
    """
    if not plaintext:
        return ""

    try:
        fernet = _get_fernet()
        encrypted = fernet.encrypt(plaintext.encode())
        return encrypted.decode()
    except Exception as e:
        logger.error("Encryption failed", error=str(e))
        raise ValueError(f"Encryption failed: {e}")


def decrypt_value(encrypted: str) -> str:
    """
    Decrypt an encrypted string.

    Args:
        encrypted: Base64-encoded encrypted string

    Returns:
        Decrypted plaintext string
    """
    if not encrypted:
        return ""

    try:
        fernet = _get_fernet()
        decrypted = fernet.decrypt(encrypted.encode())
        return decrypted.decode()
    except InvalidToken:
        logger.error("Decryption failed - invalid token or key mismatch")
        raise ValueError("Decryption failed - invalid token or key mismatch")
    except Exception as e:
        logger.error("Decryption failed", error=str(e))
        raise ValueError(f"Decryption failed: {e}")


def mask_api_key(api_key: str, visible_chars: int = 4) -> str:
    """
    Mask an API key for safe display.

    Args:
        api_key: The API key to mask
        visible_chars: Number of characters to show at start and end

    Returns:
        Masked string like "sk-xxxx...xxxx"
    """
    if not api_key:
        return ""

    if len(api_key) <= visible_chars * 2:
        return "*" * len(api_key)

    return f"{api_key[:visible_chars]}...{api_key[-visible_chars:]}"


def is_encrypted(value: str) -> bool:
    """
    Check if a value appears to be Fernet-encrypted.

    Fernet tokens start with 'gAAA' (base64 of version + timestamp).
    """
    if not value:
        return False

    # Fernet tokens are base64 and start with gAAA
    return value.startswith("gAAA") and len(value) > 100


def rotate_encryption_key(old_key: str, new_key: str, encrypted_values: list[str]) -> list[str]:
    """
    Re-encrypt values with a new key.

    Args:
        old_key: The current encryption key
        new_key: The new encryption key
        encrypted_values: List of encrypted strings

    Returns:
        List of re-encrypted strings
    """
    old_fernet = Fernet(base64.urlsafe_b64encode(
        hashlib.sha256(old_key.encode()).digest()
    ))
    new_fernet = Fernet(base64.urlsafe_b64encode(
        hashlib.sha256(new_key.encode()).digest()
    ))

    re_encrypted = []
    for encrypted in encrypted_values:
        if encrypted:
            decrypted = old_fernet.decrypt(encrypted.encode())
            re_encrypted.append(new_fernet.encrypt(decrypted).decode())
        else:
            re_encrypted.append("")

    return re_encrypted


def generate_encryption_key() -> str:
    """
    Generate a new random encryption key.

    Returns:
        A URL-safe base64-encoded 32-byte key
    """
    return Fernet.generate_key().decode()
