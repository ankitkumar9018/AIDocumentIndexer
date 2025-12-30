"""
Initialize OCR Settings
=======================

Adds OCR configuration settings to the system_settings table.
This script ensures OCR settings exist in the database.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.settings import SettingsService, SettingCategory


async def init_ocr_settings():
    """Initialize OCR settings in the database."""
    print("=" * 60)
    print("Initializing OCR Settings")
    print("=" * 60)

    settings_service = SettingsService()

    # Check if OCR settings already exist
    existing = await settings_service.get_settings_by_category(SettingCategory.OCR)

    if existing:
        print(f"✓ Found {len(existing)} existing OCR settings")
        for key, value in existing.items():
            print(f"  - {key}: {value}")
        print()
        print("OCR settings already initialized.")
        return

    print("Adding OCR settings to database...")

    # OCR settings will be automatically created from DEFAULT_SETTINGS
    # when accessed via the settings service for the first time
    # Let's trigger their creation by getting them
    ocr_settings = {
        "ocr.provider": "paddleocr",
        "ocr.paddle.variant": "server",
        "ocr.paddle.languages": ["en", "de"],
        "ocr.paddle.model_dir": "./data/paddle_models",
        "ocr.paddle.auto_download": True,
        "ocr.tesseract.fallback_enabled": True,
    }

    # Update settings (this will create them if they don't exist)
    for key, value in ocr_settings.items():
        await settings_service.update_setting(key, value)
        print(f"  ✓ Added: {key}")

    print()
    print("=" * 60)
    print("OCR settings initialized successfully!")
    print("=" * 60)

    # Verify
    print()
    print("Verifying OCR settings:")
    verified = await settings_service.get_settings_by_category(SettingCategory.OCR)
    for key, value in verified.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    asyncio.run(init_ocr_settings())
