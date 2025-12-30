"""
PaddleOCR Model Migration Script
=================================

Migrates PaddleOCR models from system directory (~/.paddlex) to project directory
(./data/paddle_models) for better portability and version control.

Usage:
    python backend/scripts/migrate_paddle_models.py
"""

import os
import shutil
import sys
from pathlib import Path


def get_model_size(path: Path) -> float:
    """Calculate total size of directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB


def main():
    print("=" * 60)
    print("PaddleOCR Model Migration Script")
    print("=" * 60)
    print()

    # Define paths
    OLD_PATH = Path.home() / ".paddlex"
    NEW_PATH = Path("./data/paddle_models")

    print(f"Old location: {OLD_PATH}")
    print(f"New location: {NEW_PATH}")
    print()

    # Check if old models exist
    if not OLD_PATH.exists():
        print("✗ No existing models found at ~/.paddlex")
        print("  Models will be downloaded on first run")
        print()
        return 0

    print(f"✓ Found existing models at {OLD_PATH}")

    # Calculate size
    model_size = get_model_size(OLD_PATH)
    print(f"  Total size: {model_size:.1f} MB")
    print()

    # Check if new location already has models
    if NEW_PATH.exists() and any(NEW_PATH.iterdir()):
        print(f"⚠ Warning: {NEW_PATH} already exists and is not empty")
        response = input("  Overwrite existing models? (y/N): ")
        if response.lower() != 'y':
            print("  Migration cancelled")
            return 1

    # Create new directory
    print(f"Creating directory: {NEW_PATH}")
    NEW_PATH.mkdir(parents=True, exist_ok=True)

    # Copy models
    print(f"Copying models...")
    try:
        shutil.copytree(OLD_PATH, NEW_PATH, dirs_exist_ok=True)
        print(f"✓ Migration complete!")
        print()

        # Verify migration
        new_size = get_model_size(NEW_PATH)
        print(f"Verification:")
        print(f"  Old location: {model_size:.1f} MB")
        print(f"  New location: {new_size:.1f} MB")

        if abs(model_size - new_size) < 1.0:  # Within 1MB tolerance
            print(f"  ✓ Sizes match - migration successful")
        else:
            print(f"  ⚠ Warning: Size mismatch detected")

        print()
        print(f"Cleanup:")
        print(f"  Old models at {OLD_PATH} can now be safely deleted")
        print(f"  Run: rm -rf {OLD_PATH}")
        print()

        return 0

    except Exception as e:
        print(f"✗ Error during migration: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
