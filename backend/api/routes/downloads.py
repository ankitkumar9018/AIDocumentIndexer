"""Downloads API routes for serving CLI tools and other downloadable assets."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse

router = APIRouter(prefix="/downloads", tags=["downloads"])

# Path to the CLI tool source
CLI_SOURCE_PATH = Path(__file__).parent.parent.parent.parent / "desktop-clients" / "cli"


def create_cli_zip() -> Path:
    """Create a zip file of the CLI tool for download."""
    if not CLI_SOURCE_PATH.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CLI tool not found")

    # Create temp directory for the zip
    temp_dir = Path(tempfile.gettempdir()) / "mandala-downloads"
    temp_dir.mkdir(exist_ok=True)

    zip_path = temp_dir / "mandala-sync-cli"

    # Create the zip file (shutil adds .zip extension)
    if zip_path.with_suffix(".zip").exists():
        # Check if source is newer than zip
        source_mtime = max(
            f.stat().st_mtime
            for f in CLI_SOURCE_PATH.rglob("*")
            if f.is_file() and "__pycache__" not in str(f)
        )
        zip_mtime = zip_path.with_suffix(".zip").stat().st_mtime
        if source_mtime <= zip_mtime:
            return zip_path.with_suffix(".zip")

    # Create new zip
    shutil.make_archive(
        str(zip_path),
        'zip',
        CLI_SOURCE_PATH.parent,
        CLI_SOURCE_PATH.name
    )

    return zip_path.with_suffix(".zip")


@router.get("/cli")
async def download_cli():
    """Download the Mandala Sync CLI tool as a zip file."""
    try:
        zip_path = create_cli_zip()
        return FileResponse(
            path=zip_path,
            filename="mandala-sync-cli.zip",
            media_type="application/zip"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create download: {str(e)}")


@router.get("/cli/info")
async def get_cli_info():
    """Get information about the CLI tool."""
    if not CLI_SOURCE_PATH.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="CLI tool not found")

    # Read version from pyproject.toml
    pyproject_path = CLI_SOURCE_PATH / "pyproject.toml"
    version = "1.0.0"
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        for line in content.split("\n"):
            if line.strip().startswith("version"):
                version = line.split("=")[1].strip().strip('"')
                break

    return {
        "name": "mandala-sync",
        "version": version,
        "platforms": ["macOS", "Windows", "Linux"],
        "requirements": ["Python 3.8+"],
        "download_url": "/api/v1/downloads/cli"
    }


@router.get("/available")
async def list_available_downloads():
    """List all available downloads."""
    downloads = []

    # CLI Tool
    if CLI_SOURCE_PATH.exists():
        downloads.append({
            "id": "cli",
            "name": "Mandala Sync CLI",
            "description": "Command-line tool for syncing files to Mandala",
            "type": "cli",
            "platforms": ["macOS", "Windows", "Linux"],
            "download_url": "/api/v1/downloads/cli",
            "info_url": "/api/v1/downloads/cli/info",
            "available": True
        })

    # Desktop App (placeholder - not yet available)
    downloads.append({
        "id": "desktop",
        "name": "Mandala Desktop App",
        "description": "Full-featured desktop application with system tray",
        "type": "desktop",
        "platforms": ["macOS", "Windows", "Linux"],
        "download_url": None,
        "available": False
    })

    # Browser Extension (placeholder - not yet available)
    downloads.append({
        "id": "browser-extension",
        "name": "Mandala Browser Extension",
        "description": "Browser extension to upload downloaded files",
        "type": "extension",
        "platforms": ["Chrome", "Firefox", "Edge"],
        "download_url": None,
        "available": False
    })

    return {"downloads": downloads}
