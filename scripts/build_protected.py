#!/usr/bin/env python3
"""
AIDocumentIndexer - Protected Build Script
===========================================

Creates a protected distribution of AIDocumentIndexer for client deployment.
Uses PyArmor for code obfuscation and optionally Nuitka for compilation.

Usage:
    # Basic protected build (PyArmor only)
    python scripts/build_protected.py

    # Full protected build (PyArmor + Nuitka)
    python scripts/build_protected.py --compile

    # Build with specific license provider
    python scripts/build_protected.py --license-provider keygen

    # Build for specific platform
    python scripts/build_protected.py --platform linux

Environment Variables:
    LICENSE_SIGNING_KEY: Key for signing offline licenses (required for offline mode)
    PYARMOR_LICENSE: PyArmor license key (for commercial use)

Output:
    dist/protected/   - Protected Python files
    dist/release/     - Complete release package
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
DIST_DIR = PROJECT_ROOT / "dist"
PROTECTED_DIR = DIST_DIR / "protected"
RELEASE_DIR = DIST_DIR / "release"


def check_dependencies():
    """Check if required tools are installed."""
    missing = []

    # Check PyArmor
    try:
        result = subprocess.run(
            ["pyarmor", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            missing.append("pyarmor")
    except FileNotFoundError:
        missing.append("pyarmor")

    if missing:
        print("Missing required tools:")
        for tool in missing:
            print(f"  - {tool}")
        print("\nInstall with:")
        print("  pip install pyarmor")
        print("  # For Nuitka compilation (optional):")
        print("  pip install nuitka")
        return False

    return True


def clean_dist():
    """Clean distribution directories."""
    print("Cleaning distribution directories...")

    for dir_path in [PROTECTED_DIR, RELEASE_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True)


def obfuscate_with_pyarmor(args):
    """Obfuscate Python code with PyArmor."""
    print("\nObfuscating code with PyArmor...")

    # PyArmor configuration
    pyarmor_config = {
        "obf_mod": 1,  # Obfuscate module
        "obf_code": 1,  # Obfuscate code objects
        "wrap_mode": 1,  # Wrap mode for better protection
        "advanced_mode": 0,  # Advanced mode (requires license)
        "restrict_mode": 2,  # Restrict mode (can't be imported by unobfuscated code)
    }

    # Files/directories to exclude from obfuscation
    exclude_patterns = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "tests",
        "test_*.py",
        "*_test.py",
        "conftest.py",
    ]

    # Build PyArmor command
    cmd = [
        "pyarmor",
        "gen",
        "--output", str(PROTECTED_DIR),
        "--recursive",
        "--enable-rft",  # Enable runtime features
    ]

    # Add exclusions
    for pattern in exclude_patterns:
        cmd.extend(["--exclude", pattern])

    # Add platform-specific options
    if args.platform:
        cmd.extend(["--platform", args.platform])

    # Add the backend directory
    cmd.append(str(BACKEND_DIR))

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"PyArmor error:\n{result.stderr}")
            return False

        print(result.stdout)
        print("PyArmor obfuscation complete!")
        return True

    except Exception as e:
        print(f"PyArmor failed: {e}")
        return False


def compile_with_nuitka(args):
    """Compile to native binary with Nuitka (optional)."""
    print("\nCompiling with Nuitka...")

    # Check if Nuitka is installed
    try:
        subprocess.run(["python", "-m", "nuitka", "--version"], capture_output=True)
    except Exception:
        print("Nuitka not installed. Skipping compilation.")
        print("Install with: pip install nuitka")
        return True  # Not a failure, just skip

    # Nuitka command for the main module
    cmd = [
        "python", "-m", "nuitka",
        "--standalone",
        "--onefile",
        "--output-dir=" + str(DIST_DIR / "compiled"),
        "--include-package=backend",
        "--include-package=uvicorn",
        "--include-package=fastapi",
        "--include-package=sqlalchemy",
        "--include-package=pydantic",
        "--enable-plugin=anti-bloat",
        "--nofollow-import-to=pytest",
        "--nofollow-import-to=tests",
        str(PROTECTED_DIR / "backend" / "api" / "main.py"),
    ]

    print(f"Running: {' '.join(cmd[:5])}...")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            print(f"Nuitka error:\n{result.stderr}")
            return False

        print("Nuitka compilation complete!")
        return True

    except subprocess.TimeoutExpired:
        print("Nuitka compilation timed out (>1 hour)")
        return False
    except Exception as e:
        print(f"Nuitka failed: {e}")
        return False


def create_license_config(args):
    """Create license configuration file."""
    print("\nCreating license configuration...")

    config_content = f"""# AIDocumentIndexer License Configuration
# Generated: {datetime.now().isoformat()}

# License Provider (cryptolens, keygen, self_hosted, offline)
LICENSE_PROVIDER={args.license_provider}

# License Server URL (for self_hosted)
# LICENSE_SERVER_URL=https://license.yourcompany.com

# License API Key (for server-side validation)
# LICENSE_API_KEY=your-api-key

# Product ID
PRODUCT_ID=aidocindexer

# Grace Period (hours) for offline operation
LICENSE_GRACE_PERIOD_HOURS=72

# Development Mode (set to false in production!)
DEV_MODE=false
"""

    config_path = RELEASE_DIR / "config" / "license.env.example"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content)

    print(f"Created: {config_path}")
    return True


def create_release_package(args):
    """Create the final release package."""
    print("\nCreating release package...")

    # Create release structure
    release_structure = {
        "backend": PROTECTED_DIR / "backend",
        "config": None,  # Will create
        "scripts": None,  # Will create
        "docs": PROJECT_ROOT / "docs",
    }

    # Copy protected backend
    backend_dest = RELEASE_DIR / "backend"
    if (PROTECTED_DIR / "backend").exists():
        shutil.copytree(
            PROTECTED_DIR / "backend",
            backend_dest,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    else:
        # Fallback: copy protected files directly
        shutil.copytree(
            PROTECTED_DIR,
            backend_dest,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    # Copy essential config files
    config_dir = RELEASE_DIR / "config"
    config_dir.mkdir(exist_ok=True)

    for config_file in ["pyproject.toml", ".env.example"]:
        src = PROJECT_ROOT / config_file
        if src.exists():
            shutil.copy(src, config_dir / config_file)

    # Copy deployment scripts
    scripts_dir = RELEASE_DIR / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    # Create deployment script
    deploy_script = scripts_dir / "deploy.sh"
    deploy_script.write_text("""#!/bin/bash
# AIDocumentIndexer Deployment Script

set -e

echo "AIDocumentIndexer Deployment"
echo "============================"

# Check Python version
python3 --version || { echo "Python 3 required"; exit 1; }

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
python -m alembic upgrade head

# Start the server
echo "Starting AIDocumentIndexer..."
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
""")
    deploy_script.chmod(0o755)

    # Copy requirements
    req_src = PROJECT_ROOT / "backend" / "requirements.txt"
    if req_src.exists():
        shutil.copy(req_src, RELEASE_DIR / "requirements.txt")
    else:
        # Generate from pyproject.toml
        pyproject = PROJECT_ROOT / "pyproject.toml"
        if pyproject.exists():
            shutil.copy(pyproject, RELEASE_DIR / "pyproject.toml")

    # Copy docs (if not too large)
    docs_src = PROJECT_ROOT / "docs"
    if docs_src.exists():
        docs_dest = RELEASE_DIR / "docs"
        shutil.copytree(
            docs_src,
            docs_dest,
            ignore=shutil.ignore_patterns("*.md~", ".git"),
        )

    # Create version file
    version_file = RELEASE_DIR / "VERSION"
    version_file.write_text(f"1.0.0-{datetime.now().strftime('%Y%m%d')}\n")

    # Create README for release
    readme = RELEASE_DIR / "README.md"
    readme.write_text(f"""# AIDocumentIndexer - Enterprise Release

Version: 1.0.0-{datetime.now().strftime('%Y%m%d')}
Built: {datetime.now().isoformat()}

## Quick Start

1. Copy `config/license.env.example` to `.env` and configure your license
2. Run `./scripts/deploy.sh`

## License Activation

Set your license key in the environment:

```bash
export LICENSE_KEY=your-license-key
export LICENSE_PROVIDER=keygen  # or cryptolens, self_hosted
```

For offline/air-gapped deployments, contact support for an offline license file.

## Support

For technical support, contact: support@yourcompany.com
""")

    print(f"Release package created: {RELEASE_DIR}")
    return True


def create_archive(args):
    """Create distribution archive."""
    print("\nCreating distribution archive...")

    archive_name = f"aidocindexer-{datetime.now().strftime('%Y%m%d')}"

    if args.platform:
        archive_name += f"-{args.platform}"

    # Create tarball
    archive_path = DIST_DIR / f"{archive_name}.tar.gz"

    shutil.make_archive(
        str(DIST_DIR / archive_name),
        "gztar",
        root_dir=str(RELEASE_DIR.parent),
        base_dir="release",
    )

    print(f"Archive created: {archive_path}")

    # Also create zip for Windows users
    zip_path = DIST_DIR / f"{archive_name}.zip"
    shutil.make_archive(
        str(DIST_DIR / archive_name),
        "zip",
        root_dir=str(RELEASE_DIR.parent),
        base_dir="release",
    )

    print(f"ZIP created: {zip_path}")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build protected AIDocumentIndexer distribution"
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Also compile with Nuitka (slower but more secure)",
    )

    parser.add_argument(
        "--platform",
        choices=["linux", "darwin", "windows"],
        help="Target platform for cross-compilation",
    )

    parser.add_argument(
        "--license-provider",
        choices=["cryptolens", "keygen", "self_hosted", "offline"],
        default="self_hosted",
        help="License provider to configure (default: self_hosted)",
    )

    parser.add_argument(
        "--skip-obfuscation",
        action="store_true",
        help="Skip PyArmor obfuscation (for testing)",
    )

    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean dist directories",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AIDocumentIndexer Protected Build")
    print("=" * 60)

    # Clean
    clean_dist()

    if args.clean_only:
        print("\nClean complete.")
        return 0

    # Check dependencies
    if not args.skip_obfuscation and not check_dependencies():
        return 1

    # Obfuscate
    if not args.skip_obfuscation:
        if not obfuscate_with_pyarmor(args):
            print("\nObfuscation failed!")
            return 1
    else:
        # Copy unobfuscated for testing
        print("\nSkipping obfuscation (test mode)...")
        shutil.copytree(
            BACKEND_DIR,
            PROTECTED_DIR / "backend",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"),
        )

    # Compile (optional)
    if args.compile:
        if not compile_with_nuitka(args):
            print("\nCompilation failed! Continuing with Python files...")

    # Create license config
    if not create_license_config(args):
        return 1

    # Create release package
    if not create_release_package(args):
        return 1

    # Create archive
    if not create_archive(args):
        return 1

    print("\n" + "=" * 60)
    print("Build Complete!")
    print("=" * 60)
    print(f"\nRelease directory: {RELEASE_DIR}")
    print(f"Archives: {DIST_DIR}/*.tar.gz, {DIST_DIR}/*.zip")
    print("\nNext steps:")
    print("1. Test the release package locally")
    print("2. Configure license settings in config/license.env.example")
    print("3. Distribute to clients")

    return 0


if __name__ == "__main__":
    sys.exit(main())
