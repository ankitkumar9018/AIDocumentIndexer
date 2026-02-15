#!/usr/bin/env python3
"""
Cross-platform setup script for AIDocumentIndexer.
Detects OS and handles dependencies, services, and configuration accordingly.

This script will:
1. Check system dependencies (python3, node, npm)
2. Stop any existing services on ports 8000, 3000
3. Install Python dependencies with UV (or pip fallback)
4. Install Node.js/frontend dependencies with npm
5. Setup Ollama and pull required models
6. Create environment files if missing
7. Run database migrations
8. Start backend and frontend services

Usage:
    python scripts/setup.py [options]

Options:
    --skip-services     Skip starting backend/frontend services
    --skip-ollama       Skip Ollama installation and model pulling
    --skip-redis        Skip Redis installation and startup
    --skip-celery       Skip Celery worker startup
    --pull-optional     Also pull optional Ollama models (mistral, codellama, llama3.3:70b)
    --install-optional  Auto-install optional system deps (ffmpeg, libreoffice, tesseract, etc.)
    --verbose, -v       Enable verbose/debug output
    --log-file PATH     Write logs to specified file

Examples:
    python scripts/setup.py                      # Full setup with all services
    python scripts/setup.py --skip-services     # Setup without starting services
    python scripts/setup.py --skip-redis        # Skip Redis (disables Celery too)
    python scripts/setup.py --pull-optional     # Include optional Ollama models
    python scripts/setup.py --install-optional  # Install ffmpeg, libreoffice, etc.
    python scripts/setup.py --verbose           # Verbose output for debugging

Platform Support:
    - macOS: Uses Homebrew for package installation
    - Linux: Supports apt (Debian/Ubuntu), dnf (Fedora/RHEL), yum (CentOS)
    - Windows: Uses winget for package installation

How it works:
    - Python runs npm/node via subprocess to install frontend packages
    - Python runs uv/pip via subprocess to install backend packages
    - Services are started as background processes
"""

import argparse
import json
import logging
import os
import platform
import secrets
import shutil
import subprocess
import sys
import time
import re  # Required for env modification
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# LOGGING SETUP
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output."""

    COLORS = {
        'DEBUG': '\033[0;36m',     # Cyan
        'INFO': '\033[0;32m',      # Green
        'WARNING': '\033[1;33m',   # Yellow
        'ERROR': '\033[0;31m',     # Red
        'CRITICAL': '\033[1;31m',  # Bold Red
    }
    RESET = '\033[0m'

    SYMBOLS = {
        'DEBUG': '•',
        'INFO': '✓',
        'WARNING': '⚠',
        'ERROR': '✗',
        'CRITICAL': '✗✗',
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        symbol = self.SYMBOLS.get(record.levelname, '→')

        # Format: [TIMESTAMP] SYMBOL MESSAGE
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted = f"{color}[{timestamp}] {symbol} {record.getMessage()}{self.RESET}"
        return formatted


class FileFormatter(logging.Formatter):
    """Formatter for log files (no colors)."""

    def format(self, record):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"[{timestamp}] [{record.levelname}] {record.getMessage()}"


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging with both console and file handlers."""
    logger = logging.getLogger('setup')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FileFormatter())
        logger.addHandler(file_handler)

    return logger


# Global logger instance
logger: logging.Logger = None

# ============================================================================
# PLATFORM DETECTION
# ============================================================================

def get_platform() -> str:
    """Detect the current platform."""
    system = platform.system().lower()
    if system == 'darwin':
        return 'macos'
    elif system == 'windows':
        return 'windows'
    elif system == 'linux':
        return 'linux'
    return 'unknown'


def enable_windows_colors():
    """Enable ANSI color codes on Windows."""
    if platform.system() == 'Windows':
        os.system('color')
        # Also try to enable VT100 support
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_dependencies(deps_file: str = 'scripts/dependencies.json') -> Dict:
    """Load dependency configuration from JSON file."""
    try:
        with open(deps_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Dependencies file not found: {deps_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in dependencies file: {e}")
        sys.exit(1)


def check_command_exists(command: str) -> bool:
    """Check if a command exists in PATH."""
    return shutil.which(command) is not None


def run_command(
    cmd: List[str],
    capture: bool = False,
    check: bool = True,
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
    env: Optional[Dict[str, str]] = None
) -> Tuple[int, str, str]:
    """
    Run a shell command with proper error handling.

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    logger.debug(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
            cwd=cwd,
            timeout=timeout,
            env=env
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        logger.debug(f"Command failed with code {e.returncode}: {e.stderr}")
        return e.returncode, e.stdout or '', e.stderr or ''
    except subprocess.TimeoutExpired:
        logger.warning(f"Command timed out: {' '.join(cmd)}")
        return -1, '', 'Timeout expired'
    except FileNotFoundError:
        logger.debug(f"Command not found: {cmd[0]}")
        return 1, '', f"Command not found: {cmd[0]}"


def get_version(command: str) -> Optional[str]:
    """Get version string for a command."""
    version_flags = ['--version', '-v', '-V', 'version']

    for flag in version_flags:
        code, stdout, _ = run_command([command, flag], check=False, timeout=10)
        if code == 0 and stdout:
            # Extract first line containing version info
            for line in stdout.strip().split('\n'):
                if any(c.isdigit() for c in line):
                    return line.strip()
    return None


# ============================================================================
# SERVICE MANAGEMENT
# ============================================================================

def kill_process_on_port(port: int) -> bool:
    """Kill process running on a specific port (cross-platform)."""
    system = get_platform()
    killed = False

    logger.debug(f"Checking for processes on port {port}")

    if system == 'windows':
        # Windows: Use netstat and taskkill
        code, stdout, _ = run_command(
            ['cmd', '/c', f'netstat -ano | findstr :{port}'],
            check=False
        )
        if code == 0 and stdout:
            for line in stdout.splitlines():
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        logger.info(f"Killing process {pid} on port {port}")
                        run_command(['taskkill', '/F', '/PID', pid], check=False)
                        killed = True
    else:
        # macOS/Linux: Use lsof
        code, stdout, _ = run_command(['lsof', '-ti', f':{port}'], check=False)
        if code == 0 and stdout.strip():
            pids = stdout.strip().split('\n')
            for pid in pids:
                if pid.strip():
                    logger.info(f"Killing process {pid} on port {port}")
                    run_command(['kill', '-9', pid.strip()], check=False)
                    killed = True

    return killed


def clear_python_cache(project_root: Optional[Path] = None):
    """Clear Python bytecode cache (__pycache__ and .pyc files).

    This ensures fresh code is loaded after code changes, especially
    when restarting services. Useful when Python bytecode may be stale.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    backend_dir = project_root / 'backend'
    try:
        # Remove __pycache__ directories
        for pycache_dir in backend_dir.rglob('__pycache__'):
            if pycache_dir.is_dir():
                shutil.rmtree(pycache_dir, ignore_errors=True)

        # Remove .pyc files (in case any exist outside __pycache__)
        for pyc_file in backend_dir.rglob('*.pyc'):
            pyc_file.unlink(missing_ok=True)

        logger.info("Cleared Python bytecode cache")
    except Exception as e:
        logger.warning(f"Could not fully clear Python cache: {e}")


def is_service_running(port: int) -> bool:
    """Check if a service is running on a port."""
    system = get_platform()

    if system == 'windows':
        code, stdout, _ = run_command(
            ['cmd', '/c', f'netstat -ano | findstr :{port}'],
            check=False
        )
        return code == 0 and f':{port}' in stdout
    else:
        code, _, _ = run_command(['lsof', '-ti', f':{port}'], check=False)
        return code == 0


def start_background_process(
    cmd: List[str],
    log_file: Optional[Path] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Optional[subprocess.Popen]:
    """Start a process in the background (cross-platform).

    If log_file is provided, stdout/stderr are written there.
    Otherwise, stdout/stderr go to DEVNULL.

    Returns the Popen object, or None on failure.
    """
    system = get_platform()
    try:
        if system == 'windows':
            return subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            if log_file:
                f = open(log_file, 'w')
                proc = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                f.close()  # Child process has its own fd after fork
                return proc
            else:
                return subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
    except Exception as e:
        logger.error(f"Failed to start process {cmd[0]}: {e}")
        return None


def get_dev_environment(project_root: Path, load_env_file: bool = False) -> Dict[str, str]:
    """Build environment dict for local dev services (backend, Celery).

    Sets PYTHONPATH, DEV_MODE, LOCAL_MODE, model source checks, and
    forces the SQLite database path for local mode.

    If load_env_file is True, also loads variables from backend/.env
    (needed for Celery workers to pick up LLM/API key settings).
    """
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)

    # Load .env file first so dev mode overrides below take precedence
    if load_env_file:
        env_file = project_root / 'backend' / '.env'
        if env_file.exists():
            try:
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, _, value = line.partition('=')
                            key = key.strip()
                            value = value.strip()
                            # Don't override dev mode settings set below
                            if key not in ('DEV_MODE', 'LOCAL_MODE', 'DATABASE_URL'):
                                env[key] = value
                logger.debug("Loaded .env file for service environment")
            except Exception as e:
                logger.warning(f"Could not load .env file: {e}")

    # Force dev mode settings
    env['DEV_MODE'] = 'true'
    env['LOCAL_MODE'] = 'true'
    env['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    env['FASTEMBED_DISABLE_MODEL_SOURCE_CHECK'] = '1'

    # Force SQLite database path for local mode
    db_dir = project_root / 'backend' / 'data'
    db_dir.mkdir(parents=True, exist_ok=True)
    env['DATABASE_URL'] = f'sqlite:///{db_dir / "aidocindexer.db"}'

    return env


# ============================================================================
# DEPENDENCY CHECKING
# ============================================================================

def check_system_dependencies(deps: Dict) -> Tuple[bool, List[str]]:
    """
    Check system dependencies.

    Returns:
        Tuple of (all_required_found, list_of_missing_required)
    """
    logger.info("Checking system dependencies...")
    missing_required = []
    system = get_platform()

    # Check required dependencies
    for dep in deps['system']['required']:
        if check_command_exists(dep):
            version = get_version(dep)
            version_str = f" ({version})" if version else ""
            logger.info(f"Found: {dep}{version_str}")
        else:
            logger.error(f"Missing required: {dep}")
            missing_required.append(dep)

    # Check optional dependencies (new nested structure)
    optional_deps = deps['system'].get('optional', {})
    descriptions = deps['system'].get('description', {})
    install_commands = deps['system'].get('install_commands', {})

    # Get platform-specific install commands
    if system == 'macos':
        platform_cmds = install_commands.get('macos', {})
    elif system == 'linux':
        # Try to detect apt vs dnf
        if check_command_exists('apt'):
            platform_cmds = install_commands.get('linux_apt', {})
        else:
            platform_cmds = install_commands.get('linux_dnf', {})
    else:
        platform_cmds = install_commands.get('windows', {})

    # Flatten optional deps from categories
    all_optional = []
    if isinstance(optional_deps, dict):
        for category, dep_list in optional_deps.items():
            if isinstance(dep_list, list):
                all_optional.extend(dep_list)
    elif isinstance(optional_deps, list):
        all_optional = optional_deps

    for dep in all_optional:
        # Handle special cases for dependency checking
        check_cmd = dep
        is_python_pkg = False

        if dep == 'poppler':
            check_cmd = 'pdftotext'  # Check for pdftotext which is part of poppler
        elif dep == 'redis-server':
            check_cmd = 'redis-server'
        elif dep == 'ray':
            # Ray is a Python package, check via pip/uv
            is_python_pkg = True

        # Check Python packages differently
        if is_python_pkg:
            # Try to import the package
            code, _, _ = run_command(['python3', '-c', f'import {dep}'], check=False, timeout=10)
            found = (code == 0)
        else:
            found = check_command_exists(check_cmd)

        if found:
            if is_python_pkg:
                # Get version for Python package
                code, stdout, _ = run_command(
                    ['python3', '-c', f'import {dep}; print({dep}.__version__)'],
                    check=False, timeout=10
                )
                version_str = f" ({stdout.strip()})" if code == 0 and stdout.strip() else ""
            else:
                version = get_version(check_cmd)
                version_str = f" ({version})" if version else ""
            logger.info(f"Found: {dep}{version_str}")
        else:
            desc = descriptions.get(dep, "")
            install_cmd = platform_cmds.get(dep, "")
            msg = f"Missing optional: {dep}"
            if desc:
                msg += f" - {desc}"
            logger.warning(msg)
            if install_cmd:
                logger.debug(f"  Install with: {install_cmd}")

    return len(missing_required) == 0, missing_required


def install_optional_dependencies(deps: Dict) -> Dict[str, bool]:
    """
    Install optional system dependencies based on platform.

    Args:
        deps: Dependencies configuration from dependencies.json

    Returns:
        Dict mapping dependency name to success status
    """
    logger.info("Installing optional system dependencies...")
    system = get_platform()
    results = {}

    optional_deps = deps['system'].get('optional', {})
    descriptions = deps['system'].get('description', {})
    install_commands = deps['system'].get('install_commands', {})

    # Get platform-specific install commands
    if system == 'macos':
        platform_cmds = install_commands.get('macos', {})
        pkg_manager = 'brew'
        sudo_prefix = []
    elif system == 'linux':
        if check_command_exists('apt'):
            platform_cmds = install_commands.get('linux_apt', {})
            pkg_manager = 'apt'
            sudo_prefix = ['sudo']
        elif check_command_exists('dnf'):
            platform_cmds = install_commands.get('linux_dnf', {})
            pkg_manager = 'dnf'
            sudo_prefix = ['sudo']
        elif check_command_exists('yum'):
            platform_cmds = install_commands.get('linux_dnf', {})  # yum syntax similar to dnf
            pkg_manager = 'yum'
            sudo_prefix = ['sudo']
        else:
            logger.warning("No supported package manager found (apt/dnf/yum)")
            return results
    else:  # Windows
        platform_cmds = install_commands.get('windows', {})
        pkg_manager = 'winget'
        sudo_prefix = []

    # Flatten optional deps from categories
    all_optional = []
    if isinstance(optional_deps, dict):
        for category, dep_list in optional_deps.items():
            if isinstance(dep_list, list):
                all_optional.extend(dep_list)
    elif isinstance(optional_deps, list):
        all_optional = optional_deps

    for dep in all_optional:
        # Check if already installed
        check_cmd = dep
        if dep == 'poppler':
            check_cmd = 'pdftotext'
        elif dep == 'ray':
            # Ray is a Python package, skip system install
            continue

        if check_command_exists(check_cmd):
            logger.info(f"Already installed: {dep}")
            results[dep] = True
            continue

        install_cmd_str = platform_cmds.get(dep, "")
        if not install_cmd_str:
            logger.warning(f"No install command for {dep} on {system}")
            results[dep] = False
            continue

        desc = descriptions.get(dep, "")
        logger.info(f"Installing {dep}" + (f" ({desc})" if desc else ""))

        # Parse and execute the install command
        try:
            if system == 'windows':
                if install_cmd_str.startswith('winget'):
                    # Winget command
                    parts = install_cmd_str.split()
                    code, stdout, stderr = run_command(parts, check=False, timeout=300)
                elif install_cmd_str.startswith('Download'):
                    # Manual download required
                    logger.warning(f"  Manual installation required for {dep} on Windows:")
                    logger.warning(f"  {install_cmd_str}")
                    results[dep] = False
                    continue
                else:
                    code, stdout, stderr = run_command(
                        ['cmd', '/c', install_cmd_str],
                        check=False,
                        timeout=300
                    )
            elif system == 'macos':
                if install_cmd_str.startswith('brew install --cask'):
                    # Cask install (GUI apps like LibreOffice)
                    parts = install_cmd_str.split()
                    code, stdout, stderr = run_command(parts, check=False, timeout=600)
                elif install_cmd_str.startswith('brew'):
                    parts = install_cmd_str.split()
                    code, stdout, stderr = run_command(parts, check=False, timeout=300)
                else:
                    code, stdout, stderr = run_command(
                        ['bash', '-c', install_cmd_str],
                        check=False,
                        timeout=300
                    )
            else:  # Linux
                if install_cmd_str.startswith(('apt', 'dnf', 'yum')):
                    # Add sudo and -y flag for non-interactive install
                    parts = install_cmd_str.split()
                    if parts[0] in ('apt', 'dnf', 'yum'):
                        # Insert sudo at beginning
                        cmd = sudo_prefix + parts
                        # Add -y flag if not present
                        if '-y' not in cmd:
                            cmd.insert(cmd.index('install') + 1 if 'install' in cmd else 2, '-y')
                        code, stdout, stderr = run_command(cmd, check=False, timeout=300)
                    else:
                        code, stdout, stderr = run_command(
                            ['bash', '-c', install_cmd_str],
                            check=False,
                            timeout=300
                        )
                else:
                    code, stdout, stderr = run_command(
                        ['bash', '-c', install_cmd_str],
                        check=False,
                        timeout=300
                    )

            if code == 0:
                logger.info(f"Successfully installed {dep}")
                results[dep] = True
            else:
                logger.error(f"Failed to install {dep}: {stderr}")
                results[dep] = False

        except Exception as e:
            logger.error(f"Error installing {dep}: {e}")
            results[dep] = False

    # Install WeasyPrint dependencies for PDF generation
    weasyprint_deps = deps['system'].get('weasyprint_deps', {})
    weasyprint_cmd = None
    if system == 'macos':
        weasyprint_cmd = weasyprint_deps.get('macos')
    elif system == 'linux':
        if check_command_exists('apt'):
            weasyprint_cmd = weasyprint_deps.get('linux_apt')
        else:
            weasyprint_cmd = weasyprint_deps.get('linux_dnf')

    if weasyprint_cmd and not weasyprint_cmd.startswith('Use'):
        logger.info("Installing WeasyPrint dependencies (for PDF generation)...")
        try:
            if system == 'linux':
                parts = weasyprint_cmd.split()
                cmd = sudo_prefix + parts
                if '-y' not in cmd and 'install' in cmd:
                    cmd.insert(cmd.index('install') + 1, '-y')
                code, _, stderr = run_command(cmd, check=False, timeout=120)
            else:
                code, _, stderr = run_command(
                    ['bash', '-c', weasyprint_cmd] if system != 'windows' else weasyprint_cmd.split(),
                    check=False,
                    timeout=120
                )
            if code == 0:
                logger.info("WeasyPrint dependencies installed")
            else:
                logger.warning(f"WeasyPrint deps install failed: {stderr}")
        except Exception as e:
            logger.warning(f"Could not install WeasyPrint deps: {e}")

    return results


# ============================================================================
# OLLAMA SETUP
# ============================================================================

def install_ollama() -> bool:
    """Install Ollama based on platform."""
    system = get_platform()

    if check_command_exists('ollama'):
        version = get_version('ollama')
        logger.info(f"Ollama already installed ({version})")
        return True

    logger.info("Installing Ollama...")

    if system == 'macos':
        if check_command_exists('brew'):
            code, _, stderr = run_command(['brew', 'install', 'ollama'], check=False)
            if code != 0:
                logger.error(f"Failed to install Ollama via Homebrew: {stderr}")
                return False
        else:
            logger.warning("Homebrew not found. Please install Ollama manually from https://ollama.com/download")
            return False
    elif system == 'linux':
        logger.info("Running Ollama install script...")
        code, _, stderr = run_command(
            ['bash', '-c', 'curl -fsSL https://ollama.com/install.sh | sh'],
            check=False
        )
        if code != 0:
            logger.error(f"Failed to install Ollama: {stderr}")
            return False
    elif system == 'windows':
        # Try winget first
        if check_command_exists('winget'):
            logger.info("Installing Ollama via winget...")
            code, _, stderr = run_command(
                ['winget', 'install', '--id', 'Ollama.Ollama', '-e', '--accept-package-agreements', '--accept-source-agreements'],
                check=False,
                timeout=300
            )
            if code == 0:
                logger.info("Ollama installed via winget")
            else:
                logger.warning(f"winget install failed: {stderr}")
                logger.warning("Please install Ollama manually from https://ollama.com/download")
                return False
        else:
            logger.warning("winget not found. Please install Ollama manually from https://ollama.com/download")
            return False

    return check_command_exists('ollama')


def start_ollama_service() -> bool:
    """Start Ollama service if not running."""
    system = get_platform()

    # Check if Ollama is already running by trying to list models
    code, _, _ = run_command(['ollama', 'list'], check=False, timeout=10)
    if code == 0:
        logger.info("Ollama service is already running")
        return True

    logger.info("Starting Ollama service...")

    proc = start_background_process(['ollama', 'serve'])
    if proc is None:
        return False

    # Wait for service to start
    logger.debug("Waiting for Ollama service to start...")
    for i in range(10):
        time.sleep(1)
        code, _, _ = run_command(['ollama', 'list'], check=False, timeout=5)
        if code == 0:
            logger.info("Ollama service started successfully")
            return True
        logger.debug(f"Waiting... ({i+1}/10)")

    logger.error("Ollama service failed to start within timeout")
    return False


def get_installed_ollama_models() -> List[str]:
    """Get list of already installed Ollama models."""
    code, stdout, _ = run_command(['ollama', 'list'], check=False, timeout=10)
    if code != 0:
        return []

    models = []
    for line in stdout.strip().split('\n')[1:]:  # Skip header line
        if line.strip():
            # Format: NAME    ID    SIZE    MODIFIED
            parts = line.split()
            if parts:
                models.append(parts[0])  # Model name is first column
    return models


def pull_ollama_models(deps: Dict, include_optional: bool = False) -> Dict[str, bool]:
    """
    Pull required Ollama models, skipping those already installed.

    Args:
        deps: Dependencies configuration
        include_optional: If True, also pull optional models

    Returns:
        Dict mapping model name to success status
    """
    models_config = deps.get('ollama', {}).get('models', {})
    optional_models = deps.get('ollama', {}).get('optional_models', {})
    descriptions = deps.get('ollama', {}).get('description', {})
    results = {}

    # Get already installed models to skip re-pulling
    installed_models = get_installed_ollama_models()
    logger.debug(f"Already installed models: {installed_models}")

    def is_model_installed(model_name: str) -> bool:
        """Check if model is installed (handles version tags)."""
        # Model name might be 'llama3.2:latest' or just 'llama3.2'
        base_name = model_name.split(':')[0]
        for installed in installed_models:
            if installed.startswith(base_name):
                return True
        return False

    # Required models (text and embedding - these must succeed)
    for category in ['text', 'embedding']:
        for model in models_config.get(category, []):
            if is_model_installed(model):
                logger.info(f"Model already installed: {model}")
                results[model] = True
                continue

            desc = descriptions.get(model, "")
            logger.info(f"Pulling {category} model: {model}")
            if desc:
                logger.debug(f"  Purpose: {desc}")
            code, stdout, stderr = run_command(
                ['ollama', 'pull', model],
                check=False,
                timeout=600  # 10 minutes for large models
            )
            if code == 0:
                logger.info(f"Successfully pulled: {model}")
                results[model] = True
            else:
                logger.error(f"Failed to pull {model}: {stderr}")
                results[model] = False

    # Vision models (optional - graceful failure)
    for model in models_config.get('vision', []):
        if is_model_installed(model):
            logger.info(f"Vision model already installed: {model}")
            results[model] = True
            continue

        desc = descriptions.get(model, "")
        logger.info(f"Pulling vision model: {model}")
        if desc:
            logger.debug(f"  Purpose: {desc}")
        code, stdout, stderr = run_command(
            ['ollama', 'pull', model],
            check=False,
            timeout=600
        )
        if code == 0:
            logger.info(f"Successfully pulled: {model}")
            results[model] = True
        else:
            logger.warning(f"Vision model {model} not available - will use cloud fallback")
            results[model] = False

    # Optional models (only if requested)
    if include_optional:
        logger.info("Pulling optional models...")
        for category, models in optional_models.items():
            if isinstance(models, list):
                for model in models:
                    if is_model_installed(model):
                        logger.info(f"Optional model already installed: {model}")
                        results[model] = True
                        continue

                    desc = descriptions.get(model, "")
                    logger.info(f"Pulling optional {category} model: {model}")
                    if desc:
                        logger.debug(f"  Purpose: {desc}")
                    code, stdout, stderr = run_command(
                        ['ollama', 'pull', model],
                        check=False,
                        timeout=1200  # 20 minutes for large models like 70b
                    )
                    if code == 0:
                        logger.info(f"Successfully pulled: {model}")
                        results[model] = True
                    else:
                        logger.warning(f"Optional model {model} not available")
                        results[model] = False

    return results


# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def setup_python_environment(project_root: Path) -> bool:
    """Setup Python environment with UV or pip."""
    backend_dir = project_root / 'backend'

    if not backend_dir.exists():
        logger.error(f"Backend directory not found: {backend_dir}")
        return False

    # Check for UV first (preferred)
    if check_command_exists('uv'):
        logger.info("Using UV for Python dependency management")
        code, stdout, stderr = run_command(
            ['uv', 'sync'],
            cwd=str(backend_dir),
            check=False,
            timeout=300
        )
        if code != 0:
            logger.error(f"UV sync failed: {stderr}")
            return False
        logger.info("Python dependencies synced successfully")
        return True

    # Install UV if not found
    logger.info("UV not found, installing...")
    system = get_platform()

    if system == 'windows':
        code, _, stderr = run_command(
            ['powershell', '-c', 'irm https://astral.sh/uv/install.ps1 | iex'],
            check=False
        )
    else:
        code, _, stderr = run_command(
            ['bash', '-c', 'curl -LsSf https://astral.sh/uv/install.sh | sh'],
            check=False
        )

    if code != 0:
        logger.error(f"Failed to install UV: {stderr}")
        logger.info("Trying pip as fallback...")

        # Fallback to pip
        code, _, stderr = run_command(
            ['pip', 'install', '-r', 'requirements.txt'],
            cwd=str(backend_dir),
            check=False
        )
        if code != 0:
            logger.error(f"Pip install failed: {stderr}")
            return False
        return True

    # Retry with UV after installation
    # Need to reload PATH
    if system != 'windows':
        os.environ['PATH'] = f"{os.environ['HOME']}/.cargo/bin:" + os.environ['PATH']

    code, stdout, stderr = run_command(
        ['uv', 'sync'],
        cwd=str(backend_dir),
        check=False,
        timeout=300
    )
    if code != 0:
        logger.error(f"UV sync failed: {stderr}")
        return False

    logger.info("Python dependencies synced successfully")
    return True


def setup_node_environment(project_root: Path) -> bool:
    """Setup Node.js environment."""
    frontend_dir = project_root / 'frontend'

    if not frontend_dir.exists():
        logger.error(f"Frontend directory not found: {frontend_dir}")
        return False

    # Check if node_modules exists and package-lock.json is newer
    node_modules = frontend_dir / 'node_modules'
    package_lock = frontend_dir / 'package-lock.json'

    if node_modules.exists() and package_lock.exists():
        # Use npm ci for faster, cleaner installs when lock file exists
        logger.info("Installing Node.js dependencies (npm ci)...")
        code, stdout, stderr = run_command(
            ['npm', 'ci'],
            cwd=str(frontend_dir),
            check=False,
            timeout=300
        )
        if code != 0:
            # Fallback to npm install if ci fails
            logger.debug("npm ci failed, falling back to npm install...")
            code, stdout, stderr = run_command(
                ['npm', 'install'],
                cwd=str(frontend_dir),
                check=False,
                timeout=300
            )
    else:
        logger.info("Installing Node.js dependencies (npm install)...")
        code, stdout, stderr = run_command(
            ['npm', 'install'],
            cwd=str(frontend_dir),
            check=False,
            timeout=300
        )

    if code != 0:
        logger.error(f"npm install failed: {stderr}")
        return False

    logger.info("Node.js dependencies installed successfully")
    return True


def setup_environment_files(project_root: Path, deps: Dict) -> bool:
    """Create environment files and force APP_ENV=development."""
    env_config = deps.get('environment', {})

    # Backend .env
    backend_env = project_root / env_config.get('backend_env_file', 'backend/.env')
    example_env = project_root / env_config.get('example_file', '.env.example')

    if not backend_env.exists():
        if example_env.exists():
            shutil.copy(example_env, backend_env)
            logger.info(f"Created {backend_env} from {example_env}")
        else:
            logger.warning(f"No example env file found at {example_env}")
            # Create a minimal .env if even the example is missing
            backend_env.write_text("APP_ENV=development\n")
    
    # Force APP_ENV=development
    if backend_env.exists():
        try:
            content = backend_env.read_text()
            # If APP_ENV exists, replace it; otherwise, append it
            if "APP_ENV=" in content:
                content = re.sub(r'APP_ENV=.*', 'APP_ENV=development', content)
            else:
                content += "\nAPP_ENV=development\n"
            
            backend_env.write_text(content)
            logger.info("Set APP_ENV=development in backend/.env")
        except Exception as e:
            logger.error(f"Failed to modify backend/.env: {e}")
            return False

    # Frontend .env.local
    frontend_env = project_root / env_config.get('frontend_env_file', 'frontend/.env.local')

    if not frontend_env.exists():
        secret = secrets.token_urlsafe(32)
        # BACKEND_URL uses 127.0.0.1 for server-side stability, NEXT_PUBLIC suffixes with /api/v1
        env_content = f"""# Auto-generated by setup script
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
BACKEND_URL=http://127.0.0.1:8000/api/v1
NEXTAUTH_URL=http://localhost:3000
AUTH_SECRET={secret}
AUTH_TRUST_HOST=true
"""
        frontend_env.write_text(env_content)
        logger.info(f"Created {frontend_env}")
    else:
        logger.debug(f"Frontend env file already exists: {frontend_env}")

    return True


# ============================================================================
# DATABASE MIGRATIONS
# ============================================================================

def run_migrations(project_root: Path) -> bool:
    """
    Run database migrations.

    Handles common scenarios:
    - Fresh database: runs all migrations
    - Tables already exist: stamps to head (marks as up-to-date)
    - Partial migration state: attempts recovery
    """
    logger.info("Running database migrations...")

    # Determine the alembic command prefix
    # Note: alembic.ini is in project root, so run from there
    if check_command_exists('uv'):
        alembic_cmd = ['uv', 'run', 'alembic']
    else:
        alembic_cmd = ['python', '-m', 'alembic']

    # First, try to run migrations normally
    code, stdout, stderr = run_command(
        alembic_cmd + ['upgrade', 'head'],
        cwd=str(project_root),
        check=False,
        timeout=120
    )

    if code == 0:
        logger.info("Database migrations completed successfully")
        return True

    # Check if error is due to tables already existing
    if 'already exists' in stderr.lower():
        logger.warning("Tables already exist - stamping database to head")

        # Stamp to head to mark all migrations as applied
        stamp_code, stamp_stdout, stamp_stderr = run_command(
            alembic_cmd + ['stamp', 'head'],
            cwd=str(project_root),
            check=False,
            timeout=30
        )

        if stamp_code == 0:
            logger.info("Database stamped to head successfully")
            return True
        else:
            logger.error(f"Failed to stamp database: {stamp_stderr}")
            return False

    # Check for missing revision errors (broken migration chain)
    if 'keyerror' in stderr.lower() or 'not present' in stderr.lower():
        logger.warning("Migration chain issue detected - attempting to check current state")

        # Check current migration state
        current_code, current_stdout, current_stderr = run_command(
            alembic_cmd + ['current'],
            cwd=str(project_root),
            check=False,
            timeout=30
        )

        if current_code == 0 and 'head' in current_stdout.lower():
            logger.info("Database is already at head")
            return True

        logger.error(f"Migration chain broken. Manual intervention may be required.")
        logger.error(f"Try: cd backend && uv run alembic stamp head")

    logger.error(f"Migration failed: {stderr}")
    return False


# ============================================================================
# SERVICE STARTUP
# ============================================================================

def start_services(project_root: Path, deps: Dict) -> bool:
    """Start backend and frontend services, forcing 0.0.0.0 host binding."""
    services_config = deps.get('services', {})

    frontend_dir = project_root / 'frontend'

    # Create log directory
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)

    # Start backend from project root with PYTHONPATH set
    # Force --host 0.0.0.0 to prevent address resolution issues
    logger.info("Starting backend service on 0.0.0.0:8000...")
    backend_log = log_dir / 'backend.log'
    backend_env = get_dev_environment(project_root)

    # Get number of workers from environment (default 1 for dev, increase for production)
    # Set UVICORN_WORKERS=4 for higher concurrency (supports ~200 concurrent users)
    uvicorn_workers = int(os.environ.get('UVICORN_WORKERS', '1'))

    uvicorn_cmd = ['uv', 'run', '--project', 'backend', 'uvicorn', 'backend.api.main:app', '--host', '0.0.0.0', '--port', '8000']
    if uvicorn_workers > 1:
        uvicorn_cmd.extend(['--workers', str(uvicorn_workers)])
        logger.info(f"Starting backend with {uvicorn_workers} workers for higher concurrency")

    proc = start_background_process(uvicorn_cmd, log_file=backend_log, cwd=str(project_root), env=backend_env)
    if proc is None:
        return False
    logger.info(f"Backend started (logs: {backend_log})")

    # Start frontend
    logger.info("Starting frontend service...")
    frontend_log = log_dir / 'frontend.log'

    proc = start_background_process(['npm', 'run', 'dev'], log_file=frontend_log, cwd=str(frontend_dir))
    if proc is None:
        return False
    logger.info(f"Frontend started (logs: {frontend_log})")

    # Wait for services to start (backend may take longer due to Ray initialization)
    logger.info("Waiting for services to initialize...")

    # Check services with retry (backend can take 60-90s with service registry + table checks)
    backend_running = False
    frontend_running = False
    max_retries = 36  # 36 * 5s = 180s max wait (backend has heavy imports + table checks)
    for i in range(max_retries):
        time.sleep(5)
        if not backend_running:
            backend_running = is_service_running(services_config.get('backend', {}).get('port', 8000))
        if not frontend_running:
            frontend_running = is_service_running(services_config.get('frontend', {}).get('port', 3000))
        if backend_running and frontend_running:
            break
        if i < max_retries - 1:
            logger.debug(f"Waiting for services... ({i+1}/{max_retries})")

    if backend_running:
        logger.info("Backend is running on http://localhost:8000")
    else:
        logger.warning("Backend may not have started correctly - check logs")

    if frontend_running:
        logger.info("Frontend is running on http://localhost:3000")
    else:
        logger.warning("Frontend may not have started correctly - check logs")

    return True


# ============================================================================
# REDIS AND CELERY SERVICES
# ============================================================================

def is_redis_running() -> bool:
    """Check if Redis server is running."""
    code, _, _ = run_command(['redis-cli', 'ping'], check=False, timeout=5)
    return code == 0


def start_redis(project_root: Path) -> bool:
    """Start Redis server if not already running."""
    if not check_command_exists('redis-server'):
        logger.warning("Redis not installed - async task processing will be disabled")
        return False

    if is_redis_running():
        logger.info("Redis is already running")
        return True

    logger.info("Starting Redis server...")
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    redis_log = log_dir / 'redis.log'

    proc = start_background_process(['redis-server'], log_file=redis_log)
    if proc is None:
        return False

    # Wait for Redis to start
    for i in range(10):
        time.sleep(0.5)
        if is_redis_running():
            logger.info(f"Redis started successfully (logs: {redis_log})")
            return True
        logger.debug(f"Waiting for Redis... ({i+1}/10)")

    logger.error("Redis failed to start within timeout")
    return False


def start_celery(project_root: Path, deps: Dict) -> bool:
    """Start Celery worker if Redis is available."""
    if not is_redis_running():
        logger.warning("Redis not running - cannot start Celery worker")
        return False

    if not check_command_exists('uv'):
        logger.warning("UV not available - cannot start Celery worker")
        return False

    # Clear bytecode cache to ensure fresh code is loaded
    clear_python_cache(project_root)

    logger.info("Starting Celery worker...")
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    celery_log = log_dir / 'celery.log'

    # Get celery start command from deps or use default
    celery_config = deps.get('services', {}).get('celery', {})
    start_cmd = celery_config.get('start_command', 'uv run celery -A backend.services.task_queue worker --loglevel=info')
    cmd_parts = start_cmd.split()

    # Build environment with .env file loaded (Celery needs LLM/API key settings)
    celery_env = get_dev_environment(project_root, load_env_file=True)

    proc = start_background_process(cmd_parts, log_file=celery_log, cwd=str(project_root), env=celery_env)
    if proc is None:
        return False

    logger.info(f"Celery worker started (logs: {celery_log})")
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def print_banner():
    """Print setup script banner."""
    banner = """
╔════════════════════════════════════════════════════════════╗
║             AIDocumentIndexer Setup Script                 ║
╠════════════════════════════════════════════════════════════╣
║  Platform: {platform:<46} ║
║  Time: {time:<50} ║
╚════════════════════════════════════════════════════════════╝
""".format(
        platform=get_platform().upper(),
        time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    print(banner)


def print_summary(results: Dict, args=None):
    """Print setup summary."""
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)

    for step, status in results.items():
        symbol = "✓" if status else "✗"
        color = "\033[0;32m" if status else "\033[0;31m"
        reset = "\033[0m"
        print(f"{color}{symbol}{reset} {step}")

    print("=" * 60)

    all_success = all(results.values())
    if all_success:
        print("\n\033[0;32mSetup completed successfully!\033[0m")
        print("\nServices running:")
        print("  - Backend:  http://localhost:8000/docs")
        print("  - Frontend: http://localhost:3000")
        if not (args and args.skip_ollama):
            print("  - Ollama:   http://localhost:11434")
        if not (args and args.skip_redis):
            print("  - Redis:    localhost:6379")
        if not (args and args.skip_celery):
            print("  - Celery:   Worker running")
        print("\nLogs available in: logs/")
        print("\nLogin: admin@example.com / admin123")
    else:
        print("\n\033[0;31mSetup completed with errors. Check the log for details.\033[0m")

    print()


def stop_all_services(deps: Dict) -> Dict[str, bool]:
    """Stop all running services.

    Phase 98: Enhanced to kill orphaned processes more thoroughly,
    preventing memory leaks from zombie processes after crashes.
    """
    results = {}
    services_config = deps.get('services', {})

    # Stop services by port
    service_ports = {
        'backend': 8000,
        'frontend': 3000,
        'ollama': 11434,
        'redis': 6379,
    }

    for service, port in service_ports.items():
        if kill_process_on_port(port):
            logger.info(f"Stopped {service} (port {port})")
            results[service] = True
        else:
            logger.debug(f"{service} was not running on port {port}")
            results[service] = True  # Not an error if it wasn't running

    # Stop Celery workers — use SIGKILL to ensure all instances die
    system = get_platform()
    if system != 'windows':
        run_command(['pkill', '-9', '-f', 'celery'], check=False)
        logger.info("Stopped all Celery processes")
        results['celery'] = True

    # Stop Ray gracefully first, then force-kill stragglers
    run_command(['ray', 'stop', '--force'], check=False)
    time.sleep(2)
    for ray_pattern in ['ray::', 'raylet', 'gcs_server', 'ray.dashboard', 'ray._private']:
        run_command(['pkill', '-9', '-f', ray_pattern], check=False)
    logger.info("Stopped all Ray processes")

    # Clean Ray temporary directory (accumulates GBs of stale session data)
    ray_temp = Path(os.path.expanduser("~")) / ".ray_temp"
    if ray_temp.exists():
        import shutil
        try:
            shutil.rmtree(ray_temp, ignore_errors=True)
            logger.info("Cleaned Ray temporary directory (~/.ray_temp)")
        except Exception as e:
            logger.warning(f"Could not clean Ray temp dir: {e}")
    results['ray'] = True

    # Delete stale 0-byte database files from old code versions
    data_dir = Path(__file__).resolve().parent.parent / 'backend' / 'data'
    if data_dir.exists():
        for stale_db in ['aidoc.db', 'aidocumentindexer.db']:
            stale_path = data_dir / stale_db
            if stale_path.exists() and stale_path.stat().st_size == 0:
                stale_path.unlink()
                logger.info(f"Deleted stale empty database: {stale_db}")

    # Phase 98: Kill orphaned processes that may leak memory
    if system != 'windows':
        orphan_patterns = [
            # Backend-related processes
            'uvicorn.*backend',
            'python.*backend',
            'gunicorn.*backend',
            # ChromaDB processes (can accumulate after crashes)
            'chroma',
            'chromadb',
            # Embedding model processes (fastembed, sentence-transformers)
            'fastembed',
            'sentence.transformers',
            # Frontend-related processes
            'node.*frontend',
            'next.*dev',
            # Ollama model processes (separate from server)
            'ollama.*run',
        ]

        for pattern in orphan_patterns:
            code, _, _ = run_command(['pkill', '-f', pattern], check=False)
            if code == 0:
                logger.debug(f"Killed orphaned processes matching: {pattern}")

        results['orphaned_processes'] = True

        # Give processes time to terminate gracefully
        time.sleep(1)

        # Force kill any remaining stubborn processes (SIGKILL)
        for pattern in orphan_patterns[:4]:  # Only force-kill critical backend processes
            run_command(['pkill', '-9', '-f', pattern], check=False)

    return results


def main():
    """Main setup function."""
    global logger

    # Parse arguments
    parser = argparse.ArgumentParser(description='AIDocumentIndexer Setup Script')
    parser.add_argument('command', nargs='?', default='start', choices=['start', 'stop', 'restart'],
                       help='Command: start (default), stop, or restart')
    parser.add_argument('--skip-services', action='store_true', help='Skip starting services')
    parser.add_argument('--skip-ollama', action='store_true', help='Skip Ollama setup')
    parser.add_argument('--skip-redis', action='store_true', help='Skip Redis installation and startup')
    parser.add_argument('--skip-celery', action='store_true', help='Skip Celery worker startup (requires Redis)')
    parser.add_argument('--pull-optional', action='store_true', help='Also pull optional Ollama models (mistral, codellama, etc.)')
    parser.add_argument('--install-optional', action='store_true', help='Auto-install optional system dependencies (ffmpeg, libreoffice, etc.)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--log-file', type=str, help='Path to log file')
    args = parser.parse_args()

    # If skip-redis, also skip celery (celery requires redis)
    if args.skip_redis:
        args.skip_celery = True

    # Enable Windows colors
    enable_windows_colors()

    # Determine project root (assuming script is in scripts/ directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Setup logging
    log_file = args.log_file or str(project_root / 'logs' / 'setup.log')
    Path(log_file).parent.mkdir(exist_ok=True)
    logger = setup_logging(verbose=args.verbose, log_file=log_file)

    # Print banner
    print_banner()

    logger.info(f"Project root: {project_root}")
    logger.info(f"Log file: {log_file}")

    # Track results
    results = {}

    # Load dependencies
    deps_file = project_root / 'scripts' / 'dependencies.json'
    if not deps_file.exists():
        # Fallback if scripts dir doesn't exist or file is missing
        deps = {"environment": {}, "system": {"required": ["python3", "node", "npm"]}, "services": {"backend": {"port": 8000}, "frontend": {"port": 3000}}}
    else:
        deps = load_dependencies(str(deps_file))
        logger.debug(f"Loaded dependencies from {deps_file}")

    # Handle stop command
    if args.command == 'stop':
        logger.info("Stopping all services...")
        stop_results = stop_all_services(deps)
        print("\n" + "=" * 60)
        print("SERVICES STOPPED")
        print("=" * 60)
        for service, stopped in stop_results.items():
            symbol = "✓" if stopped else "✗"
            color = "\033[0;32m" if stopped else "\033[0;31m"
            reset = "\033[0m"
            print(f"{color}{symbol}{reset} {service}")
        print("=" * 60)
        print("\n\033[0;32mAll services stopped.\033[0m\n")
        sys.exit(0)

    # Handle restart command (log message before proceeding with start flow)
    if args.command == 'restart':
        logger.info("Restarting all services...")

    # 1. Check system dependencies
    deps_ok, missing = check_system_dependencies(deps)
    results['System dependencies'] = deps_ok
    if not deps_ok:
        logger.error(f"Missing required dependencies: {', '.join(missing)}")
        logger.error("Please install missing dependencies and re-run setup")
        print_summary(results)
        sys.exit(1)

    # 1.5 Install optional dependencies if requested
    if args.install_optional:
        optional_results = install_optional_dependencies(deps)
        success_count = sum(1 for v in optional_results.values() if v)
        total_count = len(optional_results)
        results['Optional dependencies'] = success_count == total_count
        if success_count < total_count:
            failed = [k for k, v in optional_results.items() if not v]
            logger.warning(f"Some optional dependencies failed to install: {', '.join(failed)}")

    # 2. Stop existing services (ports, Celery, Ray) and clear Python cache
    logger.info("Stopping existing services...")
    stop_all_services(deps)
    clear_python_cache(project_root)
    results['Stop existing services'] = True

    # 3. Setup Python environment
    results['Python environment'] = setup_python_environment(project_root)

    # 4. Setup Node environment
    results['Node.js environment'] = setup_node_environment(project_root)

    # 5. Setup Ollama (unless skipped)
    if args.skip_ollama:
        logger.info("Skipping Ollama setup (--skip-ollama)")
        results['Ollama setup'] = True
    else:
        ollama_installed = install_ollama()
        if ollama_installed:
            ollama_started = start_ollama_service()
            if ollama_started:
                model_results = pull_ollama_models(deps, include_optional=args.pull_optional)
                # Consider success if at least text and embedding models are available
                text_models = deps.get('ollama', {}).get('models', {}).get('text', [])
                embed_models = deps.get('ollama', {}).get('models', {}).get('embedding', [])
                required_models = text_models + embed_models
                results['Ollama setup'] = all(model_results.get(m, False) for m in required_models)
            else:
                results['Ollama setup'] = False
        else:
            logger.warning("Ollama not installed - LLM features will use cloud providers")
            results['Ollama setup'] = False

    # 6. Setup environment files
    results['Environment files'] = setup_environment_files(project_root, deps)

    # 7. Run migrations
    results['Database migrations'] = run_migrations(project_root)

    # 8. Start Redis (unless skipped)
    if args.skip_redis:
        logger.info("Skipping Redis (--skip-redis)")
        results['Redis'] = True
    else:
        results['Redis'] = start_redis(project_root)

    # 9. Start Celery (unless skipped, requires Redis)
    if args.skip_celery:
        logger.info("Skipping Celery worker (--skip-celery)")
        results['Celery worker'] = True
    else:
        results['Celery worker'] = start_celery(project_root, deps)

    # 10. Start services (unless skipped)
    if args.skip_services:
        logger.info("Skipping service startup (--skip-services)")
        results['Start services'] = True
    else:
        results['Start services'] = start_services(project_root, deps)

    # Print summary
    print_summary(results, args)

    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()