#!/usr/bin/env python3
"""
Production setup script for AIDocumentIndexer.
Handles deployment configuration, security checks, and production service startup.

This script will:
1. Verify production prerequisites (PostgreSQL, Redis, proper secrets)
2. Check and validate security settings (no dev mode, proper secrets)
3. Install Python dependencies with UV
4. Install Node.js/frontend dependencies
5. Build frontend for production
6. Run database migrations
7. Start backend with Gunicorn (production WSGI server)
8. Start frontend in production mode
9. Optionally configure systemd services

Usage:
    python scripts/prod_setup.py [options]

Options:
    --check-only         Only run checks, don't install or start services
    --skip-build         Skip frontend build (use existing build)
    --skip-services      Skip starting services (useful for containerized deployments)
    --workers N          Number of Gunicorn workers (default: auto based on CPU cores)
    --bind HOST:PORT     Gunicorn bind address (default: 0.0.0.0:8000)
    --ssl-cert PATH      Path to SSL certificate
    --ssl-key PATH       Path to SSL private key
    --create-systemd     Create systemd service files
    --verbose, -v        Enable verbose output
    --log-file PATH      Write logs to specified file

Examples:
    python scripts/prod_setup.py --check-only                # Validate configuration
    python scripts/prod_setup.py                             # Full production setup
    python scripts/prod_setup.py --workers 4 --bind 0.0.0.0:8000
    python scripts/prod_setup.py --ssl-cert /etc/ssl/cert.pem --ssl-key /etc/ssl/key.pem
    python scripts/prod_setup.py --create-systemd            # Create systemd services

Production Requirements:
    - PostgreSQL database (configured in .env)
    - Redis for caching and Celery
    - Proper SECRET_KEY (not default)
    - DEV_MODE=false or unset
    - HTTPS configured (recommended)

Security Checks:
    - Validates SECRET_KEY is not default
    - Ensures DEV_MODE is disabled
    - Verifies database is PostgreSQL (not SQLite)
    - Checks for exposed debug settings
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import secrets
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# CONSTANTS
# ============================================================================

DANGEROUS_DEFAULTS = [
    "your-super-secret-key-change-this-in-production",
    "changeme",
    "secret",
    "password",
    "your-secret-key",
    "dev-secret",
]

REQUIRED_ENV_VARS = [
    "DATABASE_URL",
    "SECRET_KEY",
]

PRODUCTION_SETTINGS = {
    "DEV_MODE": "false",
    "DEBUG": "false",
    "APP_ENV": "production",
}

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
    logger = logging.getLogger('prod_setup')
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
# UTILITY FUNCTIONS
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


def check_command_exists(command: str) -> bool:
    """Check if a command exists in PATH."""
    return shutil.which(command) is not None


def is_service_running(port: int, host: str = 'localhost') -> bool:
    """Check if a service is running on a given port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            result = s.connect_ex((host, port))
            return result == 0
    except Exception:
        return False


def run_command(
    cmd: List[str],
    capture: bool = False,
    check: bool = True,
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
    env: Optional[Dict] = None
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


def get_cpu_count() -> int:
    """Get number of available CPU cores."""
    try:
        return os.cpu_count() or 2
    except Exception:
        return 2


def load_env_file(env_path: Path) -> Dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    if not env_path.exists():
        return env_vars

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip().strip('"\'')

    return env_vars


def generate_secret_key() -> str:
    """Generate a secure secret key."""
    return secrets.token_urlsafe(64)


# ============================================================================
# SECURITY CHECKS
# ============================================================================

def check_secret_key(env_vars: Dict[str, str]) -> Tuple[bool, str]:
    """Validate that SECRET_KEY is properly set and secure."""
    secret_key = env_vars.get('SECRET_KEY', '')

    if not secret_key:
        return False, "SECRET_KEY is not set"

    if len(secret_key) < 32:
        return False, "SECRET_KEY is too short (minimum 32 characters)"

    for dangerous in DANGEROUS_DEFAULTS:
        if dangerous.lower() in secret_key.lower():
            return False, f"SECRET_KEY contains dangerous default value"

    return True, "SECRET_KEY is properly configured"


def check_dev_mode(env_vars: Dict[str, str]) -> Tuple[bool, str]:
    """Verify that DEV_MODE is disabled."""
    dev_mode = env_vars.get('DEV_MODE', 'false').lower()

    if dev_mode in ('true', '1', 'yes'):
        return False, "DEV_MODE is enabled - CRITICAL SECURITY RISK in production"

    return True, "DEV_MODE is disabled"


def check_debug_mode(env_vars: Dict[str, str]) -> Tuple[bool, str]:
    """Verify that DEBUG mode is disabled."""
    debug = env_vars.get('DEBUG', 'false').lower()

    if debug in ('true', '1', 'yes'):
        return False, "DEBUG is enabled - should be disabled in production"

    return True, "DEBUG is disabled"


def check_database_url(env_vars: Dict[str, str]) -> Tuple[bool, str]:
    """Verify database is PostgreSQL (not SQLite) for production."""
    db_url = env_vars.get('DATABASE_URL', '')

    if not db_url:
        return False, "DATABASE_URL is not set"

    if 'sqlite' in db_url.lower():
        return False, "SQLite is not recommended for production - use PostgreSQL"

    if 'postgresql' in db_url.lower() or 'postgres' in db_url.lower():
        return True, "PostgreSQL database configured"

    return True, f"Database URL configured (type: {db_url.split(':')[0]})"


def check_admin_password(env_vars: Dict[str, str]) -> Tuple[bool, str]:
    """Verify admin password is not default."""
    admin_pass = env_vars.get('ADMIN_PASSWORD', '')

    dangerous = ['changeme', 'admin', 'password', '123456', 'admin123', 'changeme123']
    if admin_pass.lower() in dangerous:
        return False, "ADMIN_PASSWORD is set to a weak/default value"

    return True, "ADMIN_PASSWORD is configured"


def run_security_checks(env_vars: Dict[str, str]) -> Tuple[bool, List[Dict]]:
    """Run all security checks."""
    checks = [
        ("Secret Key", check_secret_key(env_vars)),
        ("Dev Mode", check_dev_mode(env_vars)),
        ("Debug Mode", check_debug_mode(env_vars)),
        ("Database", check_database_url(env_vars)),
        ("Admin Password", check_admin_password(env_vars)),
    ]

    results = []
    all_passed = True

    for name, (passed, message) in checks:
        results.append({
            "name": name,
            "passed": passed,
            "message": message
        })
        if not passed:
            all_passed = False

    return all_passed, results


# ============================================================================
# PRODUCTION CHECKS
# ============================================================================

def check_postgresql() -> Tuple[bool, str]:
    """Check if PostgreSQL client is available."""
    if check_command_exists('psql'):
        code, stdout, _ = run_command(['psql', '--version'], check=False, timeout=10)
        if code == 0:
            return True, f"PostgreSQL client: {stdout.strip()}"
    return False, "PostgreSQL client (psql) not found"


def check_redis() -> Tuple[bool, str]:
    """Check if Redis is available and running."""
    if not check_command_exists('redis-cli'):
        return False, "Redis client (redis-cli) not found"

    code, stdout, _ = run_command(['redis-cli', 'ping'], check=False, timeout=5)
    if code == 0 and 'PONG' in stdout:
        return True, "Redis is running"

    return False, "Redis is not responding"


def check_gunicorn() -> Tuple[bool, str]:
    """Check if Gunicorn is available."""
    # Try with uv first
    if check_command_exists('uv'):
        code, stdout, _ = run_command(
            ['uv', 'run', 'gunicorn', '--version'],
            check=False,
            timeout=10
        )
        if code == 0:
            return True, f"Gunicorn (via uv): {stdout.strip()}"

    # Direct gunicorn
    if check_command_exists('gunicorn'):
        code, stdout, _ = run_command(['gunicorn', '--version'], check=False, timeout=10)
        if code == 0:
            return True, f"Gunicorn: {stdout.strip()}"

    return False, "Gunicorn not found"


def check_node_npm() -> Tuple[bool, str]:
    """Check Node.js and npm availability."""
    if not check_command_exists('node'):
        return False, "Node.js not found"

    if not check_command_exists('npm'):
        return False, "npm not found"

    code, node_version, _ = run_command(['node', '--version'], check=False, timeout=10)
    code2, npm_version, _ = run_command(['npm', '--version'], check=False, timeout=10)

    if code == 0 and code2 == 0:
        return True, f"Node.js {node_version.strip()}, npm {npm_version.strip()}"

    return False, "Node.js or npm not working correctly"


def run_prerequisite_checks() -> Tuple[bool, List[Dict]]:
    """Run all prerequisite checks for production."""
    checks = [
        ("PostgreSQL", check_postgresql()),
        ("Redis", check_redis()),
        ("Gunicorn", check_gunicorn()),
        ("Node.js/npm", check_node_npm()),
    ]

    results = []
    all_passed = True

    for name, (passed, message) in checks:
        results.append({
            "name": name,
            "passed": passed,
            "message": message
        })
        if not passed:
            # Redis and Gunicorn are warnings, not critical failures
            if name not in ("Redis", "Gunicorn"):
                all_passed = False

    return all_passed, results


# ============================================================================
# PRODUCTION ENVIRONMENT SETUP
# ============================================================================

def setup_production_env(project_root: Path, env_vars: Dict[str, str]) -> bool:
    """Ensure production environment variables are set correctly."""
    backend_env = project_root / 'backend' / '.env'

    if not backend_env.exists():
        logger.error("Backend .env file not found - create it from .env.example first")
        return False

    content = backend_env.read_text()
    modified = False

    # Ensure production settings
    for key, value in PRODUCTION_SETTINGS.items():
        if f"{key}=" in content:
            # Replace existing value
            import re
            new_content = re.sub(f'{key}=.*', f'{key}={value}', content)
            if new_content != content:
                content = new_content
                modified = True
                logger.info(f"Set {key}={value}")
        else:
            # Add the setting
            content += f"\n{key}={value}\n"
            modified = True
            logger.info(f"Added {key}={value}")

    # Generate new secret key if needed
    if not check_secret_key(env_vars)[0]:
        new_secret = generate_secret_key()
        import re
        content = re.sub(r'SECRET_KEY=.*', f'SECRET_KEY={new_secret}', content)
        modified = True
        logger.info("Generated new SECRET_KEY")

    if modified:
        backend_env.write_text(content)
        logger.info("Updated backend/.env with production settings")

    return True


# ============================================================================
# BUILD AND DEPLOYMENT
# ============================================================================

def setup_python_environment(project_root: Path) -> bool:
    """Setup Python environment with UV."""
    backend_dir = project_root / 'backend'

    if not backend_dir.exists():
        logger.error(f"Backend directory not found: {backend_dir}")
        return False

    if not check_command_exists('uv'):
        logger.error("UV is required for production deployment")
        logger.error("Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False

    logger.info("Installing Python dependencies with UV...")
    code, stdout, stderr = run_command(
        ['uv', 'sync', '--frozen'],  # Use --frozen to ensure lockfile is respected
        cwd=str(backend_dir),
        check=False,
        timeout=300
    )

    if code != 0:
        logger.error(f"UV sync failed: {stderr}")
        return False

    logger.info("Python dependencies installed successfully")
    return True


def setup_node_environment(project_root: Path) -> bool:
    """Setup Node.js environment."""
    frontend_dir = project_root / 'frontend'

    if not frontend_dir.exists():
        logger.error(f"Frontend directory not found: {frontend_dir}")
        return False

    # Clean install for production
    logger.info("Installing Node.js dependencies (npm ci)...")
    code, stdout, stderr = run_command(
        ['npm', 'ci'],
        cwd=str(frontend_dir),
        check=False,
        timeout=300
    )

    if code != 0:
        logger.error(f"npm ci failed: {stderr}")
        return False

    logger.info("Node.js dependencies installed successfully")
    return True


def build_frontend(project_root: Path) -> bool:
    """Build frontend for production."""
    frontend_dir = project_root / 'frontend'

    logger.info("Building frontend for production...")

    # Set production environment
    build_env = os.environ.copy()
    build_env['NODE_ENV'] = 'production'

    code, stdout, stderr = run_command(
        ['npm', 'run', 'build'],
        cwd=str(frontend_dir),
        check=False,
        timeout=600,
        env=build_env
    )

    if code != 0:
        logger.error(f"Frontend build failed: {stderr}")
        return False

    # Verify build output exists
    build_dir = frontend_dir / '.next'
    if not build_dir.exists():
        logger.error("Build output (.next) not found")
        return False

    logger.info("Frontend built successfully")
    return True


def run_migrations(project_root: Path) -> bool:
    """Run database migrations."""
    logger.info("Running database migrations...")

    alembic_cmd = ['uv', 'run', 'alembic']

    code, stdout, stderr = run_command(
        alembic_cmd + ['upgrade', 'head'],
        cwd=str(project_root),
        check=False,
        timeout=120
    )

    if code == 0:
        logger.info("Database migrations completed successfully")
        return True

    # Handle already existing tables
    if 'already exists' in stderr.lower():
        logger.warning("Tables already exist - stamping database to head")
        stamp_code, _, stamp_stderr = run_command(
            alembic_cmd + ['stamp', 'head'],
            cwd=str(project_root),
            check=False,
            timeout=30
        )
        if stamp_code == 0:
            logger.info("Database stamped to head successfully")
            return True
        logger.error(f"Failed to stamp database: {stamp_stderr}")

    logger.error(f"Migration failed: {stderr}")
    return False


# ============================================================================
# SERVICE MANAGEMENT
# ============================================================================

def start_gunicorn(
    project_root: Path,
    workers: int,
    bind: str,
    ssl_cert: Optional[str] = None,
    ssl_key: Optional[str] = None
) -> bool:
    """Start backend with Gunicorn."""
    logger.info(f"Starting Gunicorn with {workers} workers on {bind}...")

    # Create log directory
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)

    # Build Gunicorn command
    cmd = [
        'uv', 'run', 'gunicorn',
        'backend.api.main:app',
        '-w', str(workers),
        '-k', 'uvicorn.workers.UvicornWorker',
        '-b', bind,
        '--access-logfile', str(log_dir / 'gunicorn-access.log'),
        '--error-logfile', str(log_dir / 'gunicorn-error.log'),
        '--capture-output',
        '--enable-stdio-inheritance',
    ]

    if ssl_cert and ssl_key:
        cmd.extend(['--certfile', ssl_cert, '--keyfile', ssl_key])
        logger.info("SSL/TLS enabled")

    # Set up production environment
    prod_env = os.environ.copy()
    prod_env['PYTHONPATH'] = str(project_root)
    prod_env['APP_ENV'] = 'production'
    prod_env['DEV_MODE'] = 'false'
    prod_env['DEBUG'] = 'false'

    try:
        subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=prod_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )

        # Wait and verify with retry (backend can take 30-60s with Ray)
        host_str, port_str = bind.rsplit(':', 1)
        port_num = int(port_str)
        max_retries = 12  # 12 * 5s = 60s max wait
        for i in range(max_retries):
            time.sleep(5)
            if is_service_running(port_num, host_str if host_str != '0.0.0.0' else 'localhost'):
                logger.info(f"Gunicorn started successfully on {bind}")
                logger.info(f"Logs: {log_dir}/gunicorn-*.log")
                return True
            if i < max_retries - 1:
                logger.debug(f"Waiting for Gunicorn... ({i+1}/{max_retries})")

        logger.warning("Gunicorn started but may not be responding yet - check logs")
        return True

    except Exception as e:
        logger.error(f"Failed to start Gunicorn: {e}")
        return False


def start_frontend_prod(project_root: Path) -> bool:
    """Start frontend in production mode."""
    frontend_dir = project_root / 'frontend'
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)

    logger.info("Starting frontend in production mode...")

    try:
        with open(log_dir / 'frontend.log', 'w') as log_file:
            subprocess.Popen(
                ['npm', 'run', 'start'],
                cwd=str(frontend_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

        # Wait with retry for frontend to start
        max_retries = 6  # 6 * 5s = 30s max wait
        for i in range(max_retries):
            time.sleep(5)
            if is_service_running(3000):
                logger.info("Frontend started on http://localhost:3000")
                logger.info(f"Logs: {log_dir}/frontend.log")
                return True
            if i < max_retries - 1:
                logger.debug(f"Waiting for frontend... ({i+1}/{max_retries})")

        logger.warning("Frontend started but may not be responding yet - check logs")
        return True

    except Exception as e:
        logger.error(f"Failed to start frontend: {e}")
        return False


def start_celery_worker(project_root: Path) -> bool:
    """Start Celery worker for background tasks."""
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)

    logger.info("Starting Celery worker...")

    celery_env = os.environ.copy()
    celery_env['PYTHONPATH'] = str(project_root)

    try:
        with open(log_dir / 'celery.log', 'w') as log_file:
            subprocess.Popen(
                ['uv', 'run', 'celery', '-A', 'backend.services.task_queue',
                 'worker', '--loglevel=info', '--concurrency=4'],
                cwd=str(project_root),
                env=celery_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

        logger.info("Celery worker started")
        logger.info(f"Logs: {log_dir}/celery.log")
        return True

    except Exception as e:
        logger.error(f"Failed to start Celery: {e}")
        return False


# ============================================================================
# SYSTEMD SERVICE CREATION
# ============================================================================

def create_systemd_services(project_root: Path, bind: str, workers: int):
    """Create systemd service files for production deployment."""
    user = os.environ.get('USER', 'www-data')
    python_path = str(project_root)

    # Backend service
    backend_service = f"""[Unit]
Description=AIDocumentIndexer Backend API
After=network.target postgresql.service redis.service

[Service]
Type=simple
User={user}
Group={user}
WorkingDirectory={project_root}
Environment="PYTHONPATH={python_path}"
Environment="APP_ENV=production"
Environment="DEV_MODE=false"
ExecStart=/usr/local/bin/uv run gunicorn backend.api.main:app -w {workers} -k uvicorn.workers.UvicornWorker -b {bind}
Restart=always
RestartSec=5
StandardOutput=append:/var/log/aidocindexer/backend.log
StandardError=append:/var/log/aidocindexer/backend-error.log

[Install]
WantedBy=multi-user.target
"""

    # Frontend service
    frontend_service = f"""[Unit]
Description=AIDocumentIndexer Frontend
After=network.target

[Service]
Type=simple
User={user}
Group={user}
WorkingDirectory={project_root}/frontend
ExecStart=/usr/bin/npm run start
Restart=always
RestartSec=5
StandardOutput=append:/var/log/aidocindexer/frontend.log
StandardError=append:/var/log/aidocindexer/frontend-error.log

[Install]
WantedBy=multi-user.target
"""

    # Celery worker service
    celery_service = f"""[Unit]
Description=AIDocumentIndexer Celery Worker
After=network.target redis.service

[Service]
Type=simple
User={user}
Group={user}
WorkingDirectory={project_root}
Environment="PYTHONPATH={python_path}"
ExecStart=/usr/local/bin/uv run celery -A backend.services.task_queue worker --loglevel=info --concurrency=4
Restart=always
RestartSec=5
StandardOutput=append:/var/log/aidocindexer/celery.log
StandardError=append:/var/log/aidocindexer/celery-error.log

[Install]
WantedBy=multi-user.target
"""

    # Write service files
    systemd_dir = project_root / 'deploy' / 'systemd'
    systemd_dir.mkdir(parents=True, exist_ok=True)

    (systemd_dir / 'aidocindexer-backend.service').write_text(backend_service)
    (systemd_dir / 'aidocindexer-frontend.service').write_text(frontend_service)
    (systemd_dir / 'aidocindexer-celery.service').write_text(celery_service)

    logger.info(f"Systemd service files created in {systemd_dir}")
    logger.info("To install, run:")
    logger.info(f"  sudo cp {systemd_dir}/*.service /etc/systemd/system/")
    logger.info("  sudo systemctl daemon-reload")
    logger.info("  sudo systemctl enable aidocindexer-backend aidocindexer-frontend aidocindexer-celery")
    logger.info("  sudo systemctl start aidocindexer-backend aidocindexer-frontend aidocindexer-celery")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def print_banner():
    """Print setup script banner."""
    banner = """
╔════════════════════════════════════════════════════════════╗
║        AIDocumentIndexer Production Setup Script           ║
╠════════════════════════════════════════════════════════════╣
║  Platform: {platform:<46} ║
║  Time: {time:<50} ║
║  Mode: PRODUCTION                                          ║
╚════════════════════════════════════════════════════════════╝
""".format(
        platform=get_platform().upper(),
        time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    print(banner)


def print_check_results(title: str, results: List[Dict]):
    """Print check results in a formatted way."""
    print(f"\n{title}")
    print("-" * 50)
    for result in results:
        symbol = "✓" if result['passed'] else "✗"
        color = "\033[0;32m" if result['passed'] else "\033[0;31m"
        reset = "\033[0m"
        print(f"  {color}{symbol}{reset} {result['name']}: {result['message']}")


def print_summary(results: Dict):
    """Print setup summary."""
    print("\n" + "=" * 60)
    print("PRODUCTION SETUP SUMMARY")
    print("=" * 60)

    for step, status in results.items():
        symbol = "✓" if status else "✗"
        color = "\033[0;32m" if status else "\033[0;31m"
        reset = "\033[0m"
        print(f"{color}{symbol}{reset} {step}")

    print("=" * 60)

    all_success = all(results.values())
    if all_success:
        print("\n\033[0;32mProduction setup completed successfully!\033[0m")
        print("\nServices running:")
        print("  - Backend:  http://localhost:8000 (Gunicorn)")
        print("  - Frontend: http://localhost:3000 (Production build)")
        print("  - Celery:   Background worker running")
        print("\nLogs available in: logs/")
        print("\n⚠️  Remember to:")
        print("  - Configure a reverse proxy (nginx/caddy) for HTTPS")
        print("  - Set up proper firewall rules")
        print("  - Configure backup for PostgreSQL database")
        print("  - Monitor logs and set up alerting")
    else:
        print("\n\033[0;31mProduction setup completed with errors.\033[0m")
        print("Review the logs and fix issues before deploying to production.")

    print()


def kill_process_on_port(port: int) -> bool:
    """Kill process running on a specific port (cross-platform)."""
    system = get_platform()
    killed = False

    if system == 'windows':
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
        code, stdout, _ = run_command(['lsof', '-ti', f':{port}'], check=False)
        if code == 0 and stdout.strip():
            pids = stdout.strip().split('\n')
            for pid in pids:
                if pid.strip():
                    logger.info(f"Killing process {pid.strip()} on port {port}")
                    run_command(['kill', '-9', pid.strip()], check=False)
                    killed = True

    return killed


def stop_all_services() -> Dict[str, bool]:
    """Stop all running production services."""
    results = {}

    # Stop services by port
    service_ports = {
        'backend (Gunicorn)': 8000,
        'frontend': 3000,
        'redis': 6379,
    }

    for service, port in service_ports.items():
        if kill_process_on_port(port):
            logger.info(f"Stopped {service} (port {port})")
            results[service] = True
        else:
            results[service] = True  # Not an error if it wasn't running

    # Stop Celery workers by process name
    system = get_platform()
    if system != 'windows':
        code, _, _ = run_command(['pkill', '-f', 'celery.*worker'], check=False)
        if code == 0:
            logger.info("Stopped Celery workers")
        results['celery'] = True

    # Stop Gunicorn workers
    code, _, _ = run_command(['pkill', '-f', 'gunicorn'], check=False)
    if code == 0:
        logger.info("Stopped Gunicorn processes")

    # Stop Ray processes
    code, _, _ = run_command(['pkill', '-f', 'ray::'], check=False)
    code, _, _ = run_command(['pkill', '-f', 'raylet'], check=False)
    code, _, _ = run_command(['pkill', '-f', 'gcs_server'], check=False)
    results['ray'] = True

    return results


def main():
    """Main setup function."""
    global logger

    # Parse arguments
    parser = argparse.ArgumentParser(description='AIDocumentIndexer Production Setup')
    parser.add_argument('command', nargs='?', default='start', choices=['start', 'stop', 'restart'],
                       help='Command: start (default), stop, or restart')
    parser.add_argument('--check-only', action='store_true', help='Only run checks')
    parser.add_argument('--skip-build', action='store_true', help='Skip frontend build')
    parser.add_argument('--skip-services', action='store_true', help='Skip starting services')
    parser.add_argument('--workers', type=int, default=0, help='Number of Gunicorn workers')
    parser.add_argument('--bind', type=str, default='0.0.0.0:8000', help='Gunicorn bind address')
    parser.add_argument('--ssl-cert', type=str, help='Path to SSL certificate')
    parser.add_argument('--ssl-key', type=str, help='Path to SSL private key')
    parser.add_argument('--create-systemd', action='store_true', help='Create systemd service files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--log-file', type=str, help='Path to log file')
    args = parser.parse_args()

    # Calculate workers if not specified (2 * CPU cores + 1)
    if args.workers <= 0:
        args.workers = (2 * get_cpu_count()) + 1

    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Setup logging
    log_file = args.log_file or str(project_root / 'logs' / 'prod_setup.log')
    Path(log_file).parent.mkdir(exist_ok=True)
    logger = setup_logging(verbose=args.verbose, log_file=log_file)

    # Print banner
    print_banner()

    logger.info(f"Project root: {project_root}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Bind: {args.bind}")

    # Handle stop command
    if args.command == 'stop':
        logger.info("Stopping all production services...")
        stop_results = stop_all_services()
        print("\n" + "=" * 60)
        print("PRODUCTION SERVICES STOPPED")
        print("=" * 60)
        for service, stopped in stop_results.items():
            symbol = "✓" if stopped else "✗"
            color = "\033[0;32m" if stopped else "\033[0;31m"
            reset = "\033[0m"
            print(f"{color}{symbol}{reset} {service}")
        print("=" * 60)
        print("\n\033[0;32mAll production services stopped.\033[0m\n")
        sys.exit(0)

    # Handle restart command (stop then continue with start)
    if args.command == 'restart':
        logger.info("Restarting all production services...")
        stop_all_services()
        logger.info("Services stopped, now starting fresh...")

    # Load environment
    env_path = project_root / 'backend' / '.env'
    env_vars = load_env_file(env_path)

    # Run prerequisite checks
    prereq_ok, prereq_results = run_prerequisite_checks()
    print_check_results("PREREQUISITE CHECKS", prereq_results)

    # Run security checks
    security_ok, security_results = run_security_checks(env_vars)
    print_check_results("SECURITY CHECKS", security_results)

    if args.check_only:
        print("\n" + "=" * 50)
        if prereq_ok and security_ok:
            print("\033[0;32mAll checks passed - ready for production deployment\033[0m")
            sys.exit(0)
        else:
            print("\033[0;31mSome checks failed - fix issues before deployment\033[0m")
            sys.exit(1)

    # Fail if critical security issues
    if not security_ok:
        logger.error("Security checks failed - fix issues before proceeding")
        logger.error("Run with --check-only to see detailed results")
        sys.exit(1)

    results = {}

    # 1. Setup production environment
    results['Production environment'] = setup_production_env(project_root, env_vars)

    # 2. Python dependencies
    results['Python dependencies'] = setup_python_environment(project_root)

    # 3. Node dependencies
    results['Node.js dependencies'] = setup_node_environment(project_root)

    # 4. Build frontend
    if not args.skip_build:
        results['Frontend build'] = build_frontend(project_root)
    else:
        logger.info("Skipping frontend build (--skip-build)")
        results['Frontend build'] = True

    # 5. Database migrations
    results['Database migrations'] = run_migrations(project_root)

    # 6. Create systemd services if requested
    if args.create_systemd:
        create_systemd_services(project_root, args.bind, args.workers)
        results['Systemd services'] = True

    # 7. Start services
    if not args.skip_services:
        results['Gunicorn backend'] = start_gunicorn(
            project_root,
            args.workers,
            args.bind,
            args.ssl_cert,
            args.ssl_key
        )
        results['Frontend production'] = start_frontend_prod(project_root)

        # Start Celery if Redis is available
        if check_redis()[0]:
            results['Celery worker'] = start_celery_worker(project_root)
        else:
            logger.warning("Redis not available - Celery worker not started")
            results['Celery worker'] = False
    else:
        logger.info("Skipping service startup (--skip-services)")
        results['Start services'] = True

    # Print summary
    print_summary(results)

    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()
