"""
Configuration management for Mandala Sync CLI.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import keyring


CONFIG_DIR = Path.home() / ".mandala-sync"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
KEYRING_SERVICE = "mandala-sync"


def get_config_dir() -> Path:
    """Get the configuration directory, creating it if necessary."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    if not CONFIG_FILE.exists():
        return {
            "server": None,
            "directories": [],
            "settings": {
                "auto_start": False,
                "default_collection": None,
                "default_access_tier": 1,
                "default_folder_id": None,
            },
        }

    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f) or {}


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file."""
    get_config_dir()
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_server_url() -> Optional[str]:
    """Get the configured server URL."""
    config = load_config()
    return config.get("server")


def set_server_url(url: str) -> None:
    """Set the server URL."""
    config = load_config()
    config["server"] = url.rstrip("/")
    save_config(config)


def get_token() -> Optional[str]:
    """Get the stored authentication token."""
    try:
        return keyring.get_password(KEYRING_SERVICE, "token")
    except Exception:
        # Fallback to file-based storage
        token_file = get_config_dir() / ".token"
        if token_file.exists():
            return token_file.read_text().strip()
        return None


def set_token(token: str) -> None:
    """Store the authentication token securely."""
    try:
        keyring.set_password(KEYRING_SERVICE, "token", token)
    except Exception:
        # Fallback to file-based storage
        token_file = get_config_dir() / ".token"
        token_file.write_text(token)
        os.chmod(token_file, 0o600)


def clear_token() -> None:
    """Clear the stored authentication token."""
    try:
        keyring.delete_password(KEYRING_SERVICE, "token")
    except Exception:
        pass

    token_file = get_config_dir() / ".token"
    if token_file.exists():
        token_file.unlink()


def get_watched_directories() -> List[Dict[str, Any]]:
    """Get the list of watched directories."""
    config = load_config()
    return config.get("directories", [])


def add_watched_directory(
    path: str,
    recursive: bool = True,
    collection: Optional[str] = None,
    access_tier: int = 1,
    folder_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a directory to watch."""
    config = load_config()
    directories = config.get("directories", [])

    # Check if already exists
    path = str(Path(path).resolve())
    for d in directories:
        if d["path"] == path:
            raise ValueError(f"Directory already being watched: {path}")

    dir_config = {
        "path": path,
        "recursive": recursive,
        "collection": collection,
        "access_tier": access_tier,
        "folder_id": folder_id,
        "enabled": True,
    }

    directories.append(dir_config)
    config["directories"] = directories
    save_config(config)

    return dir_config


def remove_watched_directory(path: str) -> bool:
    """Remove a directory from watch list."""
    config = load_config()
    directories = config.get("directories", [])

    path = str(Path(path).resolve())
    new_dirs = [d for d in directories if d["path"] != path]

    if len(new_dirs) == len(directories):
        return False

    config["directories"] = new_dirs
    save_config(config)
    return True


def get_settings() -> Dict[str, Any]:
    """Get global settings."""
    config = load_config()
    return config.get("settings", {})


def set_setting(key: str, value: Any) -> None:
    """Set a global setting."""
    config = load_config()
    if "settings" not in config:
        config["settings"] = {}
    config["settings"][key] = value
    save_config(config)
