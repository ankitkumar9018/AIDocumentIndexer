"""
Cross-platform detection and configuration utilities.

This module provides utilities for detecting the current platform
and configuring behavior accordingly (Mac, Linux, Windows).
"""

import os
import platform
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
import structlog

logger = structlog.get_logger(__name__)


PlatformType = Literal["darwin", "linux", "windows", "unknown"]


@dataclass(frozen=True)
class PlatformConfig:
    """Platform-specific configuration."""

    # Platform identification
    system: PlatformType
    is_mac: bool
    is_linux: bool
    is_windows: bool

    # GPU capabilities
    has_cuda: bool
    has_mps: bool  # Apple Metal Performance Shaders
    has_rocm: bool  # AMD ROCm
    preferred_device: Literal["cuda", "mps", "cpu"]

    # Feature availability
    signal_alarm_available: bool  # SIGALRM for timeouts
    ray_available: bool
    multiprocessing_start_method: Literal["fork", "spawn", "forkserver"]

    # System info
    cpu_count: int
    total_memory_gb: float
    python_version: str


@lru_cache(maxsize=1)
def get_platform_config() -> PlatformConfig:
    """
    Detect and return platform configuration.

    Results are cached for performance.

    Returns:
        PlatformConfig with all platform-specific settings
    """
    system = platform.system().lower()

    is_mac = system == "darwin"
    is_linux = system == "linux"
    is_windows = system == "windows"

    # Map to our type
    platform_type: PlatformType
    if is_mac:
        platform_type = "darwin"
    elif is_linux:
        platform_type = "linux"
    elif is_windows:
        platform_type = "windows"
    else:
        platform_type = "unknown"

    # GPU detection
    has_cuda = _check_cuda()
    has_mps = _check_mps()
    has_rocm = _check_rocm()

    # Determine preferred device
    if has_cuda and not is_mac:
        preferred_device: Literal["cuda", "mps", "cpu"] = "cuda"
    elif has_mps:
        preferred_device = "mps"
    else:
        preferred_device = "cpu"

    # Feature availability
    signal_alarm_available = not is_windows

    # Ray availability
    ray_available = _check_ray()

    # Multiprocessing start method
    if is_windows:
        mp_method: Literal["fork", "spawn", "forkserver"] = "spawn"
    elif is_mac:
        # macOS defaults to spawn since Python 3.8
        mp_method = "spawn"
    else:
        mp_method = "fork"

    # System info
    cpu_count = os.cpu_count() or 1
    total_memory_gb = _get_total_memory_gb()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    config = PlatformConfig(
        system=platform_type,
        is_mac=is_mac,
        is_linux=is_linux,
        is_windows=is_windows,
        has_cuda=has_cuda,
        has_mps=has_mps,
        has_rocm=has_rocm,
        preferred_device=preferred_device,
        signal_alarm_available=signal_alarm_available,
        ray_available=ray_available,
        multiprocessing_start_method=mp_method,
        cpu_count=cpu_count,
        total_memory_gb=total_memory_gb,
        python_version=python_version,
    )

    logger.info(
        "Platform detected",
        system=config.system,
        device=config.preferred_device,
        cpu_count=config.cpu_count,
        memory_gb=round(config.total_memory_gb, 1),
    )

    return config


def _check_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
    except Exception:
        return False


def _check_mps() -> bool:
    """Check if Apple Metal Performance Shaders is available."""
    try:
        import torch

        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False
    except Exception:
        return False


def _check_rocm() -> bool:
    """Check if AMD ROCm is available."""
    try:
        import torch

        return torch.cuda.is_available() and "rocm" in torch.version.hip if hasattr(torch.version, "hip") else False
    except ImportError:
        return False
    except Exception:
        return False


def _check_ray() -> bool:
    """Check if Ray is available."""
    try:
        import ray  # noqa: F401

        return True
    except ImportError:
        return False


def _get_total_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # Fallback: try reading from /proc/meminfo on Linux
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # Format: "MemTotal:       16384000 kB"
                        kb = int(line.split()[1])
                        return kb / (1024**2)
        except Exception:
            pass
        return 8.0  # Default assumption


def get_device_for_torch() -> str:
    """
    Get the appropriate PyTorch device string.

    Returns:
        Device string like "cuda", "mps", or "cpu"
    """
    config = get_platform_config()
    return config.preferred_device


def get_recommended_workers() -> int:
    """
    Get the recommended number of worker processes.

    Returns:
        Number of workers based on CPU and memory
    """
    config = get_platform_config()

    # Base on CPU count
    workers = config.cpu_count

    # Limit based on memory (assume ~2GB per worker for ML tasks)
    memory_limited = max(1, int(config.total_memory_gb / 2))

    # Use the smaller of the two
    workers = min(workers, memory_limited)

    # Cap at reasonable maximum
    workers = min(workers, 8)

    return max(1, workers)


def get_temp_dir() -> Path:
    """
    Get the platform-appropriate temporary directory.

    Returns:
        Path to temp directory
    """
    config = get_platform_config()

    if config.is_mac:
        # macOS: prefer /tmp for faster access
        return Path("/tmp")
    elif config.is_linux:
        return Path("/tmp")
    else:
        # Windows: use system temp
        import tempfile

        return Path(tempfile.gettempdir())


def get_cache_dir() -> Path:
    """
    Get the platform-appropriate cache directory.

    Returns:
        Path to cache directory
    """
    config = get_platform_config()

    if config.is_mac:
        return Path.home() / "Library" / "Caches" / "AIDocumentIndexer"
    elif config.is_linux:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            return Path(xdg_cache) / "aidocumentindexer"
        return Path.home() / ".cache" / "aidocumentindexer"
    else:
        # Windows
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "AIDocumentIndexer" / "Cache"
        return Path.home() / "AppData" / "Local" / "AIDocumentIndexer" / "Cache"


def get_data_dir() -> Path:
    """
    Get the platform-appropriate data directory.

    Returns:
        Path to data directory
    """
    config = get_platform_config()

    if config.is_mac:
        return Path.home() / "Library" / "Application Support" / "AIDocumentIndexer"
    elif config.is_linux:
        xdg_data = os.environ.get("XDG_DATA_HOME")
        if xdg_data:
            return Path(xdg_data) / "aidocumentindexer"
        return Path.home() / ".local" / "share" / "aidocumentindexer"
    else:
        # Windows
        app_data = os.environ.get("APPDATA")
        if app_data:
            return Path(app_data) / "AIDocumentIndexer"
        return Path.home() / "AppData" / "Roaming" / "AIDocumentIndexer"


def is_gpu_memory_sufficient(required_gb: float = 4.0) -> bool:
    """
    Check if GPU has sufficient memory for ML tasks.

    Args:
        required_gb: Minimum required GPU memory in GB

    Returns:
        True if GPU memory is sufficient
    """
    config = get_platform_config()

    if config.preferred_device == "cpu":
        return False

    try:
        import torch

        if config.has_cuda:
            device = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(device).total_memory
            return (total / (1024**3)) >= required_gb
        elif config.has_mps:
            # MPS doesn't expose memory directly, assume sufficient if available
            return True
    except Exception:
        pass

    return False


def get_ocr_provider_recommendation() -> str:
    """
    Get the recommended OCR provider based on platform.

    Returns:
        Recommended OCR provider: "tesseract" or "paddleocr"
    """
    config = get_platform_config()

    # PaddleOCR has issues on Mac (MPS memory leaks)
    # and requires GPU for good performance
    if config.is_mac:
        return "tesseract"

    # On Linux/Windows with CUDA, PaddleOCR can be faster and more accurate
    if config.has_cuda:
        return "paddleocr"

    # Default to Tesseract for CPU-only systems
    return "tesseract"


def log_platform_info() -> None:
    """Log detailed platform information for debugging."""
    config = get_platform_config()

    logger.info(
        "Platform configuration",
        system=config.system,
        python_version=config.python_version,
        cpu_count=config.cpu_count,
        memory_gb=round(config.total_memory_gb, 1),
        has_cuda=config.has_cuda,
        has_mps=config.has_mps,
        has_rocm=config.has_rocm,
        preferred_device=config.preferred_device,
        ray_available=config.ray_available,
        signal_alarm_available=config.signal_alarm_available,
        recommended_workers=get_recommended_workers(),
        recommended_ocr=get_ocr_provider_recommendation(),
    )
