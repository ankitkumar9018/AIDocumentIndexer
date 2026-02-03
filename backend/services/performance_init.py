"""
Performance Optimization Initializer
====================================

Initializes all performance optimizations at server startup.
Call `initialize_performance_optimizations()` during FastAPI startup.

Features:
- Cython extension compilation (if Cython available)
- GPU acceleration detection and warmup
- MinHash deduplicator initialization
- Performance status reporting

All optimizations have graceful fallbacks if dependencies are missing.

Environment Variables (for Kubernetes/AWS/Cloud):
- PERF_COMPILE_CYTHON: Enable Cython compilation (default: true)
- PERF_INIT_GPU: Enable GPU initialization (default: true)
- PERF_INIT_MINHASH: Enable MinHash deduplication (default: true)
- PERF_WARMUP_GPU: Run GPU warmup at startup (default: false)
- PERF_GPU_PREFER: Prefer GPU over CPU (default: true)
- PERF_MIXED_PRECISION: Use FP16 for GPU ops (default: true)
- PERF_MINHASH_PERMS: MinHash permutations (default: 128)
- PERF_MINHASH_THRESHOLD: MinHash similarity threshold (default: 0.8)
"""

import asyncio
import os
import time
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


def _get_bool_env(key: str, default: bool = True) -> bool:
    """Get boolean from environment variable."""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


def _get_float_env(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_int_env(key: str, default: int) -> int:
    """Get int from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


async def initialize_performance_optimizations(
    compile_cython: Optional[bool] = None,
    init_gpu: Optional[bool] = None,
    init_minhash: Optional[bool] = None,
    warmup_gpu: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Initialize all performance optimizations.

    Should be called during FastAPI startup event.
    Settings can be overridden via environment variables for K8s/cloud deployments.

    Args:
        compile_cython: Compile Cython extensions (env: PERF_COMPILE_CYTHON)
        init_gpu: Initialize GPU accelerator (env: PERF_INIT_GPU)
        init_minhash: Initialize MinHash deduplicator (env: PERF_INIT_MINHASH)
        warmup_gpu: Run GPU warmup (env: PERF_WARMUP_GPU)

    Returns:
        Status dictionary with optimization information
    """
    # Read from env vars if not explicitly provided
    if compile_cython is None:
        compile_cython = _get_bool_env("PERF_COMPILE_CYTHON", True)
    if init_gpu is None:
        init_gpu = _get_bool_env("PERF_INIT_GPU", True)
    if init_minhash is None:
        init_minhash = _get_bool_env("PERF_INIT_MINHASH", True)
    if warmup_gpu is None:
        warmup_gpu = _get_bool_env("PERF_WARMUP_GPU", False)

    start_time = time.time()
    status = {
        "initialized": True,
        "cython": {"status": "skipped"},
        "gpu": {"status": "skipped"},
        "minhash": {"status": "skipped"},
        "initialization_time_ms": 0,
        "config": {
            "compile_cython": compile_cython,
            "init_gpu": init_gpu,
            "init_minhash": init_minhash,
            "warmup_gpu": warmup_gpu,
        },
    }

    logger.info(
        "Initializing performance optimizations...",
        compile_cython=compile_cython,
        init_gpu=init_gpu,
        init_minhash=init_minhash,
        warmup_gpu=warmup_gpu,
    )

    # 1. Initialize Cython extensions
    if compile_cython:
        status["cython"] = await _init_cython()

    # 2. Initialize GPU acceleration
    if init_gpu:
        prefer_gpu = _get_bool_env("PERF_GPU_PREFER", True)
        mixed_precision = _get_bool_env("PERF_MIXED_PRECISION", True)
        status["gpu"] = await _init_gpu(
            warmup=warmup_gpu,
            prefer_gpu=prefer_gpu,
            mixed_precision=mixed_precision,
        )

    # 3. Initialize MinHash deduplicator
    if init_minhash:
        num_perm = _get_int_env("PERF_MINHASH_PERMS", 128)
        threshold = _get_float_env("PERF_MINHASH_THRESHOLD", 0.8)
        status["minhash"] = await _init_minhash(num_perm=num_perm, threshold=threshold)

    elapsed_ms = (time.time() - start_time) * 1000
    status["initialization_time_ms"] = round(elapsed_ms, 2)

    logger.info(
        "Performance optimizations initialized",
        cython=status["cython"]["status"],
        gpu=status["gpu"]["status"],
        minhash=status["minhash"]["status"],
        time_ms=round(elapsed_ms, 2),
    )

    return status


async def _init_cython() -> Dict[str, Any]:
    """Initialize Cython extensions."""
    try:
        # Import triggers compilation if needed
        from backend.services.cython_extensions import (
            get_optimization_status,
            is_using_fallback,
        )

        opt_status = get_optimization_status()

        if opt_status["using_cython"]:
            logger.info("Cython extensions loaded (10-100x faster)")
            return {
                "status": "cython",
                "using_cython": True,
                "speedup": "10-100x",
            }
        else:
            logger.info("Using NumPy fallbacks (Cython not compiled)")
            return {
                "status": "fallback",
                "using_cython": False,
                "speedup": "1x (baseline)",
            }

    except ImportError as e:
        logger.warning("Cython extensions not available", error=str(e))
        return {
            "status": "unavailable",
            "error": str(e),
        }
    except Exception as e:
        logger.error("Cython initialization failed", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


async def _init_gpu(
    warmup: bool = False,
    prefer_gpu: bool = True,
    mixed_precision: bool = True,
) -> Dict[str, Any]:
    """Initialize GPU acceleration with configurable settings."""
    try:
        from backend.services.gpu_acceleration import (
            GPUSimilarityAccelerator,
            is_gpu_available,
            check_gpu_availability,
            _similarity_accelerator,
        )

        # Check availability
        availability = check_gpu_availability()

        # Initialize accelerator with settings
        global _similarity_accelerator
        from backend.services import gpu_acceleration
        gpu_acceleration._similarity_accelerator = GPUSimilarityAccelerator(
            prefer_gpu=prefer_gpu,
            mixed_precision=mixed_precision,
        )
        accelerator = gpu_acceleration._similarity_accelerator
        accel_status = accelerator.get_status()

        if accel_status["has_gpu"]:
            device = accel_status["device"]
            logger.info(f"GPU acceleration enabled ({device})")

            # Optional warmup
            if warmup:
                logger.info("Running GPU warmup...")
                import numpy as np
                # Small matrix multiply to warm up GPU
                test_query = np.random.randn(768).astype(np.float32)
                test_corpus = np.random.randn(100, 768).astype(np.float32)
                _ = accelerator.cosine_similarity_batch(test_query, test_corpus)
                logger.info("GPU warmup complete")

            return {
                "status": "gpu",
                "device": device,
                "mixed_precision": accel_status["mixed_precision"],
                **availability,
            }
        else:
            logger.info("GPU not available, using CPU fallback")
            return {
                "status": "cpu_fallback",
                "device": "cpu",
                **availability,
            }

    except ImportError as e:
        logger.warning("GPU acceleration not available", error=str(e))
        return {
            "status": "unavailable",
            "error": str(e),
        }
    except Exception as e:
        logger.error("GPU initialization failed", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


async def _init_minhash(num_perm: int = 128, threshold: float = 0.8) -> Dict[str, Any]:
    """Initialize MinHash deduplicator with configurable settings."""
    try:
        from backend.services.minhash_dedup import (
            get_minhash_deduplicator,
            is_minhash_available,
        )

        # Initialize deduplicator with settings
        dedup = get_minhash_deduplicator(num_perm=num_perm, threshold=threshold)
        stats = dedup.get_stats()

        if stats["using_minhash"]:
            logger.info("MinHash LSH deduplicator initialized (O(n) complexity)")
            return {
                "status": "minhash",
                "method": "minhash_lsh",
                "complexity": "O(n)",
                "num_permutations": stats["num_permutations"],
            }
        else:
            logger.info("Using exact Jaccard fallback (O(n^2) complexity)")
            return {
                "status": "fallback",
                "method": "exact_jaccard",
                "complexity": "O(n^2)",
            }

    except ImportError as e:
        logger.warning("MinHash not available", error=str(e))
        return {
            "status": "unavailable",
            "error": str(e),
        }
    except Exception as e:
        logger.error("MinHash initialization failed", error=str(e))
        return {
            "status": "error",
            "error": str(e),
        }


def get_performance_status() -> Dict[str, Any]:
    """
    Get current status of all performance optimizations.

    Can be called anytime after initialization.
    """
    status = {
        "cython": {"status": "unknown"},
        "gpu": {"status": "unknown"},
        "minhash": {"status": "unknown"},
    }

    # Check Cython
    try:
        from backend.services.cython_extensions import get_optimization_status
        status["cython"] = get_optimization_status()
    except ImportError:
        status["cython"] = {"status": "not_installed"}

    # Check GPU
    try:
        from backend.services.gpu_acceleration import (
            get_similarity_accelerator,
            check_gpu_availability,
        )
        accel = get_similarity_accelerator(prefer_gpu=True)
        status["gpu"] = {
            **accel.get_status(),
            **check_gpu_availability(),
        }
    except ImportError:
        status["gpu"] = {"status": "not_installed"}

    # Check MinHash
    try:
        from backend.services.minhash_dedup import get_minhash_deduplicator
        dedup = get_minhash_deduplicator()
        status["minhash"] = dedup.get_stats()
    except ImportError:
        status["minhash"] = {"status": "not_installed"}

    return status
