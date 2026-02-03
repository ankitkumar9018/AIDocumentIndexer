"""
Cython Extensions Module
========================

Provides high-performance implementations of similarity computations.
Automatically compiles Cython extensions on first import if not already built.

Features:
- Runtime compilation on server start (if Cython available)
- Automatic fallback to pure Python if compilation fails
- Thread-safe initialization
- No hard dependency on Cython (graceful degradation)

Usage:
    from backend.services.cython_extensions import (
        cosine_similarity_batch,
        cosine_similarity_matrix,
        mmr_selection,
        hamming_distance_batch,
        is_using_fallback,
    )

    # Check if using optimized or fallback implementations
    if is_using_fallback():
        print("Using Python fallbacks (10-50x slower)")
    else:
        print("Using Cython optimizations")
"""

import os
import sys
import threading
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# Thread-safe initialization
_init_lock = threading.Lock()
_initialized = False
_using_cython = False

# Module-level functions (will be assigned during initialization)
cosine_similarity_batch = None
cosine_similarity_matrix = None
mmr_selection = None
hamming_distance_batch = None
weighted_mean_pooling = None
jaccard_similarity_sets = None
is_using_fallback = None


def _try_import_compiled():
    """Try to import pre-compiled Cython extension."""
    global _using_cython
    try:
        from . import similarity as cython_sim
        return cython_sim
    except ImportError:
        return None


def _try_compile():
    """Try to compile Cython extensions at runtime."""
    try:
        # Check if Cython is available
        import Cython
    except ImportError:
        logger.debug("Cython not installed, skipping compilation")
        return False

    try:
        # Check if we have a C compiler
        from distutils.ccompiler import new_compiler
        compiler = new_compiler()
        if not compiler:
            logger.debug("No C compiler found, skipping Cython compilation")
            return False
    except Exception:
        pass  # Continue anyway, let the build fail if no compiler

    # Try to build
    ext_dir = Path(__file__).parent
    setup_path = ext_dir / "setup.py"

    if not setup_path.exists():
        logger.warning("Cython setup.py not found", path=str(setup_path))
        return False

    logger.info("Compiling Cython extensions (first run)...")

    try:
        from .setup import build_extensions
        success = build_extensions()

        if success:
            logger.info("Cython extensions compiled successfully")
            return True
        else:
            logger.warning("Cython extension compilation failed")
            return False

    except Exception as e:
        logger.warning("Cython compilation error", error=str(e))
        return False


def _load_fallbacks():
    """Load pure Python fallback implementations."""
    global cosine_similarity_batch, cosine_similarity_matrix
    global mmr_selection, hamming_distance_batch
    global weighted_mean_pooling, jaccard_similarity_sets
    global is_using_fallback

    from . import _fallback as fb

    cosine_similarity_batch = fb.cosine_similarity_batch
    cosine_similarity_matrix = fb.cosine_similarity_matrix
    mmr_selection = fb.mmr_selection
    hamming_distance_batch = fb.hamming_distance_batch
    weighted_mean_pooling = fb.weighted_mean_pooling
    jaccard_similarity_sets = fb.jaccard_similarity_sets
    is_using_fallback = fb.is_using_fallback

    logger.info("Using pure Python fallbacks for similarity functions")


def _load_cython(module):
    """Load Cython implementations."""
    global cosine_similarity_batch, cosine_similarity_matrix
    global mmr_selection, hamming_distance_batch
    global weighted_mean_pooling, is_using_fallback
    global jaccard_similarity_sets, _using_cython

    cosine_similarity_batch = module.cosine_similarity_batch
    cosine_similarity_matrix = module.cosine_similarity_matrix
    mmr_selection = module.mmr_selection
    hamming_distance_batch = module.hamming_distance_batch
    weighted_mean_pooling = module.weighted_mean_pooling
    is_using_fallback = module.is_using_fallback

    # Jaccard is Python-only (set operations don't benefit much from Cython)
    from . import _fallback as fb
    jaccard_similarity_sets = fb.jaccard_similarity_sets

    _using_cython = True
    logger.info("Cython extensions loaded successfully (10-100x faster)")


def initialize():
    """
    Initialize the Cython extensions module.

    Called automatically on first import, but can be called explicitly
    to control when compilation happens (e.g., at server startup).

    Thread-safe: only one thread will perform initialization.
    """
    global _initialized

    with _init_lock:
        if _initialized:
            return

        # Try to import pre-compiled extension
        cython_module = _try_import_compiled()

        if cython_module is not None:
            _load_cython(cython_module)
            _initialized = True
            return

        # Try to compile at runtime
        if _try_compile():
            # Retry import after compilation
            cython_module = _try_import_compiled()
            if cython_module is not None:
                _load_cython(cython_module)
                _initialized = True
                return

        # Fall back to pure Python
        _load_fallbacks()
        _initialized = True


def ensure_initialized():
    """Ensure module is initialized before use."""
    if not _initialized:
        initialize()


def get_optimization_status() -> dict:
    """
    Get status of Cython optimizations.

    Returns:
        Dictionary with optimization status information
    """
    ensure_initialized()

    return {
        "initialized": _initialized,
        "using_cython": _using_cython,
        "using_fallback": not _using_cython,
        "speedup_factor": "10-100x" if _using_cython else "1x (baseline)",
        "functions": {
            "cosine_similarity_batch": "cython" if _using_cython else "numpy",
            "cosine_similarity_matrix": "cython" if _using_cython else "numpy",
            "mmr_selection": "cython" if _using_cython else "numpy",
            "hamming_distance_batch": "cython" if _using_cython else "numpy",
            "weighted_mean_pooling": "cython" if _using_cython else "numpy",
            "jaccard_similarity_sets": "python",  # Always Python
        },
    }


# Auto-initialize on import
initialize()
