"""
Cython Extensions Build Script
==============================

Build Cython extensions for performance-critical functions.

Usage:
    python setup.py build_ext --inplace

Or called automatically at runtime by the __init__.py module.
"""

import os
import sys
from pathlib import Path


def build_extensions():
    """Build Cython extensions. Returns True on success, False on failure."""
    try:
        from Cython.Build import cythonize
        from setuptools import Extension, setup
        from setuptools.dist import Distribution
        import numpy as np
    except ImportError as e:
        print(f"Cannot build Cython extensions: {e}")
        return False

    # Get the directory containing this script
    ext_dir = Path(__file__).parent.absolute()

    # Define extensions
    extensions = [
        Extension(
            "similarity",
            [str(ext_dir / "similarity.pyx")],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-ffast-math"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        ),
    ]

    try:
        # Cythonize
        ext_modules = cythonize(
            extensions,
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
            },
            quiet=True,
        )

        # Build in-place
        dist = Distribution({"ext_modules": ext_modules})
        dist.package_dir = {"": str(ext_dir)}

        # Change to extension directory for build
        old_cwd = os.getcwd()
        os.chdir(ext_dir)

        try:
            cmd = dist.get_command_obj("build_ext")
            cmd.inplace = True
            cmd.ensure_finalized()
            cmd.run()
            return True
        finally:
            os.chdir(old_cwd)

    except Exception as e:
        print(f"Failed to build Cython extensions: {e}")
        return False


if __name__ == "__main__":
    # When run directly, use standard setup
    try:
        from Cython.Build import cythonize
        from setuptools import Extension, setup
        import numpy as np

        ext_dir = Path(__file__).parent.absolute()

        extensions = [
            Extension(
                "similarity",
                [str(ext_dir / "similarity.pyx")],
                include_dirs=[np.get_include()],
                extra_compile_args=["-O3", "-ffast-math"],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            ),
        ]

        setup(
            name="cython_extensions",
            ext_modules=cythonize(
                extensions,
                compiler_directives={
                    "language_level": "3",
                    "boundscheck": False,
                    "wraparound": False,
                },
            ),
            include_dirs=[np.get_include()],
        )
    except ImportError as e:
        print(f"Cannot build: {e}")
        print("Install Cython with: pip install cython")
        sys.exit(1)
