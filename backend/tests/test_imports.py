"""
Phase 83: Import Smoke Test

Verifies that all backend service modules and API route modules can be imported
without errors. This catches broken imports, missing dependencies, syntax errors,
and undefined variable references at CI time rather than in production.
"""

import importlib
import os
import pytest


def _get_python_modules(directory: str, package_prefix: str) -> list:
    """Get all Python module names in a directory."""
    modules = []
    if not os.path.isdir(directory):
        return modules
    for f in sorted(os.listdir(directory)):
        if f.endswith(".py") and f not in ("__init__.py",):
            modules.append(f"{package_prefix}.{f[:-3]}")
    return modules


# Discover all service and route modules
_SERVICE_MODULES = _get_python_modules(
    os.path.join(os.path.dirname(__file__), "..", "services"),
    "backend.services",
)
_ROUTE_MODULES = _get_python_modules(
    os.path.join(os.path.dirname(__file__), "..", "api", "routes"),
    "backend.api.routes",
)


@pytest.mark.parametrize("module_name", _SERVICE_MODULES)
def test_service_import(module_name: str):
    """Verify each backend service module imports without error."""
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", _ROUTE_MODULES)
def test_route_import(module_name: str):
    """Verify each API route module imports without error."""
    importlib.import_module(module_name)


def test_main_app_import():
    """Verify the FastAPI application can be imported and created."""
    from backend.api.main import app
    assert app is not None
