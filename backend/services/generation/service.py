"""
AIDocumentIndexer - Document Generation Service
================================================

Main document generation service.

This module provides backward compatibility by re-exporting
the DocumentGenerationService from the original generator.py.

The modular architecture allows for incremental migration:
1. Config, models, utils, citations, styles are fully extracted
2. Format generators provide stubs with guidelines
3. Main service logic remains in generator.py for now
4. Future migration can move service methods here

Usage:
    from backend.services.generation import (
        DocumentGenerationService,
        get_generation_service,
    )

    service = get_generation_service()
    job = await service.create_job(...)
"""

# Re-export from the original module for backward compatibility
# The full service implementation remains in generator.py
# This allows gradual migration while maintaining compatibility

from backend.services.generator import (
    DocumentGenerationService,
    get_generation_service,
)

__all__ = [
    "DocumentGenerationService",
    "get_generation_service",
]
