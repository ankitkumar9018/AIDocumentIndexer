"""
AIDocumentIndexer - API Routes
==============================

FastAPI routers for all API endpoints.
"""

from backend.api.routes.documents import router as documents_router
from backend.api.routes.chat import router as chat_router
from backend.api.routes.upload import router as upload_router
from backend.api.routes.auth import router as auth_router

__all__ = [
    "documents_router",
    "chat_router",
    "upload_router",
    "auth_router",
]
