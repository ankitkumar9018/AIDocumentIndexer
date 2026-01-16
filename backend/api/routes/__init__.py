"""
AIDocumentIndexer - API Routes
==============================

FastAPI routers for all API endpoints.
"""

from backend.api.routes.documents import router as documents_router
from backend.api.routes.chat import router as chat_router
from backend.api.routes.upload import router as upload_router
from backend.api.routes.auth import router as auth_router
from backend.api.routes.agent import router as agent_router
from backend.api.routes.folders import router as folders_router
from backend.api.routes.preferences import router as preferences_router
from backend.api.routes.content_review import router as content_review_router

__all__ = [
    "documents_router",
    "chat_router",
    "upload_router",
    "auth_router",
    "agent_router",
    "folders_router",
    "preferences_router",
    "content_review_router",
]
