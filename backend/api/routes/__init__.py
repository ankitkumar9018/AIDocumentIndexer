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
from backend.api.routes.evaluation import router as evaluation_router
from backend.api.routes.cost_optimization import router as cost_optimization_router
from backend.api.routes.license import router as license_router
from backend.api.routes.menu import router as menu_router
from backend.api.routes.intelligence import router as intelligence_router

__all__ = [
    "documents_router",
    "chat_router",
    "upload_router",
    "auth_router",
    "agent_router",
    "folders_router",
    "preferences_router",
    "content_review_router",
    "evaluation_router",
    "cost_optimization_router",
    "license_router",
    "menu_router",
    "intelligence_router",
]
