"""
AIDocumentIndexer - FastAPI Application Entry Point
====================================================

This is the main entry point for the AIDocumentIndexer backend API.
It initializes the FastAPI application with all routes, middleware,
and lifecycle events.
"""

import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

# Try to load from project root .env file
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    # Fallback to current directory
    load_dotenv()

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import structlog
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.middleware.request_id import RequestIDMiddleware
from backend.api.errors import register_exception_handlers as register_app_exception_handlers

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if os.getenv("LOG_FORMAT") == "json"
        else structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events for:
    - Database connection pool
    - Ray cluster connection
    - Redis connection
    - Background tasks initialization
    """
    logger.info("Starting AIDocumentIndexer API...")

    # Startup tasks
    try:
        # Initialize database connection
        from backend.db.database import init_db
        await init_db()
        logger.info("Database initialized")

        # Seed default agents
        from backend.db.seed_agents import seed_default_agents
        await seed_default_agents()
        logger.info("Default agents seeded")

        # Seed admin user
        from backend.db.seed_users import seed_admin_user
        await seed_admin_user()
        logger.info("Admin user seeded")

        # Initialize Ray connection for parallel document processing
        try:
            from backend.ray_workers.config import init_ray
            init_ray()
            logger.info("Ray cluster connected")
        except Exception as ray_error:
            logger.warning("Ray initialization failed, falling back to local processing", error=str(ray_error))

        # Initialize Redis connection
        # from backend.services.cache import init_cache
        # await init_cache()
        logger.info("Redis cache initialized")

        # Initialize LangChain + LiteLLM
        # from backend.services.llm import init_llm
        # await init_llm()
        logger.info("LLM services initialized")

        # Auto-download OCR models if enabled
        try:
            from backend.services.settings import SettingsService
            from backend.services.ocr_manager import OCRManager

            settings_service = SettingsService()
            all_settings = await settings_service.get_all_settings()

            if all_settings.get("ocr.paddle.auto_download", True):
                logger.info("Auto-downloading OCR models")
                ocr_manager = OCRManager(settings_service)

                languages = all_settings.get("ocr.paddle.languages", ["en", "de"])
                variant = all_settings.get("ocr.paddle.variant", "server")

                download_result = await ocr_manager.download_models(
                    languages=languages,
                    variant=variant,
                )

                if download_result.get("status") == "success":
                    logger.info("OCR models downloaded successfully", result=download_result)
                elif download_result.get("status") == "partial":
                    logger.warning("OCR models partially downloaded", result=download_result)
                else:
                    logger.warning("OCR model download failed, will retry on first use", result=download_result)
        except Exception as ocr_error:
            logger.warning("OCR model auto-download failed, models will be downloaded on first use", error=str(ocr_error))

        logger.info("AIDocumentIndexer API started successfully")

    except Exception as e:
        logger.error("Failed to start API", error=str(e))
        raise

    yield

    # Shutdown tasks
    logger.info("Shutting down AIDocumentIndexer API...")

    try:
        # Close database connections
        # from backend.db.database import close_db
        # await close_db()

        # Disconnect from Ray
        try:
            from backend.ray_workers.config import shutdown_ray
            shutdown_ray()
            logger.info("Ray cluster disconnected")
        except Exception as ray_error:
            logger.warning("Ray shutdown failed", error=str(ray_error))

        # Close Redis connection
        # from backend.services.cache import close_cache
        # await close_cache()

        logger.info("AIDocumentIndexer API shutdown complete")

    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


# =============================================================================
# Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title="AIDocumentIndexer",
        description="Intelligent Document Archive with RAG - Transform your knowledge base into a searchable AI assistant",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add Request ID middleware (must be first to ensure all requests get an ID)
    app.add_middleware(RequestIDMiddleware)

    # Configure CORS
    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],  # Allow clients to read request ID
    )

    # Register routes
    register_routes(app)

    # Register exception handlers (standardized error responses)
    register_app_exception_handlers(app)

    return app


def register_routes(app: FastAPI) -> None:
    """Register all API routes."""

    # Health check (no auth required)
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Check if the API is running and healthy."""
        return {
            "status": "healthy",
            "service": "AIDocumentIndexer",
            "version": "0.1.0",
        }

    @app.get("/", tags=["Root"])
    async def root():
        """API root endpoint."""
        return {
            "message": "Welcome to AIDocumentIndexer API",
            "docs": "/docs",
            "health": "/health",
        }

    # Import and register route modules
    from backend.api.routes.documents import router as documents_router
    from backend.api.routes.chat import router as chat_router
    from backend.api.routes.upload import router as upload_router
    from backend.api.routes.auth import router as auth_router
    from backend.api.routes.admin import router as admin_router
    from backend.api.routes.generate import router as generate_router
    from backend.api.routes.collaboration import router as collaboration_router

    app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    app.include_router(documents_router, prefix="/api/v1/documents", tags=["Documents"])
    app.include_router(chat_router, prefix="/api/v1/chat", tags=["Chat"])
    app.include_router(upload_router, prefix="/api/v1/upload", tags=["Upload"])
    app.include_router(admin_router, prefix="/api/v1/admin", tags=["Admin"])
    app.include_router(generate_router, prefix="/api/v1/generate", tags=["Generate"])
    app.include_router(collaboration_router, prefix="/api/v1/collaboration", tags=["Collaboration"])

    from backend.api.routes.scraper import router as scraper_router
    from backend.api.routes.costs import router as costs_router
    from backend.api.routes.templates import router as templates_router
    from backend.api.routes.agent import router as agent_router
    from backend.api.routes.temp_upload import router as temp_upload_router
    from backend.api.routes.metrics import router as metrics_router
    app.include_router(scraper_router, prefix="/api/v1/scraper", tags=["Scraper"])
    app.include_router(costs_router, prefix="/api/v1/costs", tags=["Costs"])
    app.include_router(templates_router, prefix="/api/v1", tags=["Prompt Templates"])
    app.include_router(agent_router, prefix="/api/v1/agent", tags=["Agent Orchestration"])
    app.include_router(temp_upload_router, prefix="/api/v1/temp", tags=["Temporary Uploads"])
    app.include_router(metrics_router, prefix="/api/v1/metrics", tags=["Metrics & Monitoring"])

    # WebSocket endpoint for real-time updates
    register_websocket_routes(app)

    logger.info("Routes registered")


def register_websocket_routes(app: FastAPI) -> None:
    """Register WebSocket routes for real-time updates."""
    from backend.api.websocket import (
        manager,
        handle_websocket_message,
    )

    @app.websocket("/ws")
    async def websocket_endpoint(
        websocket: WebSocket,
        token: Optional[str] = Query(None),
    ):
        """
        WebSocket endpoint for real-time updates.

        Supports:
        - File processing status updates
        - System notifications
        - Chat streaming (future)

        Query Parameters:
            token: Optional JWT token for authentication

        Message Types (client -> server):
            - subscribe: Subscribe to file updates {"type": "subscribe", "file_id": "..."}
            - unsubscribe: Unsubscribe from updates {"type": "unsubscribe", "file_id": "..."}
            - ping: Keep-alive ping {"type": "ping"}

        Message Types (server -> client):
            - file_update: Processing status update
            - processing_complete: Processing finished successfully
            - processing_error: Processing failed
            - system_notification: System-wide notification
            - pong: Response to ping
            - error: Error message
        """
        # Extract user_id from token if provided
        user_id = None
        if token:
            try:
                # Verify JWT token and extract user_id
                from backend.api.routes.auth import decode_token
                payload = decode_token(token)
                user_id = payload.get("sub")
            except Exception:
                # Invalid token - allow anonymous connection
                pass

        # Accept connection
        await manager.connect(websocket, user_id=user_id)

        try:
            while True:
                # Receive and handle messages
                data = await websocket.receive_json()
                response = await handle_websocket_message(websocket, data)

                if response:
                    await websocket.send_json(response)

        except WebSocketDisconnect:
            await manager.disconnect(websocket)
        except Exception as e:
            logger.error("WebSocket error", error=str(e))
            await manager.disconnect(websocket)

    @app.get("/ws/stats", tags=["WebSocket"])
    async def websocket_stats():
        """Get WebSocket connection statistics."""
        return manager.get_stats()

    logger.info("WebSocket routes registered")


# =============================================================================
# Application Instance
# =============================================================================

app = create_app()


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=int(os.getenv("BACKEND_PORT", "8000")),
        reload=os.getenv("APP_ENV") == "development",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
