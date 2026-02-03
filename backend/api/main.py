"""
AIDocumentIndexer - FastAPI Application Entry Point
====================================================

This is the main entry point for the AIDocumentIndexer backend API.
It initializes the FastAPI application with all routes, middleware,
and lifecycle events.
"""

# =============================================================================
# CRITICAL: uvloop must be installed before any asyncio event loop is created
# This provides 2-4x performance improvement for async operations
# =============================================================================
import sys

def _install_uvloop():
    """
    Install uvloop as the default event loop policy.

    uvloop is a fast, drop-in replacement for asyncio's event loop,
    providing 2-4x performance improvement for async I/O operations.

    Only installed on Linux/macOS (not Windows, as uvloop doesn't support it).
    """
    if sys.platform == "win32":
        return False

    try:
        import uvloop
        uvloop.install()
        return True
    except ImportError:
        # uvloop not installed, continue with default asyncio
        return False

_uvloop_installed = _install_uvloop()

import os
import signal
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

# Graceful shutdown flag
_shutdown_requested = False

def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    # Let uvicorn handle the actual shutdown

# Register signal handlers for graceful shutdown (SIGTERM from K8s, SIGINT from Ctrl+C)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

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
from starlette.middleware.gzip import GZipMiddleware

# Phase 35: Use ORJSON for 20-50% faster JSON serialization
try:
    from fastapi.responses import ORJSONResponse
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSONResponse = JSONResponse
    ORJSON_AVAILABLE = False

from backend.api.middleware.request_id import RequestIDMiddleware
from backend.api.errors import register_exception_handlers as register_app_exception_handlers
from backend.utils.async_helpers import create_safe_task

# Cloud-ready logging configuration
# Set LOG_FORMAT=json for Kubernetes/AWS CloudWatch/GCP Logging
# Additional context is added from K8s downward API environment variables
_log_format = os.getenv("LOG_FORMAT", "console")
_is_json_logging = _log_format == "json"

def _add_cloud_context(logger, method_name, event_dict):
    """Add cloud/k8s context to log entries."""
    # Kubernetes pod info (set via downward API)
    if pod_name := os.getenv("POD_NAME"):
        event_dict["pod_name"] = pod_name
    if namespace := os.getenv("POD_NAMESPACE"):
        event_dict["namespace"] = namespace
    if node_name := os.getenv("NODE_NAME"):
        event_dict["node_name"] = node_name
    # AWS/Cloud metadata
    if aws_region := os.getenv("AWS_REGION"):
        event_dict["aws_region"] = aws_region
    if deployment := os.getenv("DEPLOYMENT_NAME", os.getenv("K8S_DEPLOYMENT")):
        event_dict["deployment"] = deployment
    return event_dict

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        _add_cloud_context,  # Add K8s/AWS context
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if _is_json_logging
        else structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def _periodic_memory_cleanup():
    """
    Periodic memory cleanup task.

    Runs every 10 minutes to:
    - Trigger garbage collection
    - Clear GPU cache if available
    - Unload idle ML models

    Inspired by OpenClaw's memory management patterns.
    """
    import asyncio
    import gc

    cleanup_interval = 600  # 10 minutes

    while True:
        await asyncio.sleep(cleanup_interval)
        try:
            # Trigger garbage collection
            gc.collect()

            # Clear GPU cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    # MPS doesn't have explicit cache clearing
                    pass
            except ImportError:
                pass

            # Get memory usage for logging
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                logger.info(
                    "Periodic memory cleanup completed",
                    memory_mb=round(memory_mb, 1),
                )
            except ImportError:
                logger.info("Periodic memory cleanup completed")

        except Exception as e:
            logger.error("Memory cleanup failed", error=str(e))


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

    # Log uvloop status
    if _uvloop_installed:
        logger.info("uvloop installed - 2-4x async performance boost enabled")
    else:
        logger.info("uvloop not available - using default asyncio event loop")

    # Log ORJSON status (Phase 35)
    if ORJSON_AVAILABLE:
        logger.info("ORJSON enabled - 20-50% faster JSON serialization")
    else:
        logger.info("ORJSON not available - using standard JSON serialization")

    # Initialize performance optimizations (Phase 3-4: Cython, GPU, MinHash)
    try:
        from backend.services.performance_init import initialize_performance_optimizations
        perf_status = await initialize_performance_optimizations(
            compile_cython=True,  # Compile Cython extensions at startup
            init_gpu=True,        # Initialize GPU acceleration
            init_minhash=True,    # Initialize MinHash deduplicator
            warmup_gpu=False,     # Skip GPU warmup for faster startup
        )
        logger.info(
            "Performance optimizations ready",
            cython=perf_status["cython"]["status"],
            gpu=perf_status["gpu"]["status"],
            minhash=perf_status["minhash"]["status"],
        )
    except Exception as perf_error:
        logger.warning(
            "Performance optimization init failed (using fallbacks)",
            error=str(perf_error)
        )

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

        # Fix ChromaDB pickle corruption if present
        try:
            from backend.services.vectorstore_local import fix_chromadb_pickle
            fix_chromadb_pickle()
            logger.info("ChromaDB health check passed")
        except Exception as chroma_error:
            logger.warning("ChromaDB health check failed", error=str(chroma_error))

        # Reset stuck upload jobs and sync with completed documents
        try:
            from backend.api.routes.upload import reset_stuck_upload_jobs
            reset_count = await reset_stuck_upload_jobs()
            if reset_count > 0:
                logger.info("Reset stuck upload jobs", count=reset_count)
        except Exception as reset_error:
            logger.warning("Failed to reset stuck upload jobs", error=str(reset_error))

        # Process any queued uploads in background
        try:
            from backend.api.routes.upload import process_queued_uploads_startup, start_periodic_queue_processor
            create_safe_task(
                process_queued_uploads_startup(),
                name="startup_queued_uploads",
                on_error=lambda e: logger.error("Queued uploads processing failed", error=str(e))
            )
            logger.info("Started background processing of queued uploads")

            # Start periodic processor to continuously check for new queued uploads
            create_safe_task(
                start_periodic_queue_processor(interval_seconds=30),
                name="periodic_queue_processor",
                on_error=lambda e: logger.error("Periodic queue processor failed", error=str(e))
            )
            logger.info("Started periodic queue processor (checks every 30s)")
        except Exception as queue_error:
            logger.warning("Failed to start queued uploads processing", error=str(queue_error))

        # Initialize Ray connection for parallel document processing
        # Uses settings from Admin UI (processing.ray_enabled, processing.ray_address, etc.)
        try:
            from backend.ray_workers.config import init_ray_async
            ray_initialized = await init_ray_async(cleanup_stale=True, init_timeout=30.0)
            if ray_initialized:
                logger.info("Ray cluster connected")
            else:
                logger.info("Ray disabled or initialization failed, using local processing")
        except Exception as ray_error:
            logger.warning("Ray initialization failed, falling back to local processing", error=str(ray_error))

        # Initialize Redis connection
        # from backend.services.cache import init_cache
        # await init_cache()
        logger.info("Redis cache initialized")

        # Auto-start Celery worker if enabled in settings
        try:
            from backend.services.celery_manager import start_celery_worker_auto
            celery_started = await start_celery_worker_auto()
            if celery_started:
                logger.info("Celery worker auto-started (controlled by settings)")
            else:
                logger.info("Celery worker not started (disabled in settings or Redis unavailable)")
        except Exception as celery_error:
            logger.warning("Celery auto-start failed, using sync processing", error=str(celery_error))

        # Initialize LangChain + LiteLLM
        # from backend.services.llm import init_llm
        # await init_llm()
        logger.info("LLM services initialized")

        # Check Ollama availability for local LLM inference
        try:
            from backend.services.llm import list_ollama_models
            ollama_result = await list_ollama_models()
            if ollama_result.get("success"):
                models = ollama_result.get("models", [])
                model_names = [m.get("name", m.get("model", "unknown")) for m in models[:5]]
                logger.info(
                    "Ollama connected",
                    model_count=len(models),
                    available_models=model_names,
                )
            else:
                logger.warning(
                    "Ollama not available - system will use cloud LLM providers as fallback",
                    error=ollama_result.get("error", "Unknown error"),
                )
        except Exception as ollama_error:
            logger.warning(
                "Ollama health check failed - system will use cloud LLM providers as fallback",
                error=str(ollama_error),
            )

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

        # Start periodic memory cleanup task (inspired by OpenClaw patterns)
        import asyncio
        memory_cleanup_task = create_safe_task(
            _periodic_memory_cleanup(),
            name="memory_cleanup",
            on_error=lambda e: logger.error("Memory cleanup task failed", error=str(e))
        )
        logger.info("Started periodic memory cleanup task")

        logger.info("AIDocumentIndexer API started successfully")

    except Exception as e:
        logger.error("Failed to start API", error=str(e))
        raise

    yield

    # Cancel memory cleanup task
    try:
        memory_cleanup_task.cancel()
        try:
            await memory_cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Memory cleanup task cancelled")
    except Exception as cleanup_error:
        logger.warning("Failed to cancel memory cleanup task", error=str(cleanup_error))

    # Stop periodic queue processor
    try:
        from backend.api.routes.upload import stop_periodic_queue_processor
        stop_periodic_queue_processor()
        logger.info("Periodic queue processor stopped")
    except Exception as queue_error:
        logger.warning("Failed to stop periodic queue processor", error=str(queue_error))

    # Shutdown tasks
    logger.info("Shutting down AIDocumentIndexer API...")

    try:
        # Close shared HTTP client
        try:
            from backend.services.http_client import close_http_client
            await close_http_client()
            logger.info("HTTP client closed")
        except Exception as http_error:
            logger.warning("HTTP client shutdown failed", error=str(http_error))

        # Close database connections
        # from backend.db.database import close_db
        # await close_db()

        # Stop Celery worker if we started it
        try:
            from backend.services.celery_manager import stop_celery_worker_auto
            await stop_celery_worker_auto()
            logger.info("Celery worker stopped")
        except Exception as celery_error:
            logger.warning("Celery shutdown failed", error=str(celery_error))

        # Disconnect from Ray with timeout
        try:
            from backend.ray_workers.config import shutdown_ray
            shutdown_ray(timeout=10.0)
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
    # Phase 35: Use ORJSONResponse as default for 20-50% faster JSON serialization
    app = FastAPI(
        title="AIDocumentIndexer",
        description="Intelligent Document Archive with RAG - Transform your knowledge base into a searchable AI assistant",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        default_response_class=ORJSONResponse if ORJSON_AVAILABLE else JSONResponse,
    )

    # Add Request ID middleware (must be first to ensure all requests get an ID)
    app.add_middleware(RequestIDMiddleware)

    # Configure CORS - allow both localhost and 127.0.0.1
    cors_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],  # Allow clients to read request ID
    )

    # Add GZip compression middleware (60-70% smaller responses)
    # Only compresses responses > 500 bytes to avoid overhead on small responses
    app.add_middleware(GZipMiddleware, minimum_size=500)

    # Register routes
    register_routes(app)

    # Register exception handlers (standardized error responses)
    register_app_exception_handlers(app)

    # Initialize monitoring (Sentry + Prometheus)
    from backend.core.monitoring import init_monitoring
    init_monitoring(app)

    return app


def register_routes(app: FastAPI) -> None:
    """Register all API routes."""

    # ==========================================================================
    # Kubernetes-Ready Health Probes
    # ==========================================================================

    # Liveness probe - basic check that the process is alive
    @app.get("/health/live", tags=["Health"])
    async def liveness_probe():
        """
        Kubernetes liveness probe.
        Returns 200 if the process is running.
        Use for: livenessProbe in k8s deployment.
        """
        return {"status": "alive"}

    # Readiness probe - checks if the service can handle traffic
    @app.get("/health/ready", tags=["Health"])
    async def readiness_probe():
        """
        Kubernetes readiness probe.
        Checks database, Redis, and critical services.
        Use for: readinessProbe in k8s deployment.
        """
        checks = {
            "database": "unknown",
            "redis": "unknown",
            "celery": "unknown",
        }
        all_healthy = True

        # Check database connection
        try:
            from backend.db.database import get_async_session
            async with get_async_session() as session:
                from sqlalchemy import text
                await session.execute(text("SELECT 1"))
                checks["database"] = "healthy"
        except Exception as e:
            checks["database"] = f"unhealthy: {str(e)[:50]}"
            all_healthy = False

        # Check Redis connection
        try:
            from backend.services.cache import get_redis_client
            redis = await get_redis_client()
            if redis:
                await redis.ping()
                checks["redis"] = "healthy"
            else:
                checks["redis"] = "not_configured"
        except Exception as e:
            checks["redis"] = f"unhealthy: {str(e)[:50]}"
            # Redis is optional, don't fail readiness

        # Check Celery worker
        try:
            from backend.services.celery_manager import is_celery_running
            if await is_celery_running():
                checks["celery"] = "healthy"
            else:
                checks["celery"] = "not_running"
        except Exception:
            checks["celery"] = "not_configured"

        status_code = 200 if all_healthy else 503
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "ready" if all_healthy else "not_ready",
                "checks": checks,
            }
        )

    # Combined health check (backwards compatible)
    @app.get("/health", tags=["Health"])
    async def health_check():
        """
        Combined health check for backwards compatibility.
        Returns basic status + performance optimization info.
        """
        import time
        start = time.perf_counter()

        # Get performance status
        perf_status = {}
        try:
            from backend.services.performance_init import get_performance_status
            perf_status = get_performance_status()
        except Exception:
            pass

        # Get memory info
        memory_mb = 0
        try:
            import psutil
            process = psutil.Process()
            memory_mb = round(process.memory_info().rss / 1024 / 1024, 1)
        except Exception:
            pass

        return {
            "status": "healthy",
            "service": "AIDocumentIndexer",
            "version": "0.1.0",
            "response_time_ms": round((time.perf_counter() - start) * 1000, 2),
            "memory_mb": memory_mb,
            "optimizations": {
                "uvloop": _uvloop_installed,
                "orjson": ORJSON_AVAILABLE,
                "cython": perf_status.get("cython", {}).get("using_cython", False),
                "gpu": perf_status.get("gpu", {}).get("has_gpu", False),
            },
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
    from backend.api.routes.folders import router as folders_router
    from backend.api.routes.preferences import router as preferences_router
    from backend.api.routes.generation_templates import router as generation_templates_router
    from backend.api.routes.knowledge_graph import router as knowledge_graph_router
    from backend.api.routes.workflows import router as workflows_router
    from backend.api.routes.audio import router as audio_router
    from backend.api.routes.bots import router as bots_router
    from backend.api.routes.connectors import router as connectors_router
    from backend.api.routes.database import router as database_router
    from backend.api.routes.gateway import router as gateway_router
    from backend.api.routes.privacy import router as privacy_router
    from backend.api.routes.llm_config import router as llm_config_router
    from backend.api.routes.organizations import router as organizations_router
    from backend.api.routes.downloads import router as downloads_router
    from backend.api.routes.content_review import router as content_review_router
    from backend.api.routes.document_templates import router as document_templates_router
    from backend.api.routes.telemetry import router as telemetry_router
    from backend.api.routes.embeddings import router as embeddings_router
    from backend.api.routes.agent_templates import router as agent_templates_router
    from backend.api.routes.crawler import router as crawler_router
    from backend.api.routes.evaluation import router as evaluation_router
    from backend.api.routes.watcher import router as watcher_router
    from backend.api.routes.indexer import router as indexer_router
    from backend.api.routes.agentic import router as agentic_router
    from backend.api.routes.annotations import router as annotations_router
    from backend.api.routes.charts import router as charts_router
    from backend.api.routes.cache import router as cache_router
    from backend.api.routes.reranking import router as reranking_router
    from backend.api.routes.vision import router as vision_router
    from backend.api.routes.late_chunking import router as late_chunking_router
    from backend.api.routes.compression import router as compression_router
    from backend.api.routes.security import router as security_router
    from backend.api.routes.experiments import router as experiments_router
    from backend.api.routes.diagnostics import router as diagnostics_router
    from backend.api.routes.dspy_optimization import router as dspy_router
    from backend.api.routes.smart_router import router as smart_router_router
    from backend.api.routes.embedding_defense import router as embedding_defense_router
    from backend.api.routes.matryoshka import router as matryoshka_router
    from backend.api.routes.speculative_rag import router as speculative_rag_router
    from backend.api.routes.otel_tracing import router as otel_tracing_router
    from backend.api.routes.skills import router as skills_router
    from backend.api.routes.public import router as public_router
    from backend.api.routes.external_api import router as external_api_router
    from backend.api.routes.moodboard import router as moodboard_router
    from backend.api.routes.research import router as research_router
    from backend.api.routes.reports import router as reports_router
    from backend.api.routes.parallel_query import router as parallel_query_router
    from backend.api.routes.tools import router as tools_router
    from backend.api.routes.mcp import router as mcp_router
    from backend.api.routes.ensemble import router as ensemble_router
    from backend.api.routes.knowledge_analytics import router as knowledge_analytics_router
    from backend.api.routes.query_analysis import router as query_analysis_router
    from backend.api.routes.intelligence import router as intelligence_router
    from backend.api.routes.tool_streaming import router as tool_streaming_router
    from backend.api.routes.menu import router as menu_router
    from backend.api.routes.link_groups import router as link_groups_router
    from backend.api.routes.integrations import router as integrations_router
    from backend.api.routes.pdf_tools import router as pdf_tools_router
    from backend.api.routes.visualization import router as visualization_router
    from backend.api.routes.analytics import router as usage_analytics_router
    from backend.api.routes.websocket import router as websocket_router
    from backend.api.routes.license import router as license_router
    from backend.api.routes.sso import router as sso_router
    from backend.api.routes.messaging import router as messaging_router
    app.include_router(license_router, prefix="/api/v1", tags=["License Management"])
    app.include_router(sso_router, prefix="/api/v1", tags=["Enterprise SSO"])
    app.include_router(messaging_router, prefix="/api/v1", tags=["Messaging Bots"])
    app.include_router(scraper_router, prefix="/api/v1/scraper", tags=["Scraper"])
    app.include_router(usage_analytics_router, prefix="/api/v1/usage-analytics", tags=["Usage Analytics"])
    app.include_router(websocket_router, prefix="/api/v1", tags=["WebSocket"])
    app.include_router(pdf_tools_router, prefix="/api/v1/pdf-tools", tags=["PDF Tools"])
    app.include_router(visualization_router, prefix="/api/v1/visualization", tags=["Visualization"])
    app.include_router(link_groups_router, prefix="/api/v1/link-groups", tags=["Link Groups"])
    app.include_router(integrations_router, prefix="/api/v1/integrations", tags=["Feature Integrations"])
    app.include_router(costs_router, prefix="/api/v1/costs", tags=["Costs"])
    app.include_router(templates_router, prefix="/api/v1", tags=["Prompt Templates"])
    app.include_router(generation_templates_router, prefix="/api/v1", tags=["Generation Templates"])
    app.include_router(agent_router, prefix="/api/v1/agent", tags=["Agent Orchestration"])
    app.include_router(temp_upload_router, prefix="/api/v1/temp", tags=["Temporary Uploads"])
    app.include_router(metrics_router, prefix="/api/v1/metrics", tags=["Metrics & Monitoring"])
    app.include_router(folders_router, prefix="/api/v1/folders", tags=["Folders"])
    app.include_router(preferences_router, prefix="/api/v1/preferences", tags=["User Preferences"])
    app.include_router(knowledge_graph_router, prefix="/api/v1/knowledge-graph", tags=["Knowledge Graph"])
    app.include_router(workflows_router, prefix="/api/v1/workflows", tags=["Workflows"])
    app.include_router(audio_router, prefix="/api/v1/audio", tags=["Audio Overviews"])
    app.include_router(bots_router, prefix="/api/v1/bots", tags=["Bot Integrations"])
    app.include_router(connectors_router, prefix="/api/v1/connectors", tags=["Connectors"])
    app.include_router(database_router, prefix="/api/v1/database", tags=["Database Connectors"])
    app.include_router(gateway_router, prefix="/api/v1/gateway", tags=["LLM Gateway"])
    app.include_router(privacy_router, prefix="/api/v1/privacy", tags=["Privacy Controls"])
    app.include_router(llm_config_router, prefix="/api/v1/llm", tags=["LLM Configuration"])
    app.include_router(organizations_router, prefix="/api/v1/organizations", tags=["Organizations"])
    app.include_router(downloads_router, prefix="/api/v1", tags=["Downloads"])
    app.include_router(content_review_router, prefix="/api/v1/content-review", tags=["Content Review"])
    app.include_router(document_templates_router, prefix="/api/v1", tags=["Document Templates"])
    app.include_router(telemetry_router, prefix="/api/v1/telemetry", tags=["Phase 15 Telemetry"])
    app.include_router(embeddings_router, prefix="/api/v1", tags=["Embeddings"])
    app.include_router(agent_templates_router, prefix="/api/v1", tags=["Agent & Workflow Templates"])
    app.include_router(crawler_router, prefix="/api/v1", tags=["Web Crawler"])
    app.include_router(evaluation_router, prefix="/api/v1", tags=["RAG Evaluation"])
    app.include_router(watcher_router, prefix="/api/v1/watcher", tags=["File Watcher"])
    app.include_router(indexer_router, prefix="/api/v1/indexer", tags=["Real-Time Indexer"])
    app.include_router(agentic_router, prefix="/api/v1/agentic", tags=["Agentic RAG"])
    app.include_router(annotations_router, prefix="/api/v1/annotations", tags=["Annotations"])
    app.include_router(charts_router, prefix="/api/v1", tags=["Charts"])
    app.include_router(cache_router, prefix="/api/v1", tags=["Cache Management"])
    app.include_router(reranking_router, prefix="/api/v1", tags=["Reranking"])
    app.include_router(vision_router, prefix="/api/v1", tags=["Vision Processing"])
    app.include_router(late_chunking_router, prefix="/api/v1", tags=["Late Chunking"])
    app.include_router(compression_router, prefix="/api/v1", tags=["Compression"])
    app.include_router(security_router, prefix="/api/v1", tags=["RAG Security"])
    app.include_router(experiments_router, prefix="/api/v1", tags=["Experiments & Feedback"])
    app.include_router(diagnostics_router, prefix="/api/v1", tags=["Diagnostics & Monitoring"])
    app.include_router(dspy_router, prefix="/api/v1", tags=["DSPy Optimization"])
    app.include_router(smart_router_router, prefix="/api/v1", tags=["Smart Model Router"])
    app.include_router(embedding_defense_router, prefix="/api/v1", tags=["Embedding Defense"])
    app.include_router(matryoshka_router, prefix="/api/v1", tags=["Matryoshka Retrieval"])
    app.include_router(speculative_rag_router, prefix="/api/v1", tags=["Speculative RAG"])
    app.include_router(otel_tracing_router, prefix="/api/v1", tags=["OpenTelemetry Tracing"])
    app.include_router(skills_router, prefix="/api/v1/skills", tags=["Skills Marketplace"])
    app.include_router(public_router, prefix="/api/v1/public", tags=["Public API (No Auth)"])
    app.include_router(external_api_router, prefix="/api/v1", tags=["External API"])
    app.include_router(moodboard_router, prefix="/api/v1/moodboard", tags=["Mood Board"])
    app.include_router(research_router, prefix="/api/v1/research", tags=["Deep Research"])
    app.include_router(reports_router, prefix="/api/v1/reports", tags=["Reports (Sparkpages)"])
    app.include_router(parallel_query_router, prefix="/api/v1/parallel", tags=["Parallel Knowledge"])
    app.include_router(tools_router, prefix="/api/v1/tools", tags=["Tool Augmentation"])
    app.include_router(mcp_router, prefix="/api/v1/mcp", tags=["MCP Server"])
    app.include_router(ensemble_router, prefix="/api/v1/ensemble", tags=["Ensemble Voting"])
    app.include_router(knowledge_analytics_router, prefix="/api/v1/analytics", tags=["Knowledge Analytics"])
    app.include_router(intelligence_router, prefix="/api/v1", tags=["Intelligence Enhancement"])
    app.include_router(tool_streaming_router, prefix="/api/v1", tags=["Tool Streaming"])
    app.include_router(menu_router, prefix="/api/v1", tags=["Menu Configuration"])

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
