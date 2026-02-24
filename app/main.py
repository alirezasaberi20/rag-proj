"""
FastAPI application entry point.

Configures the application with middleware, exception handlers,
and route registration.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.exceptions import AppException
from app.core.logging import setup_logging, get_logger
from app.api.v1 import routes, health, auth

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    setup_logging()
    logger.info(
        f"Starting {settings.app_name} v{settings.app_version} "
        f"in {settings.environment} mode"
    )
    yield
    logger.info("Shutting down application")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    RAG-based Chatbot API

    A production-ready Retrieval-Augmented Generation chatbot API
    that uses local LLMs via Ollama for on-premise deployment.

    ## Features

    - **Document Ingestion**: Upload documents to build a knowledge base
    - **RAG Chat**: Ask questions and get answers based on your documents
    - **Direct Chat**: Chat directly with the LLM without RAG
    - **Health Monitoring**: Check service health and readiness

    ## Architecture

    - **LLM**: Ollama with configurable models
    - **Embeddings**: Sentence Transformers
    - **Vector Store**: ChromaDB

    ## Usage

    1. Start Ollama: `ollama serve`
    2. Pull a model: `ollama pull tinyllama`
    3. Ingest documents via `/api/v1/documents/ingest`
    4. Chat via `/api/v1/chat`
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """Handle application-specific exceptions."""
    logger.error(
        f"AppException: {exc.message}",
        extra={"details": exc.details, "path": request.url.path},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": {"type": type(exc).__name__} if settings.debug else {},
        },
    )


app.include_router(health.router, tags=["Health"])

app.include_router(
    auth.router,
    prefix=settings.api_prefix,
    tags=["Authentication"],
)

app.include_router(
    routes.router,
    prefix=settings.api_prefix,
    tags=["Chat", "Documents"],
)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
    }
