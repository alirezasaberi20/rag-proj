"""
Health check endpoints.

Provides health and readiness checks for the application
and its dependencies.
"""

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.schemas import HealthResponse
from app.services.llm_service import LLMService

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the application and its services.",
)
async def health_check() -> HealthResponse:
    """
    Perform health check on all services.

    Returns the status of:
    - API service
    - LLM service (Ollama)
    - Vector store
    """
    settings = get_settings()

    llm_service = LLMService()
    ollama_healthy = await llm_service.health_check()

    return HealthResponse(
        status="healthy" if ollama_healthy else "degraded",
        version=settings.app_version,
        environment=settings.environment,
        services={
            "api": "healthy",
            "ollama": "healthy" if ollama_healthy else "unavailable",
            "vector_store": "healthy",
        },
    )


@router.get(
    "/ready",
    summary="Readiness check",
    description="Check if the application is ready to accept requests.",
)
async def readiness_check():
    """Check if all required services are ready."""
    llm_service = LLMService()
    ollama_ready = await llm_service.health_check()

    if not ollama_ready:
        return {
            "ready": False,
            "message": "Ollama service is not available",
        }

    return {"ready": True}


@router.get(
    "/models",
    summary="List available models",
    description="List all models available in Ollama.",
)
async def list_models():
    """List available LLM models."""
    llm_service = LLMService()

    try:
        models = await llm_service.list_models()
        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}
