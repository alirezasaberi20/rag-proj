"""
API routes for the chatbot service.

Defines all HTTP endpoints with proper documentation
and error handling. Supports user isolation.
"""

import time
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile, File, status

from app.api.dependencies import CurrentUser
from app.core.exceptions import AppException
from app.core.logging import get_logger
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    CollectionStats,
    FileUploadResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceDocument,
)
from app.models.user import User
from app.services.document_loader import DocumentLoader
from app.services.rag_engine import RAGEngine

logger = get_logger(__name__)

router = APIRouter()

_user_engines: dict[str, RAGEngine] = {}


def get_rag_engine_for_user(user: User) -> RAGEngine:
    """
    Get or create a RAG engine instance for a specific user.

    Each user gets their own isolated vector store collection.
    """
    if user.id not in _user_engines:
        _user_engines[user.id] = RAGEngine(
            collection_name=f"documents_user_{user.id}"
        )
        logger.info(f"Created RAG engine for user: {user.username}")

    return _user_engines[user.id]


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a chat message",
    description="Send a message and receive an AI-generated response with relevant sources.",
)
async def chat(request: ChatRequest, current_user: CurrentUser) -> ChatResponse:
    """
    Process a chat message using RAG pipeline.

    - Retrieves relevant documents from the user's knowledge base
    - Generates a response using the LLM
    - Returns the response with source documents
    """
    try:
        engine = get_rag_engine_for_user(current_user)
        result = await engine.query(
            question=request.message,
        )

        sources = []
        if request.include_sources:
            sources = [
                SourceDocument(
                    content=doc["content"],
                    score=doc["score"],
                    metadata=doc["metadata"],
                )
                for doc in result["sources"]
            ]

        return ChatResponse(
            conversation_id=request.conversation_id or uuid4(),
            message=result["answer"],
            sources=sources,
            processing_time_ms=result["processing_time_ms"],
        )

    except AppException as e:
        logger.error(f"Chat error: {e.message}")
        raise HTTPException(
            status_code=e.status_code,
            detail={"error": e.message, "details": e.details},
        )
    except Exception as e:
        logger.exception("Unexpected error in chat endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "message": str(e)},
        )


@router.post(
    "/chat/direct",
    response_model=ChatResponse,
    summary="Direct chat without RAG",
    description="Send a message directly to the LLM without document retrieval.",
)
async def chat_direct(request: ChatRequest, current_user: CurrentUser) -> ChatResponse:
    """Chat directly with LLM without RAG retrieval."""
    try:
        engine = get_rag_engine_for_user(current_user)
        result = await engine.query_without_rag(
            question=request.message,
        )

        return ChatResponse(
            conversation_id=request.conversation_id or uuid4(),
            message=result["answer"],
            sources=[],
            processing_time_ms=result["processing_time_ms"],
        )

    except AppException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={"error": e.message, "details": e.details},
        )


@router.post(
    "/documents/ingest",
    response_model=IngestResponse,
    summary="Ingest documents",
    description="Add documents to the user's knowledge base for RAG retrieval.",
)
async def ingest_documents(request: IngestRequest, current_user: CurrentUser) -> IngestResponse:
    """
    Ingest documents into the user's vector store.

    Documents are chunked, embedded, and stored for retrieval.
    Each user has their own isolated document collection.
    """
    try:
        engine = get_rag_engine_for_user(current_user)

        documents = [
            {"content": doc.content, "metadata": doc.metadata}
            for doc in request.documents
        ]

        result = await engine.ingest_documents(documents)

        return IngestResponse(
            ingested_count=result["ingested_count"],
            chunk_count=result["chunk_count"],
            processing_time_ms=result["processing_time_ms"],
        )

    except AppException as e:
        logger.error(f"Ingestion error: {e.message}")
        raise HTTPException(
            status_code=e.status_code,
            detail={"error": e.message, "details": e.details},
        )
    except Exception as e:
        logger.exception("Unexpected error in ingest endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "message": str(e)},
        )


@router.get(
    "/documents/stats",
    response_model=CollectionStats,
    summary="Get document statistics",
    description="Get statistics about the user's document collection.",
)
async def get_document_stats(current_user: CurrentUser) -> CollectionStats:
    """Get statistics about the user's vector store collection."""
    try:
        engine = get_rag_engine_for_user(current_user)
        stats = engine.get_stats()

        return CollectionStats(
            name=stats["vector_store"]["name"],
            document_count=stats["vector_store"]["document_count"],
            chunk_count=stats["vector_store"]["document_count"],
        )

    except AppException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={"error": e.message},
        )


@router.delete(
    "/documents",
    summary="Clear documents",
    description="Delete all documents from the user's knowledge base.",
)
async def clear_documents(current_user: CurrentUser):
    """Delete all documents from the user's vector store."""
    try:
        engine = get_rag_engine_for_user(current_user)
        engine.vector_store.delete_collection()
        return {"message": "Collection deleted successfully"}

    except AppException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail={"error": e.message},
        )


@router.post(
    "/documents/upload",
    response_model=FileUploadResponse,
    summary="Upload a document file",
    description="Upload a PDF, TXT, or Markdown file to the user's knowledge base.",
)
async def upload_document(
    current_user: CurrentUser,
    file: UploadFile = File(..., description="Document file (PDF, TXT, MD)"),
):
    """
    Upload and process a document file.

    Supported formats:
    - PDF (.pdf)
    - Plain text (.txt)
    - Markdown (.md)

    The document will be chunked and added to the user's knowledge base.
    """
    start_time = time.time()

    try:
        content = await file.read()

        doc = DocumentLoader.load_file(
            file_content=content,
            filename=file.filename or "uploaded_file",
            content_type=file.content_type,
        )

        engine = get_rag_engine_for_user(current_user)
        result = await engine.ingest_documents([doc])

        processing_time = (time.time() - start_time) * 1000

        return FileUploadResponse(
            filename=file.filename or "uploaded_file",
            content_type=file.content_type or "unknown",
            chunk_count=result["chunk_count"],
            processing_time_ms=processing_time,
        )

    except AppException as e:
        logger.error(f"Upload error: {e.message}")
        raise HTTPException(
            status_code=e.status_code,
            detail={"error": e.message, "details": e.details},
        )
    except Exception as e:
        logger.exception("Unexpected error in upload endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "message": str(e)},
        )


@router.post(
    "/documents/upload-multiple",
    summary="Upload multiple document files",
    description="Upload multiple PDF, TXT, or Markdown files at once.",
)
async def upload_multiple_documents(
    current_user: CurrentUser,
    files: list[UploadFile] = File(..., description="Document files"),
):
    """Upload and process multiple document files for the user."""
    start_time = time.time()
    results = []
    errors = []

    engine = get_rag_engine_for_user(current_user)

    for file in files:
        try:
            content = await file.read()
            doc = DocumentLoader.load_file(
                file_content=content,
                filename=file.filename or "uploaded_file",
                content_type=file.content_type,
            )

            result = await engine.ingest_documents([doc])

            results.append({
                "filename": file.filename,
                "chunk_count": result["chunk_count"],
                "status": "success",
            })
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed",
            })

    processing_time = (time.time() - start_time) * 1000

    return {
        "processed": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors,
        "processing_time_ms": processing_time,
    }
