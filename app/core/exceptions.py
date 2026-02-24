"""
Custom exception classes for the application.

Provides structured error handling with proper
HTTP status codes and error messages.
"""

from typing import Any, Optional


class AppException(Exception):
    """Base exception for application errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class LLMServiceError(AppException):
    """Exception raised when LLM service fails."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=503,
            details=details,
        )


class DocumentProcessingError(AppException):
    """Exception raised during document processing."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=400,
            details=details,
        )


class VectorStoreError(AppException):
    """Exception raised for vector store operations."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=500,
            details=details,
        )


class RetrievalError(AppException):
    """Exception raised during document retrieval."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=500,
            details=details,
        )


class ValidationError(AppException):
    """Exception raised for validation failures."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=422,
            details=details,
        )
