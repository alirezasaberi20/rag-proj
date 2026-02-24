"""
Logging configuration for the application.

Provides structured logging with configurable levels
and formats for different environments.
"""

import logging
import sys
from typing import Optional

from app.core.config import get_settings


def setup_logging(
    level: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom log format string

    Returns:
        Configured root logger
    """
    settings = get_settings()

    log_level = level or settings.log_level
    log_format = format_string or settings.log_format

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    if settings.environment == "production":
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logger = logging.getLogger("app")
    logger.info(
        "Logging configured",
        extra={
            "level": log_level,
            "environment": settings.environment,
        },
    )

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance."""
    return logging.getLogger(f"app.{name}")
