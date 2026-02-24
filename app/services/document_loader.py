"""
Document loader service for various file formats.

Supports PDF, TXT, and other common document formats.
"""

import io
from pathlib import Path
from typing import Optional

from app.core.exceptions import DocumentProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentLoader:
    """
    Load and extract text from various document formats.
    """

    @staticmethod
    def load_pdf(file_content: bytes, filename: str = "document.pdf") -> dict:
        """
        Extract text from PDF file.

        Args:
            file_content: PDF file bytes
            filename: Original filename for metadata

        Returns:
            Document dict with content and metadata
        """
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(file_content))

            text_parts = []
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            content = "\n\n".join(text_parts)

            if not content.strip():
                raise DocumentProcessingError(
                    message="PDF appears to be empty or contains only images",
                    details={"filename": filename},
                )

            return {
                "content": content,
                "metadata": {
                    "source": filename,
                    "type": "pdf",
                    "pages": len(reader.pages),
                },
            }

        except ImportError:
            raise DocumentProcessingError(
                message="pypdf not installed",
                details={"install": "pip install pypdf"},
            )
        except Exception as e:
            raise DocumentProcessingError(
                message=f"Failed to process PDF: {str(e)}",
                details={"filename": filename},
            )

    @staticmethod
    def load_text(file_content: bytes, filename: str = "document.txt") -> dict:
        """
        Load plain text file.

        Args:
            file_content: Text file bytes
            filename: Original filename for metadata

        Returns:
            Document dict with content and metadata
        """
        try:
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    content = file_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                content = file_content.decode("utf-8", errors="ignore")

            return {
                "content": content,
                "metadata": {
                    "source": filename,
                    "type": "text",
                },
            }
        except Exception as e:
            raise DocumentProcessingError(
                message=f"Failed to process text file: {str(e)}",
                details={"filename": filename},
            )

    @staticmethod
    def load_markdown(file_content: bytes, filename: str = "document.md") -> dict:
        """Load markdown file as plain text."""
        doc = DocumentLoader.load_text(file_content, filename)
        doc["metadata"]["type"] = "markdown"
        return doc

    @classmethod
    def load_file(
        cls,
        file_content: bytes,
        filename: str,
        content_type: Optional[str] = None,
    ) -> dict:
        """
        Load document based on file extension or content type.

        Args:
            file_content: File bytes
            filename: Original filename
            content_type: MIME type (optional)

        Returns:
            Document dict with content and metadata
        """
        extension = Path(filename).suffix.lower()

        loaders = {
            ".pdf": cls.load_pdf,
            ".txt": cls.load_text,
            ".md": cls.load_markdown,
            ".markdown": cls.load_markdown,
        }

        loader = loaders.get(extension)

        if loader is None:
            if content_type == "application/pdf":
                loader = cls.load_pdf
            elif content_type and content_type.startswith("text/"):
                loader = cls.load_text
            else:
                raise DocumentProcessingError(
                    message=f"Unsupported file format: {extension}",
                    details={
                        "filename": filename,
                        "supported": list(loaders.keys()),
                    },
                )

        logger.info(f"Loading document: {filename}")
        return loader(file_content, filename)
