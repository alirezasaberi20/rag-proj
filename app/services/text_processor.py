"""
Text processing service for document chunking.

Handles splitting documents into smaller chunks
for embedding and retrieval.
"""

import re
from typing import Generator

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class TextProcessor:
    """
    Process and chunk text documents for RAG pipeline.

    Implements recursive character splitting with overlap
    to maintain context across chunks.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.

        Args:
            text: Raw text content

        Returns:
            Cleaned text
        """
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text

    def split_text(self, text: str) -> list[str]:
        """
        Split text into chunks with overlap.

        Uses recursive splitting with multiple separators
        to create semantically meaningful chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        text = self.clean_text(text)

        if len(text) <= self.chunk_size:
            return [text] if text else []

        chunks = self._recursive_split(text, self.separators)
        return self._merge_chunks(chunks)

    def _recursive_split(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """Recursively split text using separators."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return list(text)

        splits = text.split(separator)

        chunks = []
        for split in splits:
            if len(split) <= self.chunk_size:
                chunks.append(split)
            else:
                chunks.extend(
                    self._recursive_split(split, remaining_separators)
                )

        return chunks

    def _merge_chunks(self, chunks: list[str]) -> list[str]:
        """Merge small chunks and add overlap."""
        if not chunks:
            return []

        merged = []
        current_chunk = ""

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            if len(current_chunk) + len(chunk) + 1 <= self.chunk_size:
                current_chunk = f"{current_chunk} {chunk}".strip()
            else:
                if current_chunk:
                    merged.append(current_chunk)
                current_chunk = chunk

        if current_chunk:
            merged.append(current_chunk)

        if self.chunk_overlap > 0 and len(merged) > 1:
            merged = self._add_overlap(merged)

        return merged

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between consecutive chunks."""
        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            overlap_text = prev_chunk[-self.chunk_overlap :]
            if not overlap_text.startswith(" "):
                space_idx = overlap_text.find(" ")
                if space_idx != -1:
                    overlap_text = overlap_text[space_idx + 1 :]

            overlapped = f"{overlap_text} {current_chunk}".strip()
            result.append(overlapped)

        return result

    def process_documents(
        self,
        documents: list[dict],
    ) -> Generator[dict, None, None]:
        """
        Process multiple documents into chunks.

        Args:
            documents: List of documents with content and metadata

        Yields:
            Document chunks with metadata
        """
        for doc_idx, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            chunks = self.split_text(content)

            for chunk_idx, chunk in enumerate(chunks):
                yield {
                    "content": chunk,
                    "metadata": {
                        **metadata,
                        "doc_index": doc_idx,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                    },
                }

        logger.info(f"Processed {len(documents)} documents")
