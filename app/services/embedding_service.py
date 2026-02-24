"""
Embedding service for text vectorization.

Uses sentence-transformers for generating embeddings
with caching and batch processing support.
"""

from typing import Optional

import numpy as np

from app.core.config import get_settings
from app.core.exceptions import AppException
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Generate embeddings for text using sentence-transformers.

    Lazy loads the model on first use to reduce startup time.
    """

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._dimension: Optional[int] = None

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            _ = self.model
        return self._dimension

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"Model loaded successfully. Dimension: {self._dimension}"
            )
        except ImportError:
            raise AppException(
                message="sentence-transformers not installed",
                status_code=500,
                details={"install": "pip install sentence-transformers"},
            )
        except Exception as e:
            raise AppException(
                message=f"Failed to load embedding model: {str(e)}",
                status_code=500,
            )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])

        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise AppException(
                message=f"Failed to generate embeddings: {str(e)}",
                status_code=500,
            )

    def similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.

        Args:
            query_embedding: Query vector
            document_embeddings: Document vectors

        Returns:
            Array of similarity scores
        """
        if len(document_embeddings) == 0:
            return np.array([])

        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = document_embeddings / np.linalg.norm(
            document_embeddings, axis=1, keepdims=True
        )

        similarities = np.dot(doc_norms, query_norm)
        return similarities
