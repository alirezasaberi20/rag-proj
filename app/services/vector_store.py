"""
Vector store service for document storage and retrieval.

Uses a simple NumPy-based vector store for compatibility
with all Python versions.
"""

import json
import os
from typing import Optional
from uuid import uuid4

import numpy as np

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    NumPy-based vector store for document embeddings.

    Provides persistent storage with cosine similarity search.
    Simple but effective for small-medium datasets.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ):
        settings = get_settings()
        self.collection_name = collection_name or settings.collection_name
        self.persist_directory = persist_directory or settings.vector_store_path
        
        self._documents: list[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._metadatas: list[dict] = []
        self._ids: list[str] = []
        
        self._load()

    def _get_storage_path(self) -> str:
        """Get path to storage file."""
        os.makedirs(self.persist_directory, exist_ok=True)
        return os.path.join(self.persist_directory, f"{self.collection_name}.npz")

    def _get_metadata_path(self) -> str:
        """Get path to metadata file."""
        return os.path.join(self.persist_directory, f"{self.collection_name}_meta.json")

    def _load(self) -> None:
        """Load data from disk if exists."""
        storage_path = self._get_storage_path()
        metadata_path = self._get_metadata_path()

        if os.path.exists(storage_path) and os.path.exists(metadata_path):
            try:
                data = np.load(storage_path, allow_pickle=True)
                self._embeddings = data["embeddings"]
                self._documents = data["documents"].tolist()
                self._ids = data["ids"].tolist()

                with open(metadata_path, "r") as f:
                    self._metadatas = json.load(f)

                logger.info(
                    f"Loaded {len(self._documents)} documents from {storage_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to load existing data: {e}")
                self._documents = []
                self._embeddings = None
                self._metadatas = []
                self._ids = []

    def persist(self) -> None:
        """Persist data to disk."""
        if not self._documents:
            return

        try:
            storage_path = self._get_storage_path()
            metadata_path = self._get_metadata_path()

            np.savez(
                storage_path,
                embeddings=self._embeddings,
                documents=np.array(self._documents, dtype=object),
                ids=np.array(self._ids, dtype=object),
            )

            with open(metadata_path, "w") as f:
                json.dump(self._metadatas, f)

            logger.info(f"Persisted {len(self._documents)} documents to disk")
        except Exception as e:
            raise VectorStoreError(
                message=f"Failed to persist: {str(e)}",
            )

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """
        Add documents with embeddings to the store.

        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: Optional metadata for each document
            ids: Optional IDs (generated if not provided)

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        if ids is None:
            ids = [str(uuid4()) for _ in documents]

        if metadatas is None:
            metadatas = [{} for _ in documents]

        new_embeddings = np.array(embeddings)

        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        self._documents.extend(documents)
        self._metadatas.extend(metadatas)
        self._ids.extend(ids)

        logger.info(f"Added {len(documents)} documents to collection")
        return ids

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter (not implemented)

        Returns:
            List of matching documents with scores
        """
        if self._embeddings is None or len(self._documents) == 0:
            return []

        query_vec = np.array(query_embedding)
        
        query_norm = query_vec / np.linalg.norm(query_vec)
        doc_norms = self._embeddings / np.linalg.norm(
            self._embeddings, axis=1, keepdims=True
        )
        
        similarities = np.dot(doc_norms, query_norm)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "id": self._ids[idx],
                "content": self._documents[idx],
                "metadata": self._metadatas[idx],
                "score": float(similarities[idx]),
            })

        return results

    def delete_collection(self) -> None:
        """Delete all documents from the collection."""
        self._documents = []
        self._embeddings = None
        self._metadatas = []
        self._ids = []

        storage_path = self._get_storage_path()
        metadata_path = self._get_metadata_path()

        if os.path.exists(storage_path):
            os.remove(storage_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        logger.info(f"Collection '{self.collection_name}' deleted")

    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "name": self.collection_name,
            "document_count": len(self._documents),
            "persist_directory": self.persist_directory,
        }

    @property
    def client(self):
        """Compatibility property."""
        return self

    @property
    def collection(self):
        """Compatibility property."""
        return self

    def count(self) -> int:
        """Return document count."""
        return len(self._documents)
