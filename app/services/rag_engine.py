"""
RAG (Retrieval-Augmented Generation) engine.

Orchestrates the complete RAG pipeline:
document ingestion, retrieval, and generation.
"""

import time
from typing import Optional

from app.core.config import get_settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.text_processor import TextProcessor
from app.services.vector_store import VectorStore

logger = get_logger(__name__)


RAG_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's question based on the provided context.

Instructions:
- Use ONLY the information from the context to answer
- If the context doesn't contain relevant information, say so
- Be concise and direct in your response
- Do not make up information

Context:
{context}
"""


class RAGEngine:
    """
    Complete RAG pipeline orchestrator.

    Handles document processing, embedding, storage,
    retrieval, and response generation.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        llm_service: Optional[LLMService] = None,
        text_processor: Optional[TextProcessor] = None,
        collection_name: Optional[str] = None,
    ):
        self.settings = get_settings()
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore(
            collection_name=collection_name
        )
        self.llm_service = llm_service or LLMService()
        self.text_processor = text_processor or TextProcessor()

    async def ingest_documents(
        self,
        documents: list[dict],
    ) -> dict:
        """
        Ingest documents into the vector store.

        Args:
            documents: List of documents with content and metadata

        Returns:
            Ingestion statistics
        """
        start_time = time.time()

        chunks = list(self.text_processor.process_documents(documents))

        if not chunks:
            return {
                "ingested_count": 0,
                "chunk_count": 0,
                "processing_time_ms": 0,
            }

        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        logger.info(f"Generating embeddings for {len(texts)} chunks")
        embeddings = self.embedding_service.embed_texts(texts)

        self.vector_store.add_documents(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        self.vector_store.persist()

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Ingested {len(documents)} documents "
            f"({len(chunks)} chunks) in {processing_time:.2f}ms"
        )

        return {
            "ingested_count": len(documents),
            "chunk_count": len(chunks),
            "processing_time_ms": processing_time,
        }

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of relevant documents with scores
        """
        top_k = top_k or self.settings.retrieval_top_k

        try:
            query_embedding = self.embedding_service.embed_text(query)

            results = self.vector_store.search(
                query_embedding=query_embedding.tolist(),
                top_k=top_k,
            )

            logger.info(f"Retrieved {len(results)} documents for query")
            return results

        except Exception as e:
            raise RetrievalError(
                message=f"Retrieval failed: {str(e)}",
            )

    def _build_context(self, documents: list[dict]) -> str:
        """Build context string from retrieved documents."""
        if not documents:
            return "No relevant context found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[{i}] {doc['content']}")

        context = "\n\n".join(context_parts)

        if len(context) > self.settings.max_context_length:
            context = context[: self.settings.max_context_length] + "..."

        return context

    async def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> dict:
        """
        Complete RAG query: retrieve and generate.

        Args:
            question: User question
            top_k: Number of documents to retrieve

        Returns:
            Response with answer and sources
        """
        start_time = time.time()

        retrieved_docs = await self.retrieve(question, top_k)

        context = self._build_context(retrieved_docs)
        system_prompt = RAG_SYSTEM_PROMPT.format(context=context)

        response = await self.llm_service.generate(
            prompt=question,
            system_prompt=system_prompt,
        )

        processing_time = (time.time() - start_time) * 1000

        return {
            "answer": response,
            "sources": retrieved_docs,
            "processing_time_ms": processing_time,
        }

    async def query_without_rag(self, question: str) -> dict:
        """
        Direct LLM query without retrieval.

        Args:
            question: User question

        Returns:
            Response from LLM
        """
        start_time = time.time()

        response = await self.llm_service.generate(prompt=question)

        processing_time = (time.time() - start_time) * 1000

        return {
            "answer": response,
            "sources": [],
            "processing_time_ms": processing_time,
        }

    def get_stats(self) -> dict:
        """Get RAG engine statistics."""
        return {
            "vector_store": self.vector_store.get_stats(),
            "embedding_model": self.embedding_service.model_name,
            "llm_model": self.llm_service.model,
        }
