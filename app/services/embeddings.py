from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
from functools import lru_cache
from app.config import get_settings

import asyncio

settings = get_settings()


class EmbeddingService:
    """Service for generating embeddings using Sentence Transformers."""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        if self.model is None:
            self.model = SentenceTransformer(settings.embedding_model)
    
    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for one or more texts.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings - this is CPU bound
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query (Async).
        """
        # Offload CPU bound task to thread pool
        embedding = await asyncio.to_thread(self.generate_embeddings, query)
        return embedding[0].tolist()
    
    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents (Async).
        """
        # Offload CPU bound task to thread pool
        embeddings = await asyncio.to_thread(self.generate_embeddings, documents)
        return embeddings.tolist()


# Global embedding service instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
