from fastembed import TextEmbedding
import numpy as np
from typing import List, Union
from app.config import get_settings
import asyncio

settings = get_settings()

class EmbeddingService:
    """Service for generating embeddings using FastEmbed (ONNX/Serverless optimized)."""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        if self.model is None:
            # fastembed automatically downloads/caches the model efficiently
            self.model = TextEmbedding(model_name=settings.embedding_model)
    
    def _generate_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding generation using FastEmbed."""
        # fastembed.embed(documents) returns a generator of numpy arrays
        embeddings_generator = self.model.embed(texts)
        return [embedding.tolist() for embedding in embeddings_generator]
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query (Async)."""
        embeddings = await asyncio.to_thread(self._generate_sync, [query])
        return embeddings[0]
    
    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents (Async)."""
        return await asyncio.to_thread(self._generate_sync, documents)


# Global embedding service instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
