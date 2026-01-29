from langchain_huggingface import HuggingFaceEndpointEmbeddings
from typing import List
from app.config import get_settings
import asyncio

settings = get_settings()

class EmbeddingService:
    """Service for generating embeddings using HuggingFace Inference API (Serverless)."""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the HuggingFace endpoint client."""
        if self.client is None:
            # This uses the hosted API instead of local RAM/CPU
            self.client = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
                task="feature-extraction",
                huggingfacehub_api_token=settings.huggingfacehub_api_token
            )
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query (Async)."""
        # LangChain's embed_query is synchronous, so we offload to thread
        return await asyncio.to_thread(self.client.embed_query, query)
    
    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents (Async)."""
        # LangChain's embed_documents is synchronous, so we offload to thread
        return await asyncio.to_thread(self.client.embed_documents, documents)



# Global embedding service instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
