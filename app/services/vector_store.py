from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Optional
from app.config import get_settings
from app.services.embeddings import get_embedding_service
import time

import asyncio

settings = get_settings()


class VectorStore:
    """Service for managing Pinecone vector store operations."""
    
    def __init__(self):
        self.pc = None
        self.index = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Pinecone client and index."""
        # Initialize Pinecone
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        
        # Check if index exists, create if not
        index_name = settings.pinecone_index_name
        
        if index_name not in self.pc.list_indexes().names():
            # Create index with dimension matching embedding model (384 for all-MiniLM-L6-v2)
            self.pc.create_index(
                name=index_name,
                dimension=384,  # Dimension for all-MiniLM-L6-v2
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            # Wait for index to be ready
            time.sleep(5)
        
        # Connect to index
        self.index = self.pc.Index(index_name)
    
    async def upsert_chunks(self, chunks: List[Dict], embeddings: List[List[float]], namespace: str) -> Dict:
        """
        Upload chunks with their embeddings to Pinecone (Async).
        """
        return await asyncio.to_thread(self._upsert_sync, chunks, embeddings, namespace)

    def _upsert_sync(self, chunks: List[Dict], embeddings: List[List[float]], namespace: str) -> Dict:
        """Synchronous version of upsert for threading."""
        vectors = []
        
        for chunk, embedding in zip(chunks, embeddings):
            vector = {
                "id": chunk["chunk_id"],
                "values": embedding,
                "metadata": {
                    "text": chunk["text"][:1000],  # Pinecone metadata limit
                    "document_id": chunk["document_id"],
                    "filename": chunk["metadata"]["filename"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "total_chunks": chunk["metadata"]["total_chunks"],
                    "file_type": chunk["metadata"]["file_type"],
                    "upload_timestamp": chunk["metadata"]["upload_timestamp"]
                }
            }
            vectors.append(vector)
        
        # Upsert in batches
        batch_size = 100
        upsert_count = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            result = self.index.upsert(vectors=batch, namespace=namespace)
            upsert_count += result.upserted_count
        
        return {
            "upserted_count": upsert_count,
            "total_vectors": len(vectors)
        }
    
    async def search_similar(
        self,
        query_embedding: List[float],
        namespace: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar chunks - Async.
        """
        return await asyncio.to_thread(
            self._search_sync, 
            query_embedding, 
            namespace, 
            top_k, 
            score_threshold, 
            filter_dict
        )

    def _search_sync(self, query_embedding, namespace, top_k, score_threshold, filter_dict):
        """Synchronous search logic."""
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            namespace=namespace,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Filter by score threshold and format results
        matches = []
        for match in results.matches:
            if match.score >= score_threshold:
                matches.append({
                    "chunk_id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "document_id": match.metadata.get("document_id", ""),
                    "filename": match.metadata.get("filename", ""),
                    "chunk_index": match.metadata.get("chunk_index", 0),
                    "metadata": match.metadata
                })
        
        return matches
    
    def delete_vectors(self, ids: List[str], namespace: str):
        """
        Delete vectors by ID from a specific namespace.
        """
        self.index.delete(ids=ids, namespace=namespace)
        
    def delete_all(self, namespace: str):
        """
        Delete all vectors in a specific namespace.
        """
        self.index.delete(delete_all=True, namespace=namespace)

    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": stats.namespaces,
            "index_name": settings.pinecone_index_name
        }


# Global vector store instance
_vector_store = None


def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
