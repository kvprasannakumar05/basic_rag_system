from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    status: str
    document_id: str
    filename: str
    chunks_processed: int
    processing_time_ms: float


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., min_length=1, max_length=1000, description="User's question")
    session_id: str = Field(..., min_length=1, description="User session ID for isolation")


class ChunkSource(BaseModel):
    """Model for a source chunk in the response."""
    chunk_text: str
    document_id: str
    similarity_score: float
    metadata: dict


class QueryMetadata(BaseModel):
    """Metadata about query processing."""
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    chunks_retrieved: int


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    sources: List[ChunkSource]
    metadata: QueryMetadata


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    services: dict
