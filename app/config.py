from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "rag-qa-system"
    
    # Groq Configuration
    groq_api_key: str
    
    # Application Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = 10
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # RAG Parameters
    top_k: int = 7
    score_threshold: float = 0.35
    
    # Rate Limiting
    upload_rate_limit: str = "10/hour"
    query_rate_limit: str = "60/hour"
    
    # LLM Settings
    llm_model: str = "llama-3.1-8b-instant"
    max_context_chunks: int = 5
    temperature: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
