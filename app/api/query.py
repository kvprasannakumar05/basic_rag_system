from fastapi import APIRouter, HTTPException
import time
import logging
from app.models import QueryRequest, QueryResponse, ChunkSource, QueryMetadata
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.llm import get_llm_service
from app.services.memory import get_memory_service
from app.config import get_settings

settings = get_settings()

import asyncio

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the knowledge base with a question.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {request.question}")
        
        # 0. Retrieve Chat History
        memory_service = get_memory_service()
        chat_history = memory_service.get_history(request.session_id)
        
        # 1. Query Routing / Intent Classification
        llm_service = get_llm_service()
        intent = await asyncio.to_thread(llm_service.classify_query, request.question, chat_history)
        logger.info(f"Query Intent: {intent}")
        
        matches = []
        retrieval_time_ms = 0
        context_chunks = []
        
        # 2. Branch: GENERAL (Skip Retrieval)
        if intent == "GENERAL":
            logger.info("Skipping retrieval for General query")
            
        # 3. Branch: RAG (Perform Retrieval)
        else:
            # Embed the query
            embedding_service = get_embedding_service()
            query_embedding = await embedding_service.embed_query(request.question)
            
            retrieval_start = time.time()
            
            # Retrieve similar chunks
            vector_store = get_vector_store()
            matches = await vector_store.search_similar(
                query_embedding=query_embedding,
                namespace=request.session_id,
                top_k=settings.top_k,
                score_threshold=settings.score_threshold
            )
            
            retrieval_time_ms = (time.time() - retrieval_start) * 1000
            
            # Detailed Logging for Retrieval Analysis
            if matches:
                top_score = matches[0]["score"]
                logger.info(f"Retrieved {len(matches)} chunks. Top Score: {top_score:.4f}")
            else:
                logger.warning(f"No chunks retrieved (Intent: RAG, Threshold: {settings.score_threshold})")

            # Context chunks
            context_chunks = [
                {
                    "text": match["text"],
                    "filename": match["filename"],
                    "document_id": match["document_id"]
                }
                for match in matches
            ]
        
        # 4. Generate Answer (Unified)
        generation_start = time.time()
        answer = await asyncio.to_thread(llm_service.generate_answer, request.question, context_chunks, chat_history)
        generation_time_ms = (time.time() - generation_start) * 1000
        
        # 5. Update Memory (User then Assistant)
        memory_service.add_message(request.session_id, "user", request.question)
        memory_service.add_message(request.session_id, "assistant", answer)
        
        logger.info(f"Answer generated in {generation_time_ms:.2f}ms")
        
        # Prepare sources
        sources = [
            ChunkSource(
                chunk_text=match["text"][:500] + "..." if len(match["text"]) > 500 else match["text"],
                document_id=match["document_id"],
                similarity_score=round(match["score"], 4),
                metadata={
                    "filename": match["filename"],
                    "chunk_index": match["chunk_index"]
                }
            )
            for match in matches
        ]
        
        # Calculate total time
        total_time_ms = (time.time() - start_time) * 1000
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            metadata=QueryMetadata(
                retrieval_time_ms=round(retrieval_time_ms, 2),
                generation_time_ms=round(generation_time_ms, 2),
                total_time_ms=round(total_time_ms, 2),
                chunks_retrieved=len(matches)
            )
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
