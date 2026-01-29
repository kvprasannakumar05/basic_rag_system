from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from fastapi.responses import JSONResponse
import time
from app.models import DocumentUploadResponse
from app.services.document_processor import process_document
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store

router = APIRouter()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    x_session_id: str = Header(..., alias="x-session-id")
):
    """
    Upload and process a document (PDF or TXT).
    
    The document will be:
    1. Validated
    2. Parsed and chunked
    3. Embedded
    4. Stored in the vector database with namespace isolation
    """
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        filename_lower = file.filename.lower()
        if not (filename_lower.endswith('.pdf') or filename_lower.endswith('.txt')):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Only PDF and TXT files are supported. Received: {file.filename}"
            )
        
        # Process document (extract text and chunk)
        chunks, document_id = await process_document(file)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content could be extracted from the document")
        
        # Generate embeddings
        embedding_service = get_embedding_service()
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = await embedding_service.embed_documents(chunk_texts)
        
        # Prepare chunks for vector store
        chunks_dict = [chunk.to_dict() for chunk in chunks]
        
        # Store in vector database with namespace
        vector_store = get_vector_store()
        upsert_result = await vector_store.upsert_chunks(
            chunks=chunks_dict,
            embeddings=embeddings,
            namespace=x_session_id
        )
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        return DocumentUploadResponse(
            status="success",
            document_id=document_id,
            filename=file.filename,
            chunks_processed=len(chunks),
            processing_time_ms=round(processing_time_ms, 2)
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
