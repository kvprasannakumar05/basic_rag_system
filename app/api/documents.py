from fastapi import APIRouter, HTTPException, Header
from typing import List, Dict
from app.services.vector_store import get_vector_store

router = APIRouter()


@router.get("/documents")
async def list_documents(x_session_id: str = Header(..., alias="x-session-id")):
    """
    List all uploaded documents with metadata.
    
    Returns a list of unique documents with their metadata.
    """
    try:
        vector_store = get_vector_store()
        
        # Query all vectors to get unique documents
        # Pinecone doesn't have a native "list all documents" so we fetch metadata
        stats = vector_store.get_stats()
        
        # Fetch sample vectors to extract document info
        # This is a limitation - Pinecone doesn't easily list all unique documents
        # For production, you'd maintain a separate documents table
        results = vector_store.index.query(
            vector=[0.0] * 384,  # Dummy vector
            namespace=x_session_id,
            top_k=10000,  # Get many results
            include_metadata=True
        )
        
        # Extract unique documents
        documents_dict = {}
        for match in results.matches:
            doc_id = match.metadata.get('document_id')
            if doc_id and doc_id not in documents_dict:
                documents_dict[doc_id] = {
                    'document_id': doc_id,
                    'filename': match.metadata.get('filename', 'Unknown'),
                    'file_type': match.metadata.get('file_type', 'unknown'),
                    'upload_timestamp': match.metadata.get('upload_timestamp', ''),
                    'total_chunks': match.metadata.get('total_chunks', 0)
                }
        
        documents = list(documents_dict.values())
        
        return {
            'total_documents': len(documents),
            'documents': documents
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching documents: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    x_session_id: str = Header(..., alias="x-session-id")
):
    """
    Delete a document and all its chunks from the vector store.
    
    Args:
        document_id: The ID of the document to delete
    """
    try:
        vector_store = get_vector_store()
        
        # Query to find all chunk IDs for this document
        # Use metadata filter to find chunks belonging to this document
        results = vector_store.index.query(
            vector=[0.0] * 384,  # Dummy vector
            namespace=x_session_id,
            top_k=10000,
            filter={"document_id": {"$eq": document_id}},
            include_metadata=True
        )
        
        if not results.matches:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Extract chunk IDs
        chunk_ids = [match.id for match in results.matches]
        
        # Delete all chunks
        vector_store.delete_vectors(ids=chunk_ids, namespace=x_session_id)
        
        return {
            'status': 'success',
            'document_id': document_id,
            'chunks_deleted': len(chunk_ids),
            'message': f'Successfully deleted document and {len(chunk_ids)} chunks'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@router.delete("/documents")
async def delete_all_documents(x_session_id: str = Header(..., alias="x-session-id")):
    """
    Delete ALL documents from the vector store for the current session.
    """
    try:
        vector_store = get_vector_store()
        
        # Delete all vectors in the namespace
        vector_store.delete_all(namespace=x_session_id)
        
        return {
            'status': 'success',
            'message': 'All documents deleted successfully'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting all documents: {str(e)}")
