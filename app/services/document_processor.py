import PyPDF2
import io
import uuid
from typing import List, Dict, Tuple
from fastapi import UploadFile
from datetime import datetime
from app.config import get_settings

import asyncio

settings = get_settings()


class DocumentChunk:
    """Represents a document chunk with metadata."""
    
    def __init__(self, text: str, chunk_id: str, document_id: str, metadata: Dict):
        self.text = text
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.metadata = metadata
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "metadata": self.metadata
        }


def extract_text_from_pdf_sync(file_content: bytes) -> str:
    """Synchronous PDF extraction."""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            text += f"\n[Page {page_num + 1}]\n{page_text}"
        
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


def extract_text_from_txt_sync(file_content: bytes) -> str:
    """Synchronous TXT extraction."""
    try:
        try:
            text = file_content.decode('utf-8')
        except UnicodeDecodeError:
            text = file_content.decode('latin-1')
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from TXT: {str(e)}")


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split text into chunks with overlap (CPU bound).
    """
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if overlap is None:
        overlap = settings.chunk_overlap
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            boundary_search_start = max(end - 100, start)
            chunk_text_segment = text[boundary_search_start:end]
            
            last_period = chunk_text_segment.rfind('.')
            last_newline = chunk_text_segment.rfind('\n')
            last_space = chunk_text_segment.rfind(' ')
            
            boundary = max(last_period, last_newline, last_space)
            if boundary != -1:
                end = boundary_search_start + boundary + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start <= 0 or (len(chunks) > 0 and start >= len(text)): # Prevent stuck
             break
        if len(chunks) > 10000: # Safety
            break
            
    return chunks


async def process_document(file: UploadFile) -> Tuple[List[DocumentChunk], str]:
    """
    Process document asynchronously.
    """
    # Read file content (Async)
    file_content = await file.read()
    file_size_mb = len(file_content) / (1024 * 1024)
    
    if file_size_mb > settings.max_file_size_mb:
        raise ValueError(f"File size ({file_size_mb:.2f}MB) exceeds limit")
    
    # Process extraction in thread pool
    filename = file.filename.lower()
    
    if filename.endswith('.pdf'):
        text = await asyncio.to_thread(extract_text_from_pdf_sync, file_content)
    elif filename.endswith('.txt'):
        text = await asyncio.to_thread(extract_text_from_txt_sync, file_content)
    else:
        raise ValueError("Unsupported file type")
    
    if not text:
        raise ValueError("No text content found")
    
    # Chunking (CPU bound, offload)
    text_chunks = await asyncio.to_thread(chunk_text, text)
    
    # Metadata assembly
    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    chunks = []
    for idx, text_chunk in enumerate(text_chunks):
        chunk_id = f"{document_id}_chunk_{idx}"
        metadata = {
            "filename": file.filename,
            "chunk_index": idx,
            "total_chunks": len(text_chunks),
            "upload_timestamp": datetime.utcnow().isoformat(),
            "file_type": "pdf" if filename.endswith('.pdf') else "txt"
        }
        chunks.append(DocumentChunk(text_chunk, chunk_id, document_id, metadata))
        
    return chunks, document_id
