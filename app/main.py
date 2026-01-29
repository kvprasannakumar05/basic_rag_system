from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from datetime import datetime
import time

from app.api import upload, query, documents
from app.models import HealthResponse
from app.config import get_settings

settings = get_settings()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create FastAPI app
app = FastAPI(
    title="RAG Question Answering System",
    description="A production-ready RAG system with document upload and intelligent Q&A",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: 
    {"error": "Rate limit exceeded. Please try again later."}
)
app.add_middleware(SlowAPIMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with rate limiting
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(query.router, prefix="/api", tags=["Query"])
app.include_router(documents.router, prefix="/api", tags=["Documents"])


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
                <body>
                    <h1>RAG QA System API</h1>
                    <p>API is running! Visit <a href="/api/docs">/api/docs</a> for documentation.</p>
                </body>
            </html>
            """
        )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        from app.services.vector_store import get_vector_store
        
        vector_store = get_vector_store()
        stats = vector_store.get_stats()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            services={
                "vector_store": "connected",
                "total_vectors": stats.get("total_vectors", 0),
                "embedding_model": settings.embedding_model,
                "llm_model": settings.llm_model
            }
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            services={"error": str(e)}
        )


# Mount static files (for CSS, JS)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass  # Skip if static directory doesn't exist


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
