from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from app.config import HOST, PORT
from app.rag_engine import SpectralRAGEngine

# Global Engine Instance
rag_engine: SpectralRAGEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_engine
    # Startup: Load or build vector database
    rag_engine = SpectralRAGEngine()
    yield
    # Shutdown: Clean up if needed
    rag_engine = None

app = FastAPI(
    title="Ghost in the Campus API",
    description="Spectral backend responding to campus FAQs with eerie wit and factual accuracy.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for any client (React, web, mobile, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500, example="Where is the IoT lab and what are the hours?")

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

class StatusResponse(BaseModel):
    status: str
    detail: str

# Endpoints
@app.get("/", tags=["Health"])
async def root():
    return {
        "status": "active",
        "message": "The Campus Spectre hovers silently. Ready for mortal questions."
    }

@app.post("/ask", response_model=QueryResponse, tags=["Chat"])
async def ask_spectre(payload: QueryRequest):
    if not rag_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The ethereal plane is not yet initialized."
        )
    try:
        answer, sources = rag_engine.query(payload.question)
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An ethereal disturbance occurred: {str(e)}"
        )

@app.post("/reindex", response_model=StatusResponse, tags=["Admin"])
async def reindex_knowledge_base():
    """Call this endpoint whenever you update or add text files in campus_docs/."""
    if not rag_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized."
        )
    try:
        total_vectors = rag_engine.reindex()
        return StatusResponse(
            status="success",
            detail=f"Campus archives absorbed. Total memory vectors: {total_vectors}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reindexing failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)