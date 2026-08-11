import os
import sys
import logging
import time
from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import uvicorn

# --- NEW: Import Rate Limiting modules ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- 1. Configure Production Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ask_my_docs_api")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ui.app import initialize_backend

# --- NEW: Initialize the Rate Limiter ---
# get_remote_address uses the user's IP address to track their request count
limiter = Limiter(key_func=get_remote_address)

# --- 2. Initialize the FastAPI application ---
app = FastAPI(
    title="Ask My Docs API",
    description="A production RAG API for Environmental Policy Q&A",
    version="1.0.0"
)

# --- NEW: Register the Rate Limiter with FastAPI ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore

# --- 3. Security: CORS Configuration ---
origins = [
    "http://localhost:3000",
    "http://localhost:8501", 
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Security: API Key Authentication ---
VALID_API_KEY = "ak-ask-my-docs-prod-777"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != VALID_API_KEY:
        logger.warning(f"Unauthorized access attempt rejected.")
        raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing API Key")
    return api_key_header

# --- 5. Middleware for Request Timing ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} | Method: {request.method} | Status: {response.status_code} | Latency: {process_time:.4f}s")
    return response

# --- 6. Define our Data Models (Pydantic) ---
class QueryRequest(BaseModel):
    question: str

class SourceMetadata(BaseModel):
    page: str | int

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceMetadata]

# --- 7. Load the RAG Pipeline ---
logger.info("Starting API Server and loading RAG backend...")
try:
    rag_chain = initialize_backend()
    logger.info("RAG backend successfully initialized.")
except Exception as e:
    logger.critical(f"CRITICAL ERROR: Failed to load backend. {e}")
    sys.exit(1)

# --- 8. Define the API Endpoint ---
@app.post("/api/v1/ask", response_model=QueryResponse)
@limiter.limit("2/minute") # <-- NEW: The user can only ask 2 questions per minute!
async def ask_question(request: Request, query: QueryRequest, api_key: str = Depends(get_api_key)):
    logger.info(f"Received query: '{query.question}' from authorized client.")
    try:
        response = rag_chain.invoke({"input": query.question})
        
        cited_sources = [SourceMetadata(page=doc.metadata.get("page", "Unknown")) for doc in response["context"]]
            
        logger.info(f"Successfully generated answer using {len(cited_sources)} sources.")
        return QueryResponse(
            answer=response["answer"],
            sources=cited_sources
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)