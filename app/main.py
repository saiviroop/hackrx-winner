# app/main.py
# OPTIMIZED MAIN APP FOR HACKRX CONTEST - FAST & RELIABLE

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
from app.config import get_settings

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create optimized FastAPI app
app = FastAPI(
    title="HackRx Insurance RAG API",
    description="Fast AI-powered insurance document Q&A system optimized for contest",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware (optimized)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log slow requests
    if process_time > 5.0:
        logger.warning(f"Slow request: {request.url.path} took {process_time:.2f}s")
    
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc)[:200],  # Limit error message length
            "path": str(request.url.path)
        }
    )

# Load HackRx routes with error handling
try:
    from app.api.hackrx_routes import router as hackrx_router
    app.include_router(hackrx_router, prefix="/hackrx", tags=["hackrx-contest"])
    logger.info("✅ HackRx routes loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load HackRx routes: {e}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with contest information"""
    return {
        "message": "HackRx 6.0 Insurance RAG API",
        "status": "🚀 OPTIMIZED FOR CONTEST",
        "contest_endpoint": "/hackrx/run",
        "health_check": "/hackrx/health",
        "api_docs": "/docs",
        "version": settings.app_version,
        "optimizations": [
            "Fast document processing",
            "Parallel question handling", 
            "Optimized OpenAI calls",
            "Smart hybrid search",
            "Contest-specific tuning"
        ]
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "service": "hackrx-insurance-rag",
        "version": settings.app_version,
        "timestamp": time.time(),
        "config": {
            "openai_model": settings.openai_model,
            "chunk_size": settings.chunk_size,
            "max_documents": settings.max_documents,
            "api_timeout": settings.api_timeout
        }
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("🚀 Starting HackRx Insurance RAG API...")
    logger.info(f"OpenAI Model: {settings.openai_model}")
    logger.info(f"Chunk Size: {settings.chunk_size}")
    logger.info(f"Max Documents: {settings.max_documents}")
    logger.info("✅ Application startup complete!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("🛑 Shutting down HackRx Insurance RAG API...")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for production
        access_log=True
    )