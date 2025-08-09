from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
from contextlib import asynccontextmanager
from app.api import routes
from app.api.hackrx_routes import router as hackrx_router  # Add this import
from app.config import get_settings
from app.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting Insurance RAG API for HackRx 6.0...")
    # Initialize components here if needed
    yield
    # Shutdown
    logger.info("Shutting down Insurance RAG API...")

# Create FastAPI app
app = FastAPI(
    title="Insurance Document RAG API - HackRx 6.0",
    description="High-performance RAG system for insurance document Q&A - HackRx Contest",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include your existing routers
app.include_router(routes.router, prefix="/api/v1", tags=["queries"])

# Include HackRx contest routes (NO PREFIX - required by contest)
app.include_router(hackrx_router, tags=["hackrx-contest"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Insurance RAG API - HackRx 6.0",
        "version": "1.0.0",
        "docs": "/docs",
        "contest_endpoint": "/hackrx/run",
        "health_check": "/hackrx/health"
    }

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=True
    )