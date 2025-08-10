from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Simple logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create minimal FastAPI app
app = FastAPI(
    title="Insurance RAG API - HackRx 6.0",
    description="AI-powered insurance document query system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to import the HackRx routes only
try:
    from app.api.hackrx_routes import router as hackrx_router
    app.include_router(hackrx_router, tags=["hackrx-contest"])
    logger.info("HackRx routes loaded successfully")
except Exception as e:
    logger.error(f"Could not load HackRx routes: {e}")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Insurance RAG API - HackRx 6.0",
        "status": "running",
        "contest_endpoint": "/hackrx/run",
        "health_check": "/hackrx/health"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "hackrx-insurance-rag"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )