# app/api/hackrx_routes.py

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import httpx
import tempfile
import os
import logging

from app.core.document_processor import DocumentProcessor
from app.core.llm_handler import LLMHandler
from app.core.vector_store import VectorStore
from app.core.hybrid_search import HybridSearch

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()

# Contest-specific models
class HackRxRequest(BaseModel):
    documents: str  # URL to PDF document
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# Expected bearer token from contest
EXPECTED_TOKEN = "6f1f341508f756f9e85ac3beeccbe53ab1808a2a650b81c04abeaa80f81356d7"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the bearer token"""
    if credentials.credentials != EXPECTED_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

@router.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(
    request: HackRxRequest,
    token: str = Depends(verify_token)
):
    """
    HackRx contest endpoint - processes documents and answers questions
    """
    try:
        logger.info(f"Processing HackRx request with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents[:100]}...")
        
        # Download the document from the provided URL
        async with httpx.AsyncClient(timeout=60.0) as client:
            logger.info("Downloading document...")
            doc_response = await client.get(request.documents)
            doc_response.raise_for_status()
            logger.info(f"Downloaded document, size: {len(doc_response.content)} bytes")
            
        # Save the document temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(doc_response.content)
            temp_doc_path = temp_file.name
        
        try:
            # Process the document
            logger.info("Processing document...")
            doc_processor = DocumentProcessor()
            chunks = doc_processor.process_document(temp_doc_path)
            logger.info(f"Processed document into {len(chunks)} chunks")
            
            # Create a temporary vector store for this document
            logger.info("Creating vector store...")
            vector_store = VectorStore()
            vector_store.add_documents(chunks)
            
            # Initialize hybrid search
            hybrid_search = HybridSearch(vector_store)
            
            # Initialize LLM handler
            llm_handler = LLMHandler()
            
            # Process each question
            answers = []
            for i, question in enumerate(request.questions):
                try:
                    logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:100]}...")
                    
                    # Get relevant context using hybrid search
                    relevant_docs = hybrid_search.search(question, k=5)
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    
                    # Log context length for debugging
                    logger.info(f"Retrieved context length: {len(context)} characters")
                    
                    # Generate answer using LLM
                    answer = await llm_handler.generate_response(question, context)
                    answers.append(answer)
                    logger.info(f"Generated answer {i+1}: {answer[:100]}...")
                    
                except Exception as e:
                    logger.error(f"Error processing question {i+1} '{question[:50]}...': {str(e)}")
                    answers.append("Unable to process this question due to a technical error.")
            
            logger.info(f"Successfully processed all {len(answers)} questions")
            return HackRxResponse(answers=answers)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_doc_path):
                os.unlink(temp_doc_path)
                logger.info("Cleaned up temporary file")
                
    except httpx.HTTPError as e:
        logger.error(f"Error downloading document: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    except Exception as e:
        logger.error(f"Error in hackrx_run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/hackrx/health")
async def health_check():
    """Health check endpoint for contest"""
    return {
        "status": "healthy", 
        "message": "HackRx Insurance RAG API is running",
        "endpoint": "/hackrx/run",
        "version": "1.0.0"
    }