from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
import time
import logging
from app.models import QueryRequest, QueryResponse, HealthCheck, SearchResult
from app.core.document_processor import DocumentProcessor
from app.core.embeddings import EmbeddingGenerator
from app.core.vector_store import VectorStore
from app.core.hybrid_search import HybridSearch
from app.core.reranker import Reranker
from app.core.llm_handler import LLMHandler
from app.core.cache_manager import CacheManager
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()

# Initialize components
doc_processor = DocumentProcessor(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)
embedding_gen = EmbeddingGenerator(settings.embedding_model)
vector_store = VectorStore(
    embedding_dim=embedding_gen.embedding_dim,
    index_path=settings.vector_db_path
)
hybrid_search = HybridSearch(vector_store, embedding_gen)
reranker = Reranker()
llm_handler = LLMHandler()
cache_manager = CacheManager()

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Main query endpoint"""
    start_time = time.time()
    
    # Check cache
    cached_response = cache_manager.get(request.query)
    if cached_response:
        return QueryResponse(**cached_response)
    
    try:
        # Search
        search_results = hybrid_search.search(
            query=request.query,
            k=settings.top_k_retrieval
        )
        
        # Rerank if enabled
        if request.enable_rerank and search_results:
            search_results = reranker.rerank(
                query=request.query,
                documents=search_results,
                top_k=settings.top_k_rerank
            )
        
        # Generate answer
        answer_text = llm_handler.generate_answer(
            query=request.query,
            context=search_results[:request.context_window],
            stream=request.stream
        )
        
        # Calculate confidence
        confidence = _calculate_confidence(search_results)
        
        # Prepare response
        response = QueryResponse(
            query=request.query,
            answer={
                "answer": answer_text,
                "confidence": confidence,
                "sources": search_results[:request.context_window],
                "metadata": {
                    "query_type": request.query_type,
                    "total_results": len(search_results)
                }
            },
            response_time_ms=(time.time() - start_time) * 1000
        )
        
        # Cache response
        cache_manager.set(request.query, response.dict())
        
        return response
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        version="1.0.0",
        uptime_seconds=time.time(),
        total_documents=len(vector_store.documents),
        cache_status="enabled" if cache_manager.enabled else "disabled"
    )

@router.post("/index/rebuild")
async def rebuild_index(background_tasks: BackgroundTasks):
    """Rebuild vector index"""
    background_tasks.add_task(_rebuild_index)
    return {"message": "Index rebuild started"}

def _calculate_confidence(results: List) -> float:
    """Calculate confidence score"""
    if not results:
        return 0.0
    
    # Average of top 3 scores
    top_scores = [r.get('score', 0) for r in results[:3]]
    return sum(top_scores) / len(top_scores) if top_scores else 0.0

async def _rebuild_index():
    """Background task to rebuild index"""
    # Implementation for rebuilding index
    pass