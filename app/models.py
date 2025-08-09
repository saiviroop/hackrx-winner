from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class QueryType(str, Enum):
    GENERAL = "general"
    COVERAGE = "coverage"
    CLAIM = "claim"
    ELIGIBILITY = "eligibility"

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    query_type: Optional[QueryType] = QueryType.GENERAL
    context_window: Optional[int] = Field(default=3, ge=1, le=10)
    enable_rerank: bool = True
    stream: bool = False

class Document(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    
class SearchResult(BaseModel):
    documents: List[Document]
    total_results: int
    search_time_ms: float

class Answer(BaseModel):
    answer: str
    confidence: float
    sources: List[Document]
    metadata: Dict[str, Any]
    
class QueryResponse(BaseModel):
    query: str
    answer: Answer
    response_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)
    
class HealthCheck(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    total_documents: int
    cache_status: str