from pydantic_settings import BaseSettings
from typing import Optional
import os
from functools import lru_cache

class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"
    
    # Fallback LLM
    use_local_llm: bool = False
    local_model_path: Optional[str] = None
    
    # Vector Store
    vector_db_path: str = "./data/vector_store"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Search
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k_retrieval: int = 10
    top_k_rerank: int = 3
    
    # Performance
    enable_cache: bool = True
    cache_ttl: int = 3600
    max_concurrent_requests: int = 10
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()