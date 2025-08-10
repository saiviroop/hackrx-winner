# app/config.py
# OPTIMIZED CONFIG FOR HACKRX CONTEST

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Optimized settings for HackRx contest performance"""
    
    # OpenAI Configuration (REQUIRED)
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"  # Faster than GPT-4
    openai_max_tokens: int = 200  # Shorter for speed
    openai_temperature: float = 0.1  # Low for consistency
    
    # Server Configuration
    app_name: str = "HackRx Insurance RAG API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Processing Configuration (OPTIMIZED FOR SPEED)
    chunk_size: int = 800  # Smaller for faster processing
    chunk_overlap: int = 100
    max_documents: int = 20  # Limit for speed
    max_search_results: int = 3  # Fewer results for speed
    
    # API Configuration
    api_timeout: int = 30  # Total timeout for requests
    openai_timeout: int = 8  # Fast OpenAI timeout
    
    # Redis Configuration (Optional)
    redis_url: str = "redis://localhost:6379"
    redis_enabled: bool = False  # Disable for simplicity
    
    # Vector Store Configuration
    embedding_dim: int = 1536  # OpenAI ada-002 dimensions
    similarity_threshold: float = 0.7
    
    # Contest Specific
    contest_token: str = "6f1f341508f756f9e85ac3beeccbe53ab1808a2a650b81c04abeaa80f81356d7"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """Get settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings