# app/core/embeddings.py
# FINAL VERSION - accepts model parameter

import logging
from typing import List
import openai
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class EmbeddingGenerator:
    """Generate embeddings using OpenAI API"""
    
    def __init__(self, model_name: str = None):
        """Initialize with optional model name (ignored, we use OpenAI)"""
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = "text-embedding-3-small"  # Always use OpenAI model
        logger.info(f"EmbeddingGenerator initialized (using OpenAI {self.model})")
        
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = []
        
        for text in texts:
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text[:8000]  # Limit length
                )
                embeddings.append(response.data[0].embedding)
                
                # Small delay to avoid rate limits
                import time
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error generating embedding: {e}")
                embeddings.append([0.0] * 1536)  # Zero vector fallback
        
        return embeddings
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate single embedding"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text[:8000]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 1536