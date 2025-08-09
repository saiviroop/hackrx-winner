# app/core/embeddings.py
# OpenAI-only embeddings - no sentence_transformers

import logging
from typing import List
import openai
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class EmbeddingGenerator:
    """Generate embeddings using OpenAI API"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = "text-embedding-3-small"  # Cheaper, faster model
        
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