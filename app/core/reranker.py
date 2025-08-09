# app/core/reranker.py
# Disabled reranker to avoid sentence_transformers dependency

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Reranker:
    """Dummy reranker that just returns documents as-is"""
    
    def __init__(self, model_name: str = None):
        logger.info("Reranker initialized (disabled for deployment)")
        
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Just return the documents without reranking"""
        return documents[:top_k]