from sentence_transformers import CrossEncoder
import torch
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)
        logger.info(f"Reranker initialized on {self.device}")
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder"""
        if not documents:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc['content']] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs, batch_size=16)
        
        # Add rerank scores to documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score
        documents.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return documents[:top_k]