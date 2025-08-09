from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import torch
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    @lru_cache(maxsize=1000)
    def _cached_encode(self, text: str) -> np.ndarray:
        """Cache individual embeddings"""
        return self.model.encode(text, convert_to_numpy=True)
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for text(s)"""
        if isinstance(texts, str):
            return self._cached_encode(texts)
        
        # Batch processing for multiple texts
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query with query-specific preprocessing"""
        # Add query markers for better retrieval
        query_text = f"Query: {query}"
        return self.encode(query_text)