from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from app.core.vector_store import VectorStore
from app.core.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

class HybridSearch:
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.bm25 = None
        self.corpus = []
        
    def build_bm25_index(self, documents: List[Dict[str, Any]]):
        """Build BM25 index for sparse retrieval"""
        self.corpus = [doc['content'].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)
        logger.info("BM25 index built successfully")
    
    def search(self, query: str, k: int = 10, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Hybrid search combining dense and sparse retrieval
        alpha: weight for dense search (1-alpha for sparse)
        """
        # Dense search
        query_embedding = self.embedding_generator.encode_query(query)
        dense_results = self.vector_store.search(query_embedding, k=k*2)
        
        # Sparse search
        sparse_scores = []
        if self.bm25:
            tokenized_query = query.lower().split()
            sparse_scores = self.bm25.get_scores(tokenized_query)
        
        # Combine scores
        combined_results = self._combine_scores(
            dense_results, 
            sparse_scores, 
            alpha=alpha,
            k=k
        )
        
        return combined_results
    
    def _combine_scores(self, dense_results: List[Tuple[Dict, float]], 
                       sparse_scores: np.ndarray, 
                       alpha: float, 
                       k: int) -> List[Dict[str, Any]]:
        """Combine dense and sparse scores with normalization"""
        score_dict = {}
        
        # Add dense scores
        for doc, score in dense_results:
            doc_idx = self.vector_store.id_to_index.get(doc['id'])
            if doc_idx is not None:
                score_dict[doc_idx] = {
                    'doc': doc,
                    'dense_score': score,
                    'sparse_score': 0
                }
        
        # Add sparse scores
        if len(sparse_scores) > 0:
            # Normalize sparse scores
            max_sparse = np.max(sparse_scores) if np.max(sparse_scores) > 0 else 1
            normalized_sparse = sparse_scores / max_sparse
            
            for idx, score in enumerate(normalized_sparse):
                if idx in score_dict:
                    score_dict[idx]['sparse_score'] = score
                elif idx < len(self.vector_store.documents):
                    score_dict[idx] = {
                        'doc': self.vector_store.documents[idx],
                        'dense_score': 0,
                        'sparse_score': score
                    }
        
        # Calculate combined scores
        results = []
        for item in score_dict.values():
            combined_score = (alpha * item['dense_score'] + 
                            (1 - alpha) * item['sparse_score'])
            doc = item['doc'].copy()
            doc['score'] = combined_score
            doc['dense_score'] = item['dense_score']
            doc['sparse_score'] = item['sparse_score']
            results.append(doc)
        
        # Sort by combined score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:k]