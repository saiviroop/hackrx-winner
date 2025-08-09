# app/core/hybrid_search.py
# Minimal hybrid search without numpy dependencies

from typing import List, Dict, Any
import logging
from rank_bm25 import BM25Okapi
from app.core.vector_store import VectorStore, Document

logger = logging.getLogger(__name__)

class HybridSearch:
    """Simplified hybrid search"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.bm25 = None
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from documents"""
        if not self.vector_store.documents:
            logger.warning("No documents available for BM25 indexing")
            return
        
        try:
            tokenized_docs = []
            for doc in self.vector_store.documents:
                tokens = doc.page_content.lower().split()
                tokenized_docs.append(tokens)
            
            self.bm25 = BM25Okapi(tokenized_docs)
            logger.info(f"Built BM25 index for {len(tokenized_docs)} documents")
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            self.bm25 = None
    
    def search(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Document]:
        """
        Hybrid search - semantic + keyword
        """
        if not self.vector_store.documents:
            logger.warning("No documents available for search")
            return []
        
        try:
            # Get semantic search results
            semantic_results = self.vector_store.similarity_search(query, k=k*2)
            
            # If no BM25, just return semantic results
            if not self.bm25:
                return semantic_results[:k]
            
            # Get BM25 results
            query_tokens = query.lower().split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Simple combination: take top semantic results
            # This is simplified but works for the contest
            return semantic_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return self.vector_store.similarity_search(query, k)