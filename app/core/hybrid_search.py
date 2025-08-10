# app/core/hybrid_search.py
# OPTIMIZED HYBRID SEARCH FOR HACKRX - FAST RETRIEVAL

from typing import List, Dict, Any
import logging
from rank_bm25 import BM25Okapi
from app.core.vector_store import VectorStore, Document

logger = logging.getLogger(__name__)

class HybridSearch:
    """Fast hybrid search combining BM25 and semantic search"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.bm25 = None
        self.documents = []
        logger.info("HybridSearch initialized")
    
    def update_bm25(self, documents: List[Document]):
        """Fast BM25 index update"""
        try:
            self.documents = documents
            
            # Prepare texts for BM25 (simple tokenization)
            corpus = []
            for doc in documents:
                # Simple but effective tokenization
                tokens = doc.page_content.lower().split()
                corpus.append(tokens)
            
            # Build BM25 index
            self.bm25 = BM25Okapi(corpus)
            logger.info(f"BM25 index updated with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"BM25 update error: {e}")
            # Fallback: create empty BM25
            self.bm25 = BM25Okapi([[]])
    
    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        Fast hybrid search optimized for insurance Q&A
        Returns top-k most relevant documents
        """
        try:
            if not self.documents:
                logger.warning("No documents available for search")
                return []
            
            # Step 1: Fast BM25 search
            bm25_scores = []
            if self.bm25:
                query_tokens = query.lower().split()
                bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Step 2: Fast semantic search
            semantic_docs = self.vector_store.similarity_search(query, k=min(k*2, 10))
            
            # Step 3: Fast score combination
            if len(bm25_scores) == len(self.documents) and semantic_docs:
                # Combine scores using simple approach
                combined_results = self._combine_scores_fast(
                    query, bm25_scores, semantic_docs, k
                )
            elif semantic_docs:
                # Fallback to semantic only
                combined_results = semantic_docs[:k]
            elif bm25_scores:
                # Fallback to BM25 only
                top_bm25_indices = sorted(
                    range(len(bm25_scores)), 
                    key=lambda i: bm25_scores[i], 
                    reverse=True
                )[:k]
                combined_results = [self.documents[i] for i in top_bm25_indices]
            else:
                # Last resort: return first k documents
                combined_results = self.documents[:k]
            
            logger.info(f"Search completed: {len(combined_results)} documents returned")
            return combined_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            # Return fallback results
            return self.documents[:k] if self.documents else []
    
    def _combine_scores_fast(self, query: str, bm25_scores: List[float], 
                           semantic_docs: List[Document], k: int) -> List[Document]:
        """Fast score combination for optimal results"""
        try:
            # Create a set of semantic doc contents for quick lookup
            semantic_contents = {doc.page_content: doc for doc in semantic_docs}
            
            # Score each document
            doc_scores = []
            
            for i, doc in enumerate(self.documents):
                bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0.0
                
                # Check if document is in semantic results
                semantic_score = 1.0 if doc.page_content in semantic_contents else 0.0
                
                # Simple but effective combination
                combined_score = (0.4 * bm25_score) + (0.6 * semantic_score)
                
                doc_scores.append((combined_score, doc))
            
            # Sort by combined score and return top k
            doc_scores.sort(key=lambda x: x[0], reverse=True)
            return [doc for score, doc in doc_scores[:k]]
            
        except Exception as e:
            logger.error(f"Score combination error: {e}")
            # Fallback to semantic results
            return semantic_docs[:k]