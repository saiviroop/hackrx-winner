# app/core/hybrid_search.py
# FINAL VERSION - Works with updated vector store

from typing import List, Dict, Any
import logging
from rank_bm25 import BM25Okapi
import numpy as np
from app.core.vector_store import VectorStore, Document

logger = logging.getLogger(__name__)

class HybridSearch:
    """Hybrid search combining semantic and keyword search"""
    
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
            # Tokenize documents for BM25
            tokenized_docs = []
            for doc in self.vector_store.documents:
                # Simple tokenization
                tokens = doc.page_content.lower().split()
                tokenized_docs.append(tokens)
            
            self.bm25 = BM25Okapi(tokenized_docs)
            logger.info(f"Built BM25 index for {len(tokenized_docs)} documents")
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            self.bm25 = None
    
    def search(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Document]:
        """
        Hybrid search combining semantic and keyword search
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for semantic search (1-alpha for BM25)
        """
        if not self.vector_store.documents:
            logger.warning("No documents available for search")
            return []
        
        try:
            # Get semantic search results
            semantic_results = self.vector_store.similarity_search(query, k=min(k*2, len(self.vector_store.documents)))
            
            # If BM25 is not available, just return semantic results
            if not self.bm25:
                logger.info("Using semantic search only (BM25 not available)")
                return semantic_results[:k]
            
            # Get BM25 results
            bm25_results = []
            try:
                query_tokens = query.lower().split()
                bm25_scores = self.bm25.get_scores(query_tokens)
                
                # Get top documents by BM25 score
                top_bm25_indices = sorted(
                    range(len(bm25_scores)), 
                    key=lambda i: bm25_scores[i], 
                    reverse=True
                )[:min(k*2, len(self.vector_store.documents))]
                
                for idx in top_bm25_indices:
                    if idx < len(self.vector_store.documents):
                        doc = self.vector_store.documents[idx]
                        doc_copy = Document(
                            page_content=doc.page_content,
                            metadata={**doc.metadata, 'bm25_score': float(bm25_scores[idx])}
                        )
                        bm25_results.append(doc_copy)
            except Exception as e:
                logger.error(f"Error in BM25 search: {e}")
                # Fall back to semantic only
                return semantic_results[:k]
            
            # Combine results using hybrid scoring
            combined_scores = {}
            
            # Add semantic scores
            for i, doc in enumerate(semantic_results):
                doc_text = doc.page_content
                semantic_score = doc.metadata.get('similarity_score', 0.0)
                combined_scores[doc_text] = {
                    'document': doc,
                    'semantic_score': semantic_score,
                    'bm25_score': 0.0,
                    'hybrid_score': alpha * semantic_score
                }
            
            # Add BM25 scores
            max_bm25_score = max([doc.metadata.get('bm25_score', 0.0) for doc in bm25_results]) if bm25_results else 1.0
            if max_bm25_score == 0:
                max_bm25_score = 1.0
                
            for doc in bm25_results:
                doc_text = doc.page_content
                bm25_score = doc.metadata.get('bm25_score', 0.0) / max_bm25_score  # Normalize
                
                if doc_text in combined_scores:
                    combined_scores[doc_text]['bm25_score'] = bm25_score
                    combined_scores[doc_text]['hybrid_score'] = (
                        alpha * combined_scores[doc_text]['semantic_score'] + 
                        (1 - alpha) * bm25_score
                    )
                else:
                    combined_scores[doc_text] = {
                        'document': doc,
                        'semantic_score': 0.0,
                        'bm25_score': bm25_score,
                        'hybrid_score': (1 - alpha) * bm25_score
                    }
            
            # Sort by hybrid score and return top k
            sorted_results = sorted(
                combined_scores.values(),
                key=lambda x: x['hybrid_score'],
                reverse=True
            )
            
            final_results = []
            for result in sorted_results[:k]:
                doc = result['document']
                # Add all scores to metadata
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        'semantic_score': result['semantic_score'],
                        'bm25_score': result['bm25_score'],
                        'hybrid_score': result['hybrid_score']
                    }
                )
                final_results.append(doc_copy)
            
            logger.info(f"Hybrid search returned {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to simple semantic search
            return self.vector_store.similarity_search(query, k)