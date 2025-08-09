# app/core/vector_store.py
# FINAL VERSION - Works with the DocumentProcessor

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import logging
import pickle
from pathlib import Path

# Import Document from document_processor
from app.core.document_processor import Document

logger = logging.getLogger(__name__)

class VectorStore:
    """Simple vector store using sklearn for semantic search"""
    
    def __init__(self, persist_directory: Optional[str] = None):
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []
        self.embedding_model = None
        self.persist_directory = persist_directory
        
    def _get_embedding_model(self):
        """Lazy load the embedding model"""
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence transformer model...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Model loaded successfully")
        return self.embedding_model
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        if not documents:
            logger.warning("No documents provided to add")
            return
            
        model = self._get_embedding_model()
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Extract text content
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings in batches to manage memory
        batch_size = 16  # Smaller batch size for stability
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
                embeddings.extend(batch_embeddings)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            except Exception as e:
                logger.error(f"Error encoding batch {i//batch_size + 1}: {e}")
                # Add zero embeddings for failed batches
                zero_embedding = np.zeros(384)  # MiniLM embedding dimension
                embeddings.extend([zero_embedding] * len(batch_texts))
        
        # Store documents and embeddings
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        
        logger.info(f"Vector store now contains {len(self.documents)} documents")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if not self.documents:
            logger.warning("No documents in vector store")
            return []
        
        try:
            model = self._get_embedding_model()
            
            # Encode the query
            query_embedding = model.encode([query], show_progress_bar=False)
            
            # Ensure we have embeddings
            if not self.embeddings:
                logger.error("No embeddings available")
                return self.documents[:k]  # Return first k documents as fallback
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top k most similar documents
            k = min(k, len(self.documents))  # Ensure k doesn't exceed available documents
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents):  # Safety check
                    doc = self.documents[idx]
                    # Add similarity score to metadata
                    doc_copy = Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, 'similarity_score': float(similarities[idx])}
                    )
                    results.append(doc_copy)
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            # Return first k documents as fallback
            return self.documents[:min(k, len(self.documents))]
    
    def save(self, path: str):
        """Save vector store to disk"""
        try:
            data = {
                'documents': self.documents,
                'embeddings': self.embeddings
            }
            
            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved vector store to {path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def load(self, path: str):
        """Load vector store from disk"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data.get('documents', [])
            self.embeddings = data.get('embeddings', [])
            
            logger.info(f"Loaded vector store from {path} with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            # Initialize empty store if loading fails
            self.documents = []
            self.embeddings = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'num_documents': len(self.documents),
            'num_embeddings': len(self.embeddings),
            'embedding_dimension': len(self.embeddings[0]) if self.embeddings else 0
        }