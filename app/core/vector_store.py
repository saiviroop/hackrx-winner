# app/core/vector_store.py
# Ultra-lightweight version using OpenAI embeddings

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional
import logging
import pickle
from pathlib import Path
import openai
from app.config import get_settings
from app.core.document_processor import Document

logger = logging.getLogger(__name__)
settings = get_settings()

class VectorStore:
    """Ultra-lightweight vector store using OpenAI embeddings"""
    
    def __init__(self, persist_directory: Optional[str] = None):
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.persist_directory = persist_directory
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",  # Smaller, cheaper model
                input=text[:8000]  # Limit text length
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(1536)  # text-embedding-3-small dimension
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        if not documents:
            logger.warning("No documents provided to add")
            return
            
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Generate embeddings one by one to manage API limits
        embeddings = []
        for i, doc in enumerate(documents):
            try:
                embedding = self._get_embedding(doc.page_content)
                embeddings.append(embedding)
                logger.info(f"Processed document {i+1}/{len(documents)}")
                
                # Small delay to avoid rate limits
                import time
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing document {i+1}: {e}")
                # Add zero embedding for failed documents
                embeddings.append(np.zeros(1536))
        
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
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Ensure we have embeddings
            if not self.embeddings:
                logger.error("No embeddings available")
                return self.documents[:k]
            
            # Calculate similarities
            similarities = cosine_similarity([query_embedding], self.embeddings)[0]
            
            # Get top k most similar documents
            k = min(k, len(self.documents))
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    doc_copy = Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, 'similarity_score': float(similarities[idx])}
                    )
                    results.append(doc_copy)
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return self.documents[:min(k, len(self.documents))]
    
    def save(self, path: str):
        """Save vector store to disk"""
        try:
            data = {
                'documents': self.documents,
                'embeddings': self.embeddings
            }
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
            self.documents = []
            self.embeddings = []