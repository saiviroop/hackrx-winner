# app/core/vector_store.py
# Ultra-minimal version without NumPy/SciKit-Learn

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
    """Ultra-minimal vector store using only OpenAI embeddings"""
    
    def __init__(self, persist_directory: Optional[str] = None):
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.persist_directory = persist_directory
        
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",  # Cheaper, smaller model
                input=text[:8000]  # Limit text length to avoid token limits
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {e}")
            # Return zero vector as fallback (1536 dimensions for text-embedding-3-small)
            return [0.0] * 1536
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Simple cosine similarity without numpy"""
        try:
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            
            # Calculate magnitudes
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            
            # Avoid division by zero
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
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
                embeddings.append([0.0] * 1536)
        
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
            similarities = []
            for doc_embedding in self.embeddings:
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append(similarity)
            
            # Get top k most similar documents
            k = min(k, len(self.documents))
            
            # Create list of (similarity, index) pairs and sort by similarity
            similarity_indices = [(similarities[i], i) for i in range(len(similarities))]
            similarity_indices.sort(key=lambda x: x[0], reverse=True)
            
            # Get top k results
            results = []
            for similarity, idx in similarity_indices[:k]:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    doc_copy = Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, 'similarity_score': similarity}
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