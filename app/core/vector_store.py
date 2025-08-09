# app/core/vector_store.py
# Final working version with OpenAI embeddings

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
    """Vector store using OpenAI embeddings"""
    
    def __init__(self, persist_directory: Optional[str] = None):
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.persist_directory = persist_directory
        
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text[:8000]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {e}")
            return [0.0] * 1536
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity without numpy"""
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        except Exception:
            return 0.0
    
    def add_documents(self, documents: List[Document]):
        """Add documents to vector store"""
        if not documents:
            return
            
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        embeddings = []
        for i, doc in enumerate(documents):
            try:
                embedding = self._get_embedding(doc.page_content)
                embeddings.append(embedding)
                logger.info(f"Processed document {i+1}/{len(documents)}")
                
                import time
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error processing document {i+1}: {e}")
                embeddings.append([0.0] * 1536)
        
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        logger.info(f"Vector store now contains {len(self.documents)} documents")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if not self.documents:
            return []
        
        try:
            query_embedding = self._get_embedding(query)
            
            if not self.embeddings:
                return self.documents[:k]
            
            similarities = []
            for doc_embedding in self.embeddings:
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                similarities.append(similarity)
            
            k = min(k, len(self.documents))
            similarity_indices = [(similarities[i], i) for i in range(len(similarities))]
            similarity_indices.sort(key=lambda x: x[0], reverse=True)
            
            results = []
            for similarity, idx in similarity_indices[:k]:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    doc_copy = Document(
                        page_content=doc.page_content,
                        metadata={**doc.metadata, 'similarity_score': similarity}
                    )
                    results.append(doc_copy)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return self.documents[:k]
    
    def save(self, path: str):
        """Save vector store"""
        try:
            data = {'documents': self.documents, 'embeddings': self.embeddings}
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving: {e}")
    
    def load(self, path: str):
        """Load vector store"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.documents = data.get('documents', [])
            self.embeddings = data.get('embeddings', [])
        except Exception as e:
            logger.error(f"Error loading: {e}")
            self.documents = []
            self.embeddings = []