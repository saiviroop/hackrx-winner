import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, embedding_dim: int = 384, index_path: Optional[str] = None):
        self.embedding_dim = embedding_dim
        self.index_path = index_path
        
        # Initialize FAISS index with HNSW for fast search
        self.index = faiss.IndexHNSWFlat(embedding_dim, 32)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 50
        
        self.documents = []
        self.id_to_index = {}
        
        if index_path and os.path.exists(index_path):
            self.load(index_path)
    
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """Add documents to vector store"""
        if len(embeddings) != len(documents):
            raise ValueError("Embeddings and documents must have same length")
        
        start_idx = len(self.documents)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents
        for i, doc in enumerate(documents):
            doc_id = f"doc_{start_idx + i}"
            doc['id'] = doc_id
            self.documents.append(doc)
            self.id_to_index[doc_id] = start_idx + i
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents"""
        if len(self.documents) == 0:
            return []
        
        # Ensure correct shape
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, len(self.documents)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(1 / (1 + dist))  # Convert distance to similarity
                results.append((doc, doc['score']))
        
        return results
    
    def save(self, path: str):
        """Save index and documents"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save documents
        with open(f"{path}.docs", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'id_to_index': self.id_to_index
            }, f)
        
        logger.info(f"Saved vector store to {path}")
    
    def load(self, path: str):
        """Load index and documents"""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.index")
        
        # Load documents
        with open(f"{path}.docs", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.id_to_index = data['id_to_index']
        
        logger.info(f"Loaded vector store from {path}")