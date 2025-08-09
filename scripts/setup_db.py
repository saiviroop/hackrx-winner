#!/usr/bin/env python3
"""Setup vector database with insurance documents"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.document_processor import DocumentProcessor
from app.core.embeddings import EmbeddingGenerator
from app.core.vector_store import VectorStore
from app.core.hybrid_search import HybridSearch
from app.config import get_settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    settings = get_settings()
    
    # Initialize components
    doc_processor = DocumentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    embedding_gen = EmbeddingGenerator(settings.embedding_model)
    vector_store = VectorStore(embedding_dim=embedding_gen.embedding_dim)
    
    # Process documents
    pdf_dir = "data/documents"
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_dir}")
        return
    
    logger.info(f"Processing {len(pdf_files)} PDF files...")
    all_chunks = doc_processor.process_documents(pdf_files)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    texts = [chunk['content'] for chunk in all_chunks]
    embeddings = embedding_gen.encode(texts, batch_size=32)
    
    # Add to vector store
    logger.info("Adding to vector store...")
    vector_store.add_documents(embeddings, all_chunks)
    
    # Build BM25 index
    hybrid_search = HybridSearch(vector_store, embedding_gen)
    hybrid_search.build_bm25_index(all_chunks)
    
    # Save
    vector_store.save(settings.vector_db_path)
    logger.info(f"Database setup complete! Indexed {len(all_chunks)} chunks.")

if __name__ == "__main__":
    setup_database()