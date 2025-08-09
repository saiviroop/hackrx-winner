from app.core.vector_store import Document
import os
import hashlib
from typing import List, Dict, Any, Tuple
import pypdf
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", ";", ",", " ", ""],
            length_function=len,
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from PDF"""
        text = ""
        metadata = {
            "source": pdf_path,
            "pages": 0,
            "file_hash": self._get_file_hash(pdf_path)
        }
        
        try:
            # Try pdfplumber first (better for tables)
            with pdfplumber.open(pdf_path) as pdf:
                metadata["pages"] = len(pdf.pages)
                metadata["title"] = pdf.metadata.get('Title', os.path.basename(pdf_path))
                
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying pypdf: {e}")
            # Fallback to pypdf
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                metadata["pages"] = len(pdf_reader.pages)
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                    
        return self._clean_text(text), metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep Indian currency symbol
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\/\₹\%]', '', text)
        # Fix common OCR errors
        text = text.replace('lnsured', 'Insured')
        text = text.replace('Hospit al', 'Hospital')
        return text.strip()
    
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create overlapping chunks with metadata"""
        chunks = self.text_splitter.split_text(text)
        
        chunk_documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                "chunk_id": i,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks)
            }
            
            # Add section detection
            section = self._detect_section(chunk)
            if section:
                chunk_metadata["section"] = section
                
            chunk_documents.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
            
        return chunk_documents
    
    def _detect_section(self, text: str) -> str:
        """Detect which section of document this chunk belongs to"""
        text_lower = text.lower()
        
        if "claim" in text_lower and "procedure" in text_lower:
            return "claims_procedure"
        elif "exclusion" in text_lower:
            return "exclusions"
        elif "coverage" in text_lower or "benefit" in text_lower:
            return "coverage"
        elif "definition" in text_lower:
            return "definitions"
        elif "premium" in text_lower:
            return "premium"
        else:
            return "general"
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate file hash for caching"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def process_documents(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple PDF documents"""
        all_chunks = []
        
        for pdf_path in pdf_paths:
            logger.info(f"Processing: {pdf_path}")
            text, metadata = self.extract_text_from_pdf(pdf_path)
            chunks = self.create_chunks(text, metadata)
            all_chunks.extend(chunks)
            
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks