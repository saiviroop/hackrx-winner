# app/core/document_processor.py
# Simplified version without LangChain dependency

import logging
from typing import List, Dict, Any
import pypdf
from pathlib import Path
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]

class SimpleTextSplitter:
    """Simple text splitter replacement for LangChain"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending punctuation
                for i in range(end, start + self.chunk_size//2, -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks

class DocumentProcessor:
    """Process documents and create text chunks"""
    
    def __init__(self):
        self.text_splitter = SimpleTextSplitter(chunk_size=1000, chunk_overlap=200)
        
    def process_document(self, file_path: str) -> List[Document]:
        """Process a single document and return chunks"""
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Extract text from PDF
            text = self._extract_text_from_pdf(file_path)
            
            if not text.strip():
                logger.warning(f"No text extracted from {file_path}")
                return []
            
            # Split into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create Document objects
            documents = []
            filename = Path(file_path).name
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'source': filename,
                            'chunk_id': i,
                            'chunk_size': len(chunk)
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Created {len(documents)} chunks from {filename}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return []
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using pypdf"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            # Clean up the text
                            page_text = self._clean_text(page_text)
                            text += f"\nPage {page_num + 1}:\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        continue
            
            return text
            
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {str(e)}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\"\'\/\%\$\@]', '', text)
        
        # Normalize line breaks
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        return text.strip()
    
    def process_documents(self, file_paths: List[str]) -> List[Document]:
        """Process multiple documents"""
        all_documents = []
        
        for file_path in file_paths:
            documents = self.process_document(file_path)
            all_documents.extend(documents)
        
        logger.info(f"Processed {len(file_paths)} files, created {len(all_documents)} total chunks")
        return all_documents