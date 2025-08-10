# app/core/document_processor.py
# OPTIMIZED DOCUMENT PROCESSOR FOR HACKRX - FAST & EFFICIENT

import logging
from typing import List, Dict, Any
import pypdf
from pathlib import Path
import re
from dataclasses import dataclass
import io

logger = logging.getLogger(__name__)

@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]

class DocumentProcessor:
    """Fast, optimized document processor for insurance documents"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100, max_docs: int = 20, **kwargs):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_docs = max_docs
        logger.info(f"DocumentProcessor initialized: chunk_size={chunk_size}, max_docs={max_docs}")
    
    def process_pdf_bytes(self, pdf_bytes: bytes) -> List[Document]:
        """
        Fast PDF processing optimized for contest speed
        """
        try:
            logger.info("Starting fast PDF processing...")
            
            # Use io.BytesIO for faster processing
            pdf_file = io.BytesIO(pdf_bytes)
            reader = pypdf.PdfReader(pdf_file)
            
            # Extract text from all pages quickly
            full_text = ""
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():  # Only add non-empty pages
                        full_text += f"\n\nPage {page_num + 1}:\n{text}"
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
                    continue
            
            if not full_text.strip():
                raise ValueError("No text content extracted from PDF")
            
            # Fast text cleaning
            cleaned_text = self._clean_text(full_text)
            
            # Smart chunking for insurance documents
            documents = self._smart_chunk_text(cleaned_text)
            
            # Limit to max_docs for speed
            if len(documents) > self.max_docs:
                documents = documents[:self.max_docs]
            
            logger.info(f"PDF processed successfully: {len(documents)} chunks created")
            return documents
            
        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            # Return a fallback document to avoid complete failure
            return [Document(
                page_content=f"Error processing PDF: {str(e)}",
                metadata={"source": "error", "chunk": 0}
            )]
    
    def _clean_text(self, text: str) -> str:
        """Fast text cleaning optimized for insurance documents"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'Page \d+.*?\n', '', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add spaces between words
        
        return text.strip()
    
    def _smart_chunk_text(self, text: str) -> List[Document]:
        """
        Smart chunking optimized for insurance policy documents
        Preserves important sections and context
        """
        documents = []
        
        # Try to split by logical sections first (insurance-specific)
        section_patterns = [
            r'(?i)(section|clause|article|chapter)\s+\d+',
            r'(?i)(coverage|benefit|exclusion|condition)',
            r'(?i)(definitions?|terms?)',
            r'\n\s*\d+\.',  # Numbered lists
            r'\n\s*[A-Z][A-Z\s]{5,}:',  # All caps headers
        ]
        
        # Split by sections if found
        sections = []
        current_section = ""
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new section
            is_new_section = any(re.search(pattern, line) for pattern in section_patterns)
            
            if is_new_section and current_section and len(current_section) > 100:
                sections.append(current_section)
                current_section = line
            else:
                current_section += " " + line
        
        # Add the last section
        if current_section:
            sections.append(current_section)
        
        # If no good sections found, use simple chunking
        if len(sections) < 2:
            sections = self._simple_chunk(text)
        
        # Create Document objects
        for i, section in enumerate(sections):
            if len(section.strip()) < 50:  # Skip very short sections
                continue
                
            # Ensure chunks are reasonable size
            if len(section) > self.chunk_size * 2:
                # Split large sections
                sub_chunks = self._simple_chunk(section)
                for j, chunk in enumerate(sub_chunks):
                    documents.append(Document(
                        page_content=chunk.strip(),
                        metadata={
                            "source": "insurance_policy",
                            "chunk": f"{i}_{j}",
                            "section_type": "large_section_split"
                        }
                    ))
            else:
                documents.append(Document(
                    page_content=section.strip(),
                    metadata={
                        "source": "insurance_policy", 
                        "chunk": str(i),
                        "section_type": "natural_section"
                    }
                ))
        
        return documents
    
    def _simple_chunk(self, text: str) -> List[str]:
        """Simple overlapping chunking fallback"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to end at sentence boundary
            if end < len(text):
                # Look for sentence ending within reasonable distance
                for i in range(end, min(end + 100, len(text))):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= len(text) or len(chunks) > 50:
                break
        
        return chunks