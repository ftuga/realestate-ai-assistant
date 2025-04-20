import os
import re
import json
import fitz
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import nltk
from nltk.tokenize import sent_tokenize
import logging
import unicodedata
import requests
from datetime import datetime
from io import BytesIO


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class RealEstatePDFProcessor:
    
    PATTERNS = {
            'name': r'^([A-Z][A-Z\s]+)(?:\n|$)',  
            'price_range': r'PRICE RANGE\s*\n\s*([^\n]+)',
            'amenities': r'AMENITIES\s*\n([\s\S]*?)(?:\n\s*[A-Z ]+\s*\n|\Z)',
            'building_info': r'THE BUILDING\s*\n([\s\S]*?)(?:\n\s*[A-Z ]+\s*\n|\Z)',
            'developer': r'DEVELOPER\s*\n([^\n]+)',
            'completion': r'COMPLETION DATE\s*\n([^\n]+)'
        }
    
    def __init__(self, min_chunk_size: int = 200, max_chunk_size: int = 1000):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
                
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        
        if not isinstance(text, str):
            text = str(text)
        
        text = text.lower()
        
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        
        text = re.sub(r'[^\w\s.,;:!?-]', ' ', text)
        
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        overlap_size = min(200, self.max_chunk_size // 4)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            if len(current_chunk) + len(sentence) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                chunks.append(current_chunk.strip())
                
                words = current_chunk.split()
                if len(words) > overlap_size // 10:
                    overlap_text = " ".join(words[-overlap_size // 10:])
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk.strip())
        elif current_chunk:
            if chunks:
                chunks[-1] = chunks[-1] + " " + current_chunk
            else:
                chunks.append(current_chunk.strip())
                
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_size * 1.5:
                sub_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
                
        logger.info(f"Created {len(final_chunks)} chunks from text of length {len(text)}")
        return final_chunks
    
    def improved_chunk_text(self, text: str) -> List[str]:
        """Chunk text with improved logic from DAG"""
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            min_chunk_size = self.min_chunk_size
            max_chunk_size = self.max_chunk_size
            overlap_size = min(200, self.max_chunk_size // 4)
            
            sentences = sent_tokenize(text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                if len(current_chunk) + len(sentence) > max_chunk_size and len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                    
                    words = current_chunk.split()
                    if len(words) > overlap_size // 10:
                        overlap_text = " ".join(words[-overlap_size // 10:])
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
            elif current_chunk:
                if chunks:
                    chunks[-1] = chunks[-1] + " " + current_chunk
                else:
                    chunks.append(current_chunk.strip())
            
        except Exception as e:
            logger.error(f"Error in sentence tokenization, using basic chunking: {e}")
            chunks = self.chunk_text(text)
        
        return chunks

    def _split_large_chunk(self, chunk: str) -> List[str]:
        sentences = sent_tokenize(chunk)
        sub_chunks = []
        current_sub_chunk = ""
        
        for sentence in sentences:
            if len(current_sub_chunk) + len(sentence) > self.max_chunk_size and current_sub_chunk:
                sub_chunks.append(current_sub_chunk.strip())
                current_sub_chunk = sentence
            else:
                current_sub_chunk += " " + sentence if current_sub_chunk else sentence
        
        if current_sub_chunk:
            sub_chunks.append(current_sub_chunk.strip())
            
        return sub_chunks
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        metadata = {}
        
        for key, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if key in ['price', 'area', 'rooms', 'bathrooms', 'year_built']:
                    try:
                        clean_values = [re.sub(r',', '', match) for match in matches]
                        numeric_values = [float(val) if '.' in val else int(val) for val in clean_values]
                        metadata[key] = numeric_values
                    except ValueError:
                        metadata[key] = matches
                else:
                    metadata[key] = matches
        
        return metadata
    
    def _estimate_page_number(self, text_position, full_text, total_pages):
        """Estimate page number based on text position in the document"""
        if text_position < 0 or total_pages <= 1:
            return 1
        
        relative_position = text_position / len(full_text)
        estimated_page = max(1, min(total_pages, round(relative_position * total_pages)))
        return estimated_page
    
    def process_pdf(self, pdf_path: str, doc_id: str) -> Dict[str, Any]:
        logger.info(f"Processing PDF: {pdf_path}")
        
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        cleaned_text = self.clean_text(raw_text)
        logger.info(f"Cleaned text: reduced from {len(raw_text)} to {len(cleaned_text)} characters")
        
        chunks = self.improved_chunk_text(cleaned_text)
        logger.info(f"Created {len(chunks)} text chunks")
        
        metadata = self.extract_metadata(raw_text)
        logger.info(f"Extracted metadata: {list(metadata.keys())}")
        
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            try:
                import requests
                ollama_url = os.environ.get('OLLAMA_API_URL', 'http://ollama:11434')
                embedding_endpoint = f"{ollama_url}/api/embeddings"
                
                response = requests.post(
                    embedding_endpoint,
                    json={
                        "model": "nomic-embed-text",
                        "prompt": chunk_text
                    }
                )
                
                if response.status_code == 200:
                    chunk_embedding = response.json().get('embedding', [])
                else:
                    logger.error(f"Error getting embedding for chunk {i}, using empty embedding")
                    chunk_embedding = []
            except Exception as e:
                logger.error(f"Exception getting chunk embedding: {e}")
                chunk_embedding = []
            
            raw_chunk_idx = raw_text.find(chunk_text)
            raw_chunk_text = raw_text[raw_chunk_idx:raw_chunk_idx + len(chunk_text) + 100] if raw_chunk_idx >= 0 else chunk_text
            chunk_metadata = self.extract_metadata(raw_chunk_text)
            
            chunk_data = {
                "chunk_id": f"{doc_id}_chunk_{i}",
                "text": chunk_text,
                "embedding": chunk_embedding,
                "metadata": {
                    **chunk_metadata,
                    "page_num": self._estimate_page_number(raw_chunk_idx, raw_text, self._get_page_count(pdf_path))
                }
            }
            processed_chunks.append(chunk_data)
        
        result = {
            "doc_id": doc_id,
            "metadata": {
                **metadata,
                "original_filename": os.path.basename(pdf_path),
                "processed_date": datetime.now().isoformat()
            },
            "page_count": self._get_page_count(pdf_path),
            "chunks": processed_chunks,
            "full_text_length": len(raw_text),
            "clean_text_length": len(cleaned_text),
            "chunk_count": len(chunks)
        }
        
        return result
    
    def _get_page_count(self, pdf_path: str) -> int:
        try:
            doc = fitz.open(pdf_path)
            return len(doc)
        except Exception as e:
            logger.error(f"Error getting page count: {e}")
            return 0
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[pd.DataFrame]:
        tables = []
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                table_data = page.find_tables()
                if table_data:
                    for table in table_data:
                        df = pd.DataFrame(table.extract())
                        if not df.empty and all(isinstance(col, str) for col in df.iloc[0]):
                            df.columns = df.iloc[0]
                            df = df.iloc[1:]
                        tables.append(df)
            
            return tables
        except Exception as e:
            logger.error(f"Error extracting tables from PDF: {e}")
            return []

    def extract_images_from_pdf(self, pdf_path: str, output_dir: str) -> List[str]:
        image_paths = []
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    image_ext = base_image["ext"]
                    
                    image_name = f"page{page_num+1}_img{img_index}.{image_ext}"
                    image_path = os.path.join(output_dir, image_name)
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    image_paths.append(image_path)
            
            return image_paths
        except Exception as e:
            logger.error(f"Error extracting images from PDF: {e}")
            return []
            
    def check_pdf_has_text(self, pdf_path: str) -> bool:
        try:
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                total_text = ""
                pages_to_check = min(5, len(pdf.pages))
                
                for i in range(pages_to_check):
                    page = pdf.pages[i]
                    text = page.extract_text() or ""
                    total_text += text
                
                if len(total_text.strip()) > 100:
                    return True
                return False
        except Exception as e:
            logger.error(f"Error checking PDF text content: {e}")
            return False