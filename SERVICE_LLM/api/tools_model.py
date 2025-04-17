import os
import re
import json
import fitz
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import nltk
from nltk.tokenize import sent_tokenize
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class RealEstatePDFProcessor:
    
    PATTERNS = {
        'price': r'(?:price|cost|value)(?:\s*(?:is|of|:))?\s*[$€£]?\s*([0-9,]+(?:\.[0-9]{2})?)',
        'area': r'(?:area|size|surface)(?:\s*(?:is|of|:))?\s*([0-9,]+(?:\.[0-9]{2})?)(?:\s*(?:sq\.?|square)\s*(?:ft|feet|m|meters|metres))',
        'rooms': r'(?:([0-9]+)(?:\s*(?:bed|bedroom|room)s?))',
        'bathrooms': r'(?:([0-9]+)(?:\s*(?:bath|bathroom)s?))',
        'address': r'(?:address|location|situated at|located at)(?:\s*(?:is|:))?\s*([^\n,\.]{5,100})',
        'year_built': r'(?:built|constructed|year)(?:\s*(?:in|:))?\s*([0-9]{4})',
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
    
    def chunk_text(self, text: str) -> List[str]:

        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            if len(current_chunk) + len(sentence) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
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
    
    def process_pdf(self, pdf_path: str, doc_id: str) -> Dict[str, Any]:

        logger.info(f"Processing PDF: {pdf_path}")
        
        full_text = self.extract_text_from_pdf(pdf_path)
        
        chunks = self.chunk_text(full_text)
        logger.info(f"Created {len(chunks)} text chunks")
        
        metadata = self.extract_metadata(full_text)
        logger.info(f"Extracted metadata: {list(metadata.keys())}")
        
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):

            chunk_metadata = self.extract_metadata(chunk_text)
            
            chunk_data = {
                "chunk_id": f"{doc_id}_chunk_{i}",
                "text": chunk_text,
                "metadata": chunk_metadata
            }
            processed_chunks.append(chunk_data)
        
        result = {
            "doc_id": doc_id,
            "metadata": metadata,
            "page_count": self._get_page_count(pdf_path),
            "chunks": processed_chunks,
            "full_text_length": len(full_text),
            "chunk_count": len(chunks)
        }
        
        return result
    
    def _get_page_count(self, pdf_path: str) -> int:
        """Get the number of pages in a PDF file."""
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