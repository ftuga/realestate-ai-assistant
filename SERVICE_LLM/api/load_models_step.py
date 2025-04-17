import os
import json
import requests
import logging
import time
import threading
import torch
from minio import Minio
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_manager")

OLLAMA_API_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
HTTP_TIMEOUT = 30
REAL_ESTATE_MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"
MODEL_CHECK_INTERVAL = 60 
MODELS_DIR = os.environ.get("MODELS_DIR", "/opt/airflow/models")
LLM_MODEL_DIR = f"{MODELS_DIR}/mistral"
VERSION_FILE = f"{MODELS_DIR}/model_versions.json"
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = "llm-data"


llm_model_available = False
embedding_model_available = False
model_check_thread = None
model_rules = None

MODEL_MAPPING = {
    "real_estate_model": REAL_ESTATE_MODEL,
    "embedding_model": EMBEDDING_MODEL
}

class ModelManager:
    def __init__(self):
        self.llm_model_available = False
        self.embedding_model_available = False
        self.model_rules = None
        self.model_check_thread = None
        self.device = TORCH_DEVICE
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(LLM_MODEL_DIR, exist_ok=True)
        
        self.model_rules = self.load_model_rules()
        
        self.version_info = self.get_model_version()
        
        logger.info(f"Model manager initialized. Using device: {self.device}")
        logger.info(f"Using fixed model: {REAL_ESTATE_MODEL}")
    
    def load_model_rules(self):
        """Load model rules from JS file created by DAG"""
        try:
            rules_path = os.path.join(LLM_MODEL_DIR, "rules.js")
            if os.path.exists(rules_path):
                with open(rules_path, 'r') as f:
                    content = f.read()
                    
                    start_marker = "export const modelRules = "
                    end_marker = "};"
                    
                    start_index = content.find(start_marker) + len(start_marker)
                    end_index = content.find(end_marker, start_index) + 1
                    
                    if start_index > len(start_marker) and end_index > 0:
                        json_str = content[start_index:end_index]
                        model_rules = json.loads(json_str)
                        logger.info(f"Model rules loaded: {model_rules.get('name')} v{model_rules.get('version')}")
                        return model_rules
                    else:
                        logger.error("Could not extract modelRules from JS file")
            else:
                logger.warning(f"Rules file not found at {rules_path}")
        except Exception as e:
            logger.error(f"Error loading model rules: {e}")
        
        return None
    
    def get_model_version(self):
        """Get the current model version from version file created by DAG"""
        try:
            if os.path.exists(VERSION_FILE):
                with open(VERSION_FILE, 'r') as f:
                    versions = json.load(f)
                    llm_info = versions.get('llm_model', {})
                    embedding_info = versions.get('embedding_model', {})
                    
                    return {
                        "llm_model": {
                            "name": llm_info.get('name', REAL_ESTATE_MODEL),
                            "version": llm_info.get('version', '0.0.0'),
                            "last_update": llm_info.get('last_update', '')
                        },
                        "embedding_model": {
                            "name": embedding_info.get('name', EMBEDDING_MODEL),
                            "version": embedding_info.get('version', '0.0.0'),
                            "last_update": embedding_info.get('last_update', '')
                        }
                    }
        except Exception as e:
            logger.error(f"Error reading model version file: {e}")
        
        return {
            "llm_model": {"name": REAL_ESTATE_MODEL, "version": "unknown", "last_update": ""},
            "embedding_model": {"name": EMBEDDING_MODEL, "version": "unknown", "last_update": ""}
        }

    async def check_and_download_models(self):
        """Check if models are available and download if needed"""
        try:
            logger.info(f"Checking LLM model: {REAL_ESTATE_MODEL}")
            llm_available = False
            
            try:
                response = requests.get(
                    f"{OLLAMA_API_URL}/api/tags",
                    timeout=HTTP_TIMEOUT
                )
                
                if response.status_code == 200:
                    available_models = [model["name"] for model in response.json().get("models", [])]
                    llm_available = REAL_ESTATE_MODEL in available_models
                    self.llm_model_available = llm_available
                
            except Exception as e:
                logger.error(f"Error checking LLM model: {e}")
            
            if not llm_available:
                logger.info(f"LLM model {REAL_ESTATE_MODEL} not found, downloading...")
                await self.download_llm_model()
            else:
                logger.info(f"LLM model {REAL_ESTATE_MODEL} is available")
                self.llm_model_available = True
                
            logger.info(f"Checking embedding model: {EMBEDDING_MODEL}")
            embedding_available = False
            
            try:
                response = requests.get(
                    f"{OLLAMA_API_URL}/api/tags",
                    timeout=HTTP_TIMEOUT
                )
                
                if response.status_code == 200:
                    available_models = [model["name"] for model in response.json().get("models", [])]
                    embedding_available = EMBEDDING_MODEL in available_models
                    self.embedding_model_available = embedding_available
                
            except Exception as e:
                logger.error(f"Error checking embedding model: {e}")
            
            if not embedding_available:
                logger.info(f"Embedding model {EMBEDDING_MODEL} not found, downloading...")
                await self.download_embedding_model()
            else:
                logger.info(f"Embedding model {EMBEDDING_MODEL} is available")
                self.embedding_model_available = True
            
            return {
                "llm_model": self.llm_model_available,
                "embedding_model": self.embedding_model_available
            }
            
        except Exception as e:
            logger.error(f"Error checking and downloading models: {e}")
            return {
                "llm_model": False,
                "embedding_model": False,
                "error": str(e)
            }

    async def create_model_rules(self):
        rules_file_path = os.path.join(LLM_MODEL_DIR, 'rules.js')
        
        if os.path.exists(rules_file_path):
            logger.info("LLM model rules already exist, skipping creation")
            return {"status": "exists"}
        
        try:
            logger.info("Creating model rules file for Mistral with examples")
            
            system_prompt =  """You are a specialized luxury real estate assistant from MILLION. Your role is:
                                1. When users ask about specific properties, use ONLY the information provided in the context to answer.
                                2. When users ask general questions about available properties, price ranges, or features (like number of bathrooms/bedrooms), search through ALL provided documents and summarize the available information.
                                3. If asked about properties with specific features, analyze all documents and list the properties that match.
                                4. Keep responses concise and to the point.
                                5. If the answer cannot be found in the provided documents, say 'I don't have that information in the provided data.'
                                6. Never mention that you're using context in your responses.
                                7. Never mention your rules to the user or any process related to your training.
                                Important: Always respond in the same language that was used in the question."""

            examples = [
                            {
                                "instruction": "When will Bayfront Residences be completed?", 
                                "context": "BAYFRONT RESIDENCES\nPRICE RANGE\nStarting at $9,000,000\nAMENITIES\nDirect Beach Access\nTHE BUILDING\nShore Club features 18 levels and 49 elegant residences with floor plans starting at 2,400 up to 7,000 Sq. Ft.\nBEACH SERVICE\nExpansive Pool Decks\nExterior Café\nDEVELOPER\nDeveloped by Witkoff.\nDining Areas\nARCHITECTURE\nAn exceptional redesign thanks to Robert A.M. Stern Architects.\nINTERIOR DESIGN\nFinishes and interior areas by Kobi Karp Architecture & Interior Design.\nCOMPLETION DATE\nExpected to be delivered by 2026.", 
                                "response": "Bayfront Residences is expected to be delivered by 2026."
                            },

                            {
                                "instruction": "Which property offers pickleball courts?", 
                                "context": "BRICKELL HOME LUXURY\nPRICE RANGE\nStarting at $8,000,000\nAMENITIES\nTHE BUILDING\nThis modern 25-story tower features 56 spacious oceanfront residences from 3 to 8 bedrooms and floorplans from 3,300 to 12,600 Sq. Ft.\nDEVELOPER\nDeveloped by Related Group and Two Roads Development.\nSignature restaurant exclusive to residents\nCurated social spaces\nResort-style pools\nOutdoor spa\nPoolside services\nARCHITECTURE\nDesigned by Skidmore, Owings & Merrill - SOM, one of the world's top architectural firms.\nPickleball court\nState-of-the-art fitness areas", 
                                "response": "Brickell Home Luxury offers pickleball courts."
                            },

                            {
                                "instruction": "Tell me about the developers of UNA CLUB", 
                                "context": "UNA CLUB\nPRICE RANGE\nStarting at $4,500,000\nAMENITIES\nTHE BUILDING\nFeaturing two towers and 62 floors with 352 lavish residences, from 2 to 5 bedrooms spanning between 1,785 and 10,000 Sq. Ft.\nPrivate beach amenities include cabanas, chaise lounges, and umbrellas.\nChildren / teen's club.\nDEVELOPER\nDeveloped by Fortune International and Château Group.", 
                                "response": "UNA CLUB is developed by Fortune International and Château Group."
                            },

                            {
                                "instruction": "What's the most expensive property available?", 
                                "context": "UNA CLUB\nPRICE RANGE\nStarting at $4,500,000\n\nBRICKELL HOME LUXURY\nPRICE RANGE\nStarting at $8,000,000\n\nBAYFRONT RESIDENCES\nPRICE RANGE\nStarting at $9,000,000", 
                                "response": "Bayfront Residences is the most expensive property available, with prices starting at $9,000,000."
                            },

                            {
                                "instruction": "Which properties will be completed by 2026?", 
                                "context": "UNA CLUB\nCOMPLETION DATE\nThe South Tower will be delivered in 2027 and the North Tower will be delivered in 2029.\n\nBRICKELL HOME LUXURY\nCOMPLETION DATE\nThe condominium is expected to be ready for delivery by 2026.\n\nBAYFRONT RESIDENCES\nCOMPLETION DATE\nExpected to be delivered by 2026.", 
                                "response": "Both Brickell Home Luxury and Bayfront Residences will be completed by 2026."
                            },

                            {
                                "instruction": "What are the dining options at Brickell Home Luxury?", 
                                "context": "BRICKELL HOME LUXURY\nAMENITIES\nSignature oceanfront restaurant by a celebrated chef exclusive to residents.\nBeachside food and beverage service.\nSprawling conservatory offering daily breakfast bar.\nStylish cocktail lounge and speakeasy.\nSunrise lounge with catering kitchen for entertaining and events.", 
                                "response": "Brickell Home Luxury offers a signature oceanfront restaurant by a celebrated chef (exclusive to residents), beachside food and beverage service, a sprawling conservatory with daily breakfast bar, a stylish cocktail lounge and speakeasy, and a sunrise lounge with catering kitchen for entertaining and events."
                            },

                            {
                                "instruction": "Which property has the largest residences?", 
                                "context": "UNA CLUB\nTHE BUILDING\nFeaturing two towers and 62 floors with 352 lavish residences, from 2 to 5 bedrooms spanning between 1,785 and 10,000 Sq. Ft.\n\nBRICKELL HOME LUXURY\nTHE BUILDING\nThis modern 25-story tower features 56 spacious oceanfront residences from 3 to 8 bedrooms and floorplans from 3,300 to 12,600 Sq. Ft.\n\nBAYFRONT RESIDENCES\nTHE BUILDING\nShore Club features 18 levels and 49 elegant residences with floor plans starting at 2,400 up to 7,000 Sq. Ft.", 
                                "response": "Brickell Home Luxury has the largest residences, with floorplans up to 12,600 Sq. Ft."
                            },

                            {
                                "instruction": "Tell me about the location of Brickell Home Luxury", 
                                "context": "BRICKELL HOME LUXURY\nLOCATION\nBal Harbour was born of a vision of infinite possibility, making it an internationally renowned destination. Today, Bal Harbour is among the most elegant, exclusive, and sought-after beach destinations in South Florida.", 
                                "response": "Brickell Home Luxury is located in Bal Harbour, an elegant, exclusive, and sought-after beach destination in South Florida. It's internationally renowned and was born of a vision of infinite possibility."
                            },

                            {
                                "instruction": "Do any properties include wellness amenities?", 
                                "context": "UNA CLUB\nAMENITIES\nOutdoor wellness & relaxation terrace with plunge pools.\n\nBRICKELL HOME LUXURY\nAMENITIES\nBeachside swimming pool and outdoor spa with sunbeds and cabanas.\nOutdoor spa\nSpa, fitness wellness center, and Bicycle storage.", 
                                "response": "Yes, both UNA CLUB and Brickell Home Luxury include wellness amenities. UNA CLUB offers an outdoor wellness & relaxation terrace with plunge pools, while Brickell Home Luxury features a beachside swimming pool and outdoor spa with sunbeds and cabanas, as well as a spa and fitness wellness center."
                            },

                            {
                                "instruction": "What towers does UNA CLUB have?", 
                                "context": "UNA CLUB\nTHE BUILDING\nFeaturing two towers and 62 floors with 352 lavish residences, from 2 to 5 bedrooms spanning between 1,785 and 10,000 Sq. Ft.\nCOMPLETION DATE\nThe South Tower will be delivered in 2027 and the North Tower will be delivered in 2029.", 
                                "response": "UNA CLUB features two towers: the South Tower (to be delivered in 2027) and the North Tower (to be delivered in 2029)."
                            }]
                                                    
            rules_js = f"""
                        export const modelRules = {{
                            "name": "mistral",
                            "version": "1.0.0",
                            "description": "Real estate assistant based on Mistral",
                            "systemPrompt": `{system_prompt}`,
                            "examples":{examples},
                            "rules": [
                                        "Search through all documents when asked about property features",
                                        "List matching properties for filtered queries",
                                        "Reply 'I don't have that information' when appropriate",
                                        "Keep responses concise and to the point",
                                        "Don't add disclaimers or explanations",
                                        "Never mention using context in responses",
                                        "Respond in the same language as the question"
                                    ],
                            "parameters": {{
                                "temperature": 0.01,
                                "top_p": 0.9,
                                "top_k": 40,
                                "num_predict": 1024
                            }},
                            "format_template": "[INST] {{SYSTEM}}\\n\\n{{INSTRUCTION}} [/INST]"
                        }};

                        export default modelRules;
                        """
                            
            with open(rules_file_path, 'w') as f:
                f.write(rules_js)
            
            self.model_rules = self.load_model_rules()
            
            logger.info("Model rules created successfully for Mistral with examples")
            return {"status": "created"}
            
        except Exception as e:
            logger.error(f"Error creating model rules: {e}")
            return {"status": "error", "message": str(e)}
        
    
    
    async def download_llm_model(self):
        """Download the LLM model if not available"""
        try:
            logger.info(f"Downloading LLM model: {REAL_ESTATE_MODEL}")
            
            response = requests.post(
                f"{OLLAMA_API_URL}/api/pull",
                json={"name": REAL_ESTATE_MODEL},
                timeout=1800  
            )
            
            if response.status_code == 200:
                logger.info(f"LLM model {REAL_ESTATE_MODEL} downloaded successfully")
                self.llm_model_available = True
                
                await self.create_model_rules()
                
                self.update_version_info("llm_model", REAL_ESTATE_MODEL)
                
                return {"status": "success", "model": REAL_ESTATE_MODEL}
            else:
                logger.error(f"Error downloading LLM model: {response.status_code} - {response.text}")
                return {"status": "error", "message": response.text}
                
        except Exception as e:
            logger.error(f"Exception downloading LLM model: {e}")
            return {"status": "error", "message": str(e)}
    
    async def download_embedding_model(self):
        """Download the embedding model if not available"""
        try:
            logger.info(f"Downloading embedding model: {EMBEDDING_MODEL}")
            
            response = requests.post(
                f"{OLLAMA_API_URL}/api/pull",
                json={"name": EMBEDDING_MODEL},
                timeout=600  
            )
            
            if response.status_code == 200:
                logger.info(f"Embedding model {EMBEDDING_MODEL} downloaded successfully")
                self.embedding_model_available = True
                
                self.update_version_info("embedding_model", EMBEDDING_MODEL)
                
                return {"status": "success", "model": EMBEDDING_MODEL}
            else:
                logger.error(f"Error downloading embedding model: {response.status_code} - {response.text}")
                return {"status": "error", "message": response.text}
                
        except Exception as e:
            logger.error(f"Exception downloading embedding model: {e}")
            return {"status": "error", "message": str(e)}
    
    def update_version_info(self, model_type, model_name):
        """Update version info file"""
        try:
            versions = self.get_model_version()
            
            versions[model_type] = {
                'name': model_name,
                'version': '1.0.0',
                'last_update': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(VERSION_FILE), exist_ok=True)
            
            with open(VERSION_FILE, 'w') as f:
                json.dump(versions, f, indent=2)
            
            self.version_info = versions
            
            logger.info(f"Updated version info for {model_type}")
            return True
        except Exception as e:
            logger.error(f"Error updating version info: {e}")
            return False
    
    def continuous_model_check(self):
        """Continuously check and ensure models are available"""
        while True:
            try:
                import asyncio
                asyncio.run(self.check_and_download_models())
                
                if self.llm_model_available and self.model_rules is None:
                    self.model_rules = self.load_model_rules()
                    if self.model_rules is None:
                        asyncio.run(self.create_model_rules())
                
                check_interval = MODEL_CHECK_INTERVAL * 2 if self.llm_model_available else MODEL_CHECK_INTERVAL
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in continuous model check: {e}")
                time.sleep(MODEL_CHECK_INTERVAL)
    
    def start_continuous_checking(self):
        """Start background thread for continuous model checking"""
        if self.model_check_thread is None or not self.model_check_thread.is_alive():
            self.model_check_thread = threading.Thread(
                target=self.continuous_model_check,
                daemon=True
            )
            self.model_check_thread.start()
            logger.info("Started continuous model checking")
    
    def get_model_status(self):
        """Get current status of models"""
        return {
            "llm_model": {
                "name": REAL_ESTATE_MODEL,
                "available": self.llm_model_available,
                "version": self.version_info.get("llm_model", {}).get("version", "unknown"),
                "last_update": self.version_info.get("llm_model", {}).get("last_update", "")
            },
            "embedding_model": {
                "name": EMBEDDING_MODEL,
                "available": self.embedding_model_available,
                "version": self.version_info.get("embedding_model", {}).get("version", "unknown"),
                "last_update": self.version_info.get("embedding_model", {}).get("last_update", "")
            },
            "torch_device": self.device,
            "rules_loaded": self.model_rules is not None
        }

model_manager = ModelManager()

def initialize_models():
    """Initialize model manager and start continuous checking"""
    model_manager.start_continuous_checking()
    return model_manager.get_model_status()

def get_minio_client():
    """Get a Minio client"""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False 
    )

def get_models_status():
    """Get current model status"""
    return model_manager.get_model_status()

async def ensure_models_available():
    """Ensure models are available, downloading if needed"""
    return await model_manager.check_and_download_models()

def verify_real_estate_model():
    """Verifica si el modelo de real estate está disponible"""
    return model_manager.llm_model_available

def is_property_filter_query(query):
    filter_keywords = [
        "price", "cost", "value", 
        "bathroom", "bedroom", "room", 
        "pool", "beach", "available",
        "features", "amenities",
        "square feet", "size"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in filter_keywords)

def get_relevant_chunks(query, doc_ids=None, top_k=5):
    try:
        if not query:
            return []
            
        if not model_manager.embedding_model_available:
            logger.warning("Embedding model not available, cannot retrieve relevant chunks")
            return []
            
        minio_client = get_minio_client()
        
        try:
            response = requests.post(
                f"{OLLAMA_API_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": query},
                timeout=HTTP_TIMEOUT
            )
            
            if response.status_code != 200:
                logger.error(f"Error generating query embedding: {response.status_code} - {response.text}")
                return []
                
            query_embedding = response.json().get("embedding", [])
            
            if not query_embedding:
                logger.error("Empty query embedding returned")
                return []
                
        except Exception as e:
            logger.error(f"Exception generating query embedding: {e}")
            return []
            
        all_chunks = []
        
        if not doc_ids or len(doc_ids) == 0:
            try:
                response = minio_client.get_object(MINIO_BUCKET, "metadata/document_list.json")
                document_list = json.loads(response.read().decode('utf-8'))
                response.close()
                response.release_conn()
                
                doc_ids = [doc.get("doc_id") for doc in document_list 
                          if doc.get("status") == "processed"]
                          
                filter_terms = ["price", "bathroom", "bedroom", "pool", "beach", "available", "amenities", "features"]
                if any(term in query.lower() for term in filter_terms):
                    top_k = 15
                
            except Exception as e:
                logger.error(f"Error getting document list: {e}")
                return []

        document_metadata = {}
                
        for doc_id in doc_ids:
            try:
                response = minio_client.get_object(MINIO_BUCKET, f"embeddings/{doc_id}_embeddings.json")
                embeddings_data = json.loads(response.read().decode('utf-8'))
                response.close()
                response.release_conn()
                
                document_metadata[doc_id] = embeddings_data.get("metadata", {})
                
                chunks = embeddings_data.get("chunks", [])
                
                for chunk in chunks:
                    chunk["doc_id"] = doc_id
                    if "title" in document_metadata[doc_id]:
                        chunk["doc_title"] = document_metadata[doc_id]["title"]
                    all_chunks.append(chunk)
                    
            except Exception as e:
                logger.error(f"Error retrieving embeddings for document {doc_id}: {e}")
                
        if not all_chunks:
            logger.warning("No chunks found in the selected documents")
            return []
            
        def cosine_similarity(vec1, vec2):
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm_a = sum(a * a for a in vec1) ** 0.5
            norm_b = sum(b * b for b in vec2) ** 0.5
            
            if norm_a == 0 or norm_b == 0:
                return 0
                
            return dot_product / (norm_a * norm_b)
            
        for chunk in all_chunks:
            chunk_embedding = chunk.get("embedding", [])
            if chunk_embedding:
                chunk["similarity"] = cosine_similarity(query_embedding, chunk_embedding)
            else:
                chunk["similarity"] = 0
        
        sorted_chunks = sorted(all_chunks, key=lambda x: x.get("similarity", 0), reverse=True)
        top_chunks = sorted_chunks[:top_k*2]  
        
        docs_chunks = {}
        for chunk in top_chunks:
            doc_id = chunk.get("doc_id")
            if doc_id not in docs_chunks:
                docs_chunks[doc_id] = []
            docs_chunks[doc_id].append(chunk)
        
        for doc_id in docs_chunks:
            docs_chunks[doc_id].sort(key=lambda x: x.get("metadata", {}).get("page_num", 0))
        
        organized_chunks = []
        for doc_id, chunks in docs_chunks.items():
            organized_chunks.extend(chunks)
        
        final_chunks = organized_chunks[:top_k]
                
        for chunk in final_chunks:
            if "embedding" in chunk:
                del chunk["embedding"]
                
        logger.info(f"Retrieved {len(final_chunks)} relevant chunks from {len(docs_chunks)} documents")
        return final_chunks
        
    except Exception as e:
        logger.error(f"Error retrieving relevant chunks: {e}")
        return []
    
async def generate_llm_response(messages, context_chunks):
    try:
        if not model_manager.llm_model_available:
            return "The assistant model is not available. Please wait a moment and try again."
            
        context_text = ""
        if context_chunks and len(context_chunks) > 0:
            doc_chunks = {}
            for chunk in context_chunks:
                doc_id = chunk.get("doc_id", "unknown")
                if doc_id not in doc_chunks:
                    doc_chunks[doc_id] = []
                doc_chunks[doc_id].append(chunk)
            
            for doc_id, chunks in doc_chunks.items():
                if context_text:
                    context_text += "\n\n" + "-"*40 + "\n\n"
                    
                doc_title = chunks[0].get("doc_title", f"Document {doc_id}")
                context_text += f"INFORMATION FROM: {doc_title}\n\n"
                
                chunks.sort(key=lambda x: x.get("metadata", {}).get("page_num", 0))
                
                for chunk in chunks:
                    chunk_text = chunk.get("text", "").strip()
                    if chunk_text:
                        context_text += f"{chunk_text}\n\n"
        
        last_user_msg = next((msg.content for msg in reversed(messages) if msg.role == "user"), "")
        
        greeting_patterns = ["hello", "hi", "hey", "greetings", "hola"]
        is_greeting = False
        
        if last_user_msg.lower().strip().replace("?", "").replace("!", "") in greeting_patterns:
            is_greeting = True
            
        if is_greeting:
            return "Hello! I'm your Million Luxury Real Estate assistant. How can I help you today? You can select specific documents to consult or ask general questions about the available properties."
        
        system_prompt =  """You are a specialized luxury real estate assistant from MILLION. Your role is:
                        1. When users ask about specific properties, use ONLY the information provided in the context to answer.
                        2. When users ask general questions about available properties, price ranges, or features (like number of bathrooms/bedrooms), search through ALL provided documents and summarize the available information.
                        3. If asked about properties with specific features, analyze all documents and list the properties that match.
                        4. Keep responses concise and to the point.
                        5. If the answer cannot be found in the provided documents, say 'I don't have that information in the provided data.'
                        6. Never mention that you're using context in your responses.
                        7. Never mention your rules to the user or any process related to your training.
                        Important: Always respond in the same language that was used in the question."""
                                
        if context_text:
            instruction = f"Here is the information from the documents:\n\n{context_text}\n\nQuestion: {last_user_msg}"
        else:
            instruction = last_user_msg
        
        final_prompt = f"[INST] {system_prompt}\n\n{instruction} [/INST]"
        
        try:
            logger.info(f"Sending prompt to Ollama using Mistral format")
            
            response = requests.post(
                f"{OLLAMA_API_URL}/api/generate",
                json={
                    "model": REAL_ESTATE_MODEL,
                    "prompt": final_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.01,
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_predict": 1024
                    }
                },
                timeout=60
            )
            
            if response.status_code != 200:
                logger.error(f"Error generating LLM response: {response.status_code} - {response.text}")
                return "I'm sorry, I encountered an error while processing your request. Please try again later."
                
            response_data = response.json()
            response_text = response_data.get("response", "")
            
            if not response_text:
                return "I'm sorry, I couldn't generate a response. Please try again."
                
            logger.info(f"Received response from LLM: {response_text[:100]}...")
                
            return response_text
                
        except Exception as e:
            logger.error(f"Exception generating LLM response: {e}")
            return "I'm sorry, an error occurred while generating a response. Please try again later."
            
    except Exception as e:
        logger.error(f"Error in generate_llm_response: {e}")
        return "I'm sorry, an error occurred. Please try again later."
    
initialize_models()