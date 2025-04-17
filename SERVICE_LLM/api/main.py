from fastapi import FastAPI, UploadFile, File, WebSocket, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from typing import List, Optional, Any, Dict
import os
import json
import uuid
import time
import asyncio
import requests
from datetime import datetime
from io import BytesIO
import glob
import re
from fastapi_utils.tasks import repeat_every
import load_models_step
from load_models_step import model_manager, get_relevant_chunks, generate_llm_response, verify_real_estate_model, REAL_ESTATE_MODEL, is_property_filter_query
import tools_model
from tools_model import RealEstatePDFProcessor

app = FastAPI(
    title="Million Luxury Real Estate Document Assistant",
    description="API for processing and querying real estate documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


active_sessions = {}
DATA_DIR = os.environ.get("DATA_DIR", "/opt/airflow/data")
EMBEDDINGS_DIR = os.environ.get("EMBEDDINGS_DIR", "/opt/airflow/embeddings")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    processed_date: Optional[str] = None
    status: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    doc_ids: List[str] = []

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    doc_ids: List[str] = []

class DocumentUploadResponse(BaseModel):
    status: str
    document_id: str
    message: str
    timestamp: str

class ProcessDocumentRequest(BaseModel):
    doc_id: str

@app.on_event("startup")
async def startup_tasks():
    """Tasks to execute on API startup"""
    model_manager.start_continuous_checking()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Return the index page"""
    with open("/app/Front/templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/documents")
async def list_documents():
    """List all available documents from Minio"""
    try:
        minio_client = load_models_step.get_minio_client()
        try:
            response = minio_client.get_object(load_models_step.MINIO_BUCKET, "metadata/document_list.json")
            documents_data = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()
            return documents_data
        except Exception as e:
            print(f"Document metadata file not found, scanning bucket: {e}")
            
            documents = []
            objects = minio_client.list_objects(load_models_step.MINIO_BUCKET, prefix="embeddings/", recursive=True)
            
            for obj in objects:
                if obj.object_name.endswith("_embeddings.json"):
                    doc_id = obj.object_name.replace("embeddings/", "").replace("_embeddings.json", "")
                    try:
                        minio_client.stat_object(load_models_step.MINIO_BUCKET, f"documents/{doc_id}.pdf")
                        documents.append({
                            "doc_id": doc_id,
                            "filename": f"{doc_id}.pdf",
                            "page_count": 0,  
                            "processed_date": obj.last_modified.isoformat() if hasattr(obj, 'last_modified') else datetime.now().isoformat(),
                            "status": "processed"
                        })
                    except:
                        pass
            
            return documents
            
    except Exception as e:
        print(f"Error getting documents from Minio: {e}")
        
        try:
            documents = []
            processed_files = glob.glob(os.path.join(DATA_DIR, "*_processed.txt"))
            
            for file_path in processed_files:
                filename = os.path.basename(file_path)
                doc_id = filename.replace("_processed.txt", "")
                
                if os.path.exists(os.path.join(DATA_DIR, f"{doc_id}.pdf")):
                    documents.append({
                        "doc_id": doc_id,
                        "filename": f"{doc_id}.pdf",
                        "page_count": 0, 
                        "processed_date": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                        "status": "processed"
                    })
            
            return documents
        except Exception as e:
            print(f"Error searching documents in filesystem: {e}")
            return []

@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        filename = file.filename
        if not filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")
            
        document_id = f"doc_{int(time.time())}"
        temp_file_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            minio_client = load_models_step.get_minio_client()
            
            with open(temp_file_path, "rb") as file_data:
                file_stat = os.stat(temp_file_path)
                minio_client.put_object(
                    bucket_name=load_models_step.MINIO_BUCKET,
                    object_name=f"documents/{document_id}.pdf",
                    data=file_data,
                    length=file_stat.st_size,
                    content_type="application/pdf"
                )
            
            try:
                response = minio_client.get_object(load_models_step.MINIO_BUCKET, "metadata/document_list.json")
                document_list = json.loads(response.read().decode('utf-8'))
                response.close()
                response.release_conn()
            except:
                document_list = []
            
            document_list.append({
                "doc_id": document_id,
                "filename": filename,
                "original_filename": filename,
                "page_count": 0, 
                "upload_date": datetime.now().isoformat(),
                "processed_date": None,
                "status": "uploaded"
            })
            
            document_list_json = json.dumps(document_list).encode('utf-8')
            minio_client.put_object(
                bucket_name=load_models_step.MINIO_BUCKET,
                object_name="metadata/document_list.json",
                data=BytesIO(document_list_json),
                length=len(document_list_json),
                content_type="application/json"
            )
            
            os.remove(temp_file_path)
            
            return {
                "status": "success",
                "document_id": document_id,
                "message": f"Document {filename} uploaded successfully as {document_id}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error storing in MinIO: {e}")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            raise HTTPException(
                status_code=500, 
                detail=f"Error storing document in MinIO: {str(e)}"
            )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error uploading document: {str(e)}"
        )

@app.post("/process-document")
async def process_document(request: ProcessDocumentRequest):
    doc_id = request.doc_id
    
    try:
        minio_client = load_models_step.get_minio_client()
        
        try:
            minio_client.stat_object(load_models_step.MINIO_BUCKET, f"documents/{doc_id}.pdf")
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        temp_file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
        try:
            minio_client.fget_object(
                bucket_name=load_models_step.MINIO_BUCKET,
                object_name=f"documents/{doc_id}.pdf",
                file_path=temp_file_path
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error downloading document: {str(e)}")
        
        processor = RealEstatePDFProcessor()
        try:
            result = processor.process_pdf(temp_file_path, doc_id)
            
            result_json = json.dumps(result).encode('utf-8')
            minio_client.put_object(
                bucket_name=load_models_step.MINIO_BUCKET,
                object_name=f"embeddings/{doc_id}_embeddings.json",
                data=BytesIO(result_json),
                length=len(result_json),
                content_type="application/json"
            )
            
            try:
                response = minio_client.get_object(load_models_step.MINIO_BUCKET, "metadata/document_list.json")
                document_list = json.loads(response.read().decode('utf-8'))
                response.close()
                response.release_conn()
                
                for doc in document_list:
                    if doc.get("doc_id") == doc_id:
                        doc["status"] = "processed"
                        doc["processed_date"] = datetime.now().isoformat()
                        doc["page_count"] = result.get("page_count", 0)
                        doc["chunks"] = len(result.get("chunks", []))
                
                document_list_json = json.dumps(document_list).encode('utf-8')
                minio_client.put_object(
                    bucket_name=load_models_step.MINIO_BUCKET,
                    object_name="metadata/document_list.json",
                    data=BytesIO(document_list_json),
                    length=len(document_list_json),
                    content_type="application/json"
                )
            except Exception as e:
                print(f"Error updating document metadata: {e}")
            
            os.remove(temp_file_path)
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "message": f"Document {doc_id} processed successfully",
                "metadata": result.get("metadata", {}),
                "chunk_count": len(result.get("chunks", [])),
                "page_count": result.get("page_count", 0)
            }
            
        except Exception as e:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in document processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error in document processing: {str(e)}")

@app.post("/chat")
async def chat(request: ChatRequest):
    messages = [ChatMessage(role=msg.role, content=msg.content) for msg in request.messages]
    doc_ids = request.doc_ids
    
    user_query = next((msg.content for msg in reversed(messages) if msg.role == "user"), None)
    
    if not user_query:
        raise HTTPException(status_code=400, detail="No user query found in messages")
    
    if not doc_ids and is_property_filter_query(user_query):
        try:
            minio_client = load_models_step.get_minio_client()
            response = minio_client.get_object(load_models_step.MINIO_BUCKET, "metadata/document_list.json")
            document_list = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()
            
            doc_ids = [doc.get("doc_id") for doc in document_list 
                      if doc.get("status") == "processed"]
                      
            print(f"Filter query detected: '{user_query}'. Using all available documents: {len(doc_ids)}")
            
        except Exception as e:
            print(f"Error getting document list for filter query: {e}")
    
    relevant_chunks = get_relevant_chunks(user_query, doc_ids)
    
    print(f"Retrieved {len(relevant_chunks)} chunks for query: {user_query[:50]}...")
    
    response = await generate_llm_response(messages, relevant_chunks)
    
    return {"response": response}


@app.get("/document/{doc_id}")
async def get_document_info(doc_id: str):
    try:
        minio_client = load_models_step.get_minio_client()
        
        try:
            response = minio_client.get_object(load_models_step.MINIO_BUCKET, "metadata/document_list.json")
            document_list = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()
            
            doc_info = next((doc for doc in document_list if doc.get("doc_id") == doc_id), None)
            if not doc_info:
                raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        except Exception as e:
            print(f"Error reading document list: {e}")
            raise HTTPException(status_code=404, detail="Document list not accessible")
        
        processed_info = {}
        if doc_info.get("status") == "processed":
            try:
                response = minio_client.get_object(load_models_step.MINIO_BUCKET, f"embeddings/{doc_id}_embeddings.json")
                embeddings_data = json.loads(response.read().decode('utf-8'))
                response.close()
                response.release_conn()
                
                processed_info = {
                    "metadata": embeddings_data.get("metadata", {}),
                    "chunk_count": len(embeddings_data.get("chunks", [])),
                    "page_count": embeddings_data.get("page_count", 0)
                }
            except Exception as e:
                print(f"Error reading document embeddings: {e}")
                processed_info = {"error": "Embeddings could not be retrieved"}
        
        return {
            "doc_id": doc_id,
            "basic_info": doc_info,
            "processed_info": processed_info,
            "status": doc_info.get("status", "unknown")
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error getting document info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting document info: {str(e)}")

@app.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    try:
        minio_client = load_models_step.get_minio_client()
        
        try:
            response = minio_client.get_object(load_models_step.MINIO_BUCKET, "metadata/document_list.json")
            document_list = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()
        except Exception as e:
            print(f"Error reading document list: {e}")
            document_list = []
        
        document_exists = any(doc.get("doc_id") == doc_id for doc in document_list)
        
        if not document_exists:
            raise HTTPException(status_code=404, detail=f"Document with ID {doc_id} not found")
        
        objects_to_remove = [
            f"documents/{doc_id}.pdf",
            f"embeddings/{doc_id}_embeddings.json"
        ]
        
        for obj_name in objects_to_remove:
            try:
                minio_client.remove_object(load_models_step.MINIO_BUCKET, obj_name)
            except Exception as e:
                print(f"Could not delete {obj_name}: {e}")
        
        updated_list = [doc for doc in document_list if doc.get("doc_id") != doc_id]
        document_list_json = json.dumps(updated_list).encode('utf-8')
        
        minio_client.put_object(
            bucket_name=load_models_step.MINIO_BUCKET,
            object_name="metadata/document_list.json",
            data=BytesIO(document_list_json),
            length=len(document_list_json),
            content_type="application/json"
        )
        
        return {
            "status": "success",
            "message": f"Document {doc_id} deleted successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = None
    
    try:
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            "connected_at": datetime.now().isoformat(),
            "messages": []
        }
        
        await websocket.send_json({
            "type": "connection_established",
            "session_id": session_id
        })
        
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "chat_message":
                messages = active_sessions[session_id]["messages"]
                
                user_message = ChatMessage(role="user", content=data.get("content", ""))
                messages.append(user_message)
                
                doc_ids = data.get("doc_ids", [])
                
                relevant_chunks = get_relevant_chunks(user_message.content, doc_ids)
                
                response_text = await generate_llm_response([user_message], relevant_chunks)

                assistant_message = ChatMessage(role="assistant", content=response_text)
                messages.append(assistant_message)
                
                await websocket.send_json({
                    "type": "chat_response",
                    "content": response_text
                })
            
            elif data.get("type") == "list_documents":
                documents = await list_documents()
                await websocket.send_json({
                    "type": "document_list",
                    "documents": documents
                })
                
            elif data.get("type") == "select_documents":
                doc_ids = data.get("doc_ids", [])
                active_sessions[session_id]["selected_docs"] = doc_ids
                
                await websocket.send_json({
                    "type": "documents_selected",
                    "doc_ids": doc_ids,
                    "message": f"Selected {len(doc_ids)} documents for context"
                })
    
    except Exception as e:
        print(f"Error in WebSocket: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except:
            pass
    
    finally:
        if session_id and session_id in active_sessions:
            del active_sessions[session_id]
        try:
            await websocket.close()
        except:
            pass

@app.get("/status")
async def check_status():
    """Check system status"""
    try:
        ollama_status = "unknown"
        minio_status = "unknown"
        
        try:
            import requests
            ollama_response = requests.get(f"{load_models_step.OLLAMA_API_URL}/api/tags", timeout=5)
            ollama_status = "online" if ollama_response.status_code == 200 else "offline"
        except Exception as ollama_error:
            print(f"Error checking Ollama status: {ollama_error}")
            ollama_status = "offline"
        
        try:
            minio_client = load_models_step.get_minio_client()
            minio_client.list_buckets()
            minio_status = "online"
        except Exception as minio_error:
            print(f"Error checking MinIO status: {minio_error}")
            minio_status = "offline"
        
        model_status = "available" if load_models_step.verify_real_estate_model() else "unavailable"
        
        return {
            "ollama": ollama_status,
            "database": minio_status, 
            "api": "online",
            "real_estate_model": model_status,
            "model_name": load_models_step.REAL_ESTATE_MODEL,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "ollama": ollama_status,
            "database": minio_status,
            "api": "online",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }