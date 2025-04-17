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
TEMP_DOCUMENTS = {}  # Formato: {session_id: [doc_id1, doc_id2, ...]}
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

async def cleanup_temp_documents(session_id=None):
    try:
        print(f"Starting cleanup for session: {session_id if session_id else 'ALL'}")
        minio_client = load_models_step.get_minio_client()
        docs_to_clean = []
        
        if session_id and session_id in TEMP_DOCUMENTS:
            docs_to_clean = TEMP_DOCUMENTS[session_id]
            print(f"Found {len(docs_to_clean)} temporary documents for session {session_id}")
            del TEMP_DOCUMENTS[session_id]
        else:
            for sess_id, sess_docs in TEMP_DOCUMENTS.items():
                docs_to_clean.extend(sess_docs)
                print(f"Adding {len(sess_docs)} documents from session {sess_id} to cleanup")
            print(f"Total of {len(docs_to_clean)} temporary documents found across all sessions")
            TEMP_DOCUMENTS.clear()
        
        if not docs_to_clean:
            print("No temporary documents to clean")
            return
            
        try:
            response = minio_client.get_object(load_models_step.MINIO_BUCKET, "metadata/document_list.json")
            document_list = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()
        except Exception as e:
            print(f"Error reading document list during cleanup: {e}")
            return
            
        initial_count = len(document_list)
        updated_list = [doc for doc in document_list if doc.get("doc_id") not in docs_to_clean]
        removed_count = initial_count - len(updated_list)
        print(f"Removing {removed_count} documents from the document list")
        
        for doc_id in docs_to_clean:
            try:
                objects_to_remove = [
                    f"documents/{doc_id}.pdf",
                    f"embeddings/{doc_id}_embeddings.json"
                ]
                
                for obj_name in objects_to_remove:
                    try:
                        minio_client.remove_object(load_models_step.MINIO_BUCKET, obj_name)
                        print(f"Deleted object: {obj_name}")
                    except Exception as e:
                        print(f"Could not delete {obj_name}: {e}")
                        
                print(f"Cleaned up temporary document: {doc_id}")
            except Exception as e:
                print(f"Error removing document {doc_id}: {e}")
        
        document_list_json = json.dumps(updated_list).encode('utf-8')
        
        minio_client.put_object(
            bucket_name=load_models_step.MINIO_BUCKET,
            object_name="metadata/document_list.json",
            data=BytesIO(document_list_json),
            length=len(document_list_json),
            content_type="application/json"
        )
        print(f"Updated document list after cleaning {removed_count} documents")
        
    except Exception as e:
        print(f"Critical error during cleanup of temporary documents: {e}")
        

@app.on_event("startup")
async def startup_tasks():
    model_manager.start_continuous_checking()

@app.on_event("shutdown")
async def shutdown_tasks():
    print("Shutting down API, cleaning up temporary documents...")
    await cleanup_temp_documents()
    model_manager.stop_continuous_checking()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("/app/Front/templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/documents")
async def list_documents():
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

@app.post("/cleanup-session")
async def cleanup_session(request: Request):
    data = await request.json()
    session_id = data.get("session_id")
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    
    await cleanup_temp_documents(session_id)
    
    return {
        "status": "success",
        "message": f"Cleaned up session: {session_id}",
        "timestamp": datetime.now().isoformat()
    }
    
@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), session_id: str = Form(None)):
    try:
        filename = file.filename
        if not filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")
            
        document_id = f"doc_{int(time.time())}"
        temp_file_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        needs_ocr = False
        try:
            processor = RealEstatePDFProcessor()
            has_text = processor.check_pdf_has_text(temp_file_path)
            if not has_text:
                needs_ocr = True
                print(f"PDF {document_id} requires OCR processing")
        except Exception as e:
            print(f"Error checking PDF text content: {e}")
        
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
            
            is_temporary = session_id is not None
            
            document_list.append({
                "doc_id": document_id,
                "filename": filename,
                "original_filename": filename,
                "page_count": 0, 
                "upload_date": datetime.now().isoformat(),
                "processed_date": None,
                "status": "uploaded",
                "needs_ocr": needs_ocr,
                "is_temporary": is_temporary
            })
            
            if is_temporary:
                if session_id not in TEMP_DOCUMENTS:
                    TEMP_DOCUMENTS[session_id] = []
                TEMP_DOCUMENTS[session_id].append(document_id)
                print(f"Document {document_id} marked as temporary for session {session_id}")
            
            document_list_json = json.dumps(document_list).encode('utf-8')
            minio_client.put_object(
                bucket_name=load_models_step.MINIO_BUCKET,
                object_name="metadata/document_list.json",
                data=BytesIO(document_list_json),
                length=len(document_list_json),
                content_type="application/json"
            )
            
            processor = RealEstatePDFProcessor()
            try:
                print(f"Processing uploaded document {document_id}...")
                result = processor.process_pdf(temp_file_path, document_id)
                
                result_json = json.dumps(result).encode('utf-8')
                minio_client.put_object(
                    bucket_name=load_models_step.MINIO_BUCKET,
                    object_name=f"embeddings/{document_id}_embeddings.json",
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
                        if doc.get("doc_id") == document_id:
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
                    print(f"Error updating document metadata after processing: {e}")
                
                processing_status = "success"
                processing_message = f"Document processed successfully with {len(result.get('chunks', []))} chunks"
            except Exception as e:
                print(f"Error processing uploaded document: {e}")
                processing_status = "error"
                processing_message = f"Document uploaded but processing failed: {str(e)}"
            
            os.remove(temp_file_path)
            
            return {
                "status": "success",
                "document_id": document_id,
                "message": f"Document {filename} uploaded successfully as {document_id}. {processing_message}",
                "timestamp": datetime.now().isoformat(),
                "processing_status": processing_status,
                "is_temporary": is_temporary
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
    messages = request.messages
    doc_ids = request.doc_ids

    formatted_messages = [ChatMessage(role=msg.role, content=msg.content) for msg in messages]

    session_id = None
    for msg in messages:
        if hasattr(msg, 'metadata') and msg.metadata and 'session_id' in msg.metadata:
            session_id = msg.metadata['session_id']
            break

    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in active_sessions:
        active_sessions[session_id] = SessionData(session_id)

    session = active_sessions[session_id]

    if len(formatted_messages) > 1 and len(session.messages) > 0:
        if len(formatted_messages) >= len(session.messages):
            session.messages = formatted_messages
        else:
            last_user_msg = next((msg for msg in reversed(formatted_messages) if msg.role == "user"), None)
            if last_user_msg:
                session.add_message("user", last_user_msg.content)
    elif len(formatted_messages) == 1 and formatted_messages[0].role == "user":
        session.add_message("user", formatted_messages[0].content)

    user_query = next((msg.content for msg in reversed(session.messages) if msg.role == "user"), None)

    if not user_query:
        raise HTTPException(status_code=400, detail="No user query found in messages")

    if doc_ids:
        session.selected_docs = doc_ids
    else:
        if session.selected_docs:
            doc_ids = session.selected_docs
        else:
            try:
                minio_client = load_models_step.get_minio_client()
                response = minio_client.get_object(load_models_step.MINIO_BUCKET, "metadata/document_list.json")
                document_list = json.loads(response.read().decode('utf-8'))
                response.close()
                response.release_conn()

                document_list.sort(key=lambda x: x.get("upload_date", ""), reverse=True)

                doc_ids = [doc.get("doc_id") for doc in document_list if doc.get("status") == "processed"]
            except Exception as e:
                print(f"Error getting document list: {e}")

    mentioned_docs = []
    try:
        minio_client = load_models_step.get_minio_client()
        response = minio_client.get_object(load_models_step.MINIO_BUCKET, "metadata/document_list.json")
        document_list = json.loads(response.read().decode('utf-8'))
        response.close()
        response.release_conn()

        for doc in document_list:
            filename = doc.get("original_filename", "")
            doc_id = doc.get("doc_id", "")
            if filename and filename.lower() in user_query.lower():
                mentioned_docs.append(doc_id)
                print(f"User mentioned document: {filename} (ID: {doc_id})")

        if mentioned_docs:
            doc_ids = list(set(doc_ids + mentioned_docs))
    except Exception as e:
        print(f"Error searching for document mentions: {e}")

    relevant_chunks = get_relevant_chunks(user_query, doc_ids)
    print(f"Retrieved {len(relevant_chunks)} chunks for query: {user_query[:50]}...")

    response_text = await generate_llm_response(session.messages, relevant_chunks)

    session.add_message("assistant", response_text)
    session.last_response = response_text

    return {
        "response": response_text,
        "session_id": session_id,
        "message_count": len(session.messages)
    }


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

active_sessions = {}  

class SessionData:
    def __init__(self, session_id):
        self.session_id = session_id
        self.connected_at = datetime.now().isoformat()
        self.messages = []  # Lista de ChatMessage
        self.selected_docs = []
        self.last_prompt = None
        self.last_response = None
    
    def add_message(self, role, content):
        message = ChatMessage(role=role, content=content)
        self.messages.append(message)
        return message
    
    def get_conversation_history(self, last_n=None):
        if last_n is not None and last_n > 0:
            return self.messages[-last_n:]
        return self.messages
    
    def clear_history(self):
        self.messages = []
        self.last_prompt = None
        self.last_response = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = None
    
    try:
        session_id = str(uuid.uuid4())
        session = SessionData(session_id)
        active_sessions[session_id] = session
        
        await websocket.send_json({
            "type": "connection_established",
            "session_id": session_id
        })
        
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "chat_message":
                user_content = data.get("content", "")
                
                user_message = session.add_message("user", user_content)
                
                doc_ids = data.get("doc_ids", [])
                if doc_ids:
                    session.selected_docs = doc_ids
                
                if not doc_ids:
                    try:
                        documents = await list_documents()
                        doc_ids = [doc.get("doc_id") for doc in documents 
                                 if doc.get("status") == "processed"]
                        print(f"WebSocket: No documents selected. Using all available: {len(doc_ids)}")
                    except Exception as e:
                        print(f"Error getting all documents in WebSocket: {e}")
                
                relevant_chunks = get_relevant_chunks(user_content, doc_ids)
                
                all_messages = session.get_conversation_history()
                
                response_text = await generate_llm_response(all_messages, relevant_chunks)
                
                assistant_message = session.add_message("assistant", response_text)
                
                session.last_prompt = user_content
                session.last_response = response_text
                
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
                session.selected_docs = doc_ids
                
                await websocket.send_json({
                    "type": "documents_selected",
                    "doc_ids": doc_ids,
                    "message": f"Selected {len(doc_ids)} documents for context"
                })
                
            elif data.get("type") == "new_chat":
                await cleanup_temp_documents(session_id)
                
                session.clear_history()
                
                await websocket.send_json({
                    "type": "chat_reset",
                    "message": "Chat has been reset and temporary documents cleaned up"
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
        if session_id:
            await cleanup_temp_documents(session_id)
            
        if session_id and session_id in active_sessions:
            del active_sessions[session_id]
        try:
            await websocket.close()
        except:
            pass
        
@app.post("/delete-documents")
async def delete_multiple_documents(request: List[str]):
    try:
        doc_ids = request
        minio_client = load_models_step.get_minio_client()
        
        try:
            response = minio_client.get_object(load_models_step.MINIO_BUCKET, "metadata/document_list.json")
            document_list = json.loads(response.read().decode('utf-8'))
            response.close()
            response.release_conn()
        except Exception as e:
            print(f"Error reading document list: {e}")
            document_list = []
        
        deleted_docs = []
        failed_docs = []
        
        for doc_id in doc_ids:
            document_exists = any(doc.get("doc_id") == doc_id for doc in document_list)
            
            if not document_exists:
                failed_docs.append({"doc_id": doc_id, "reason": "Document not found"})
                continue
                
            try:
                objects_to_remove = [
                    f"documents/{doc_id}.pdf",
                    f"embeddings/{doc_id}_embeddings.json"
                ]
                
                for obj_name in objects_to_remove:
                    try:
                        minio_client.remove_object(load_models_step.MINIO_BUCKET, obj_name)
                    except Exception as e:
                        print(f"Could not delete {obj_name}: {e}")
                
                deleted_docs.append(doc_id)
            except Exception as e:
                failed_docs.append({"doc_id": doc_id, "reason": str(e)})
        
        updated_list = [doc for doc in document_list if doc.get("doc_id") not in deleted_docs]
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
            "deleted": deleted_docs,
            "failed": failed_docs,
            "timestamp": datetime.now().isoformat()
        }
            
    except Exception as e:
        print(f"Error deleting documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting documents: {str(e)}")
    
@app.get("/status")
async def check_status():
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