from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.hooks.mysql import MySqlHook
import pandas as pd
import os
import requests
import json
import re
import unicodedata
from io import BytesIO
import minio
from minio import Minio
from minio.error import S3Error
import numpy as np
import PyPDF2 
import pdfplumber  
import hashlib



default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def clean_text(text):
    """
    Realiza limpieza básica del texto:
    - Convierte a minúsculas
    - Elimina caracteres especiales
    - Normaliza caracteres unicode
    - Elimina espacios extra
    """
    if not text:
        return ""
    
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    
    text = re.sub(r'[^\w\s.,;:!?-]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def check_documents_table():
    try:
        mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
        
        check_db_query = """
        SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = 'airflow_db'
        """
        
        result = mysql_hook.get_first(check_db_query)
        db_exists = result is not None
        
        if not db_exists:
            print("Creando base de datos 'airflow_db'...")
            create_db_query = """
            CREATE DATABASE IF NOT EXISTS airflow_db
            CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
            """
            mysql_hook.run(create_db_query)
            print("Base de datos 'airflow_db' creada exitosamente.")
        
        mysql_hook.run("USE airflow_db;")
        
        check_table_query = """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'airflow_db'
        AND table_name = 'documents';
        """
        
        result = mysql_hook.get_first(check_table_query)
        table_exists = result[0] > 0
        
        if table_exists:
            print("La tabla 'documents' ya existe.")
            
            columns_to_check = [
                {
                    'name': 'file_path',
                    'definition': 'VARCHAR(255) NULL'
                },
                {
                    'name': 'page_count',
                    'definition': 'INT DEFAULT 1'
                },
                {
                    'name': 'file_hash',
                    'definition': 'VARCHAR(64) NULL'
                },
                {
                    'name': 'last_updated',
                    'definition': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP'
                }
            ]
            
            for column in columns_to_check:
                check_column_query = f"""
                SELECT COUNT(*)
                FROM information_schema.columns
                WHERE table_schema = 'airflow_db'
                AND table_name = 'documents'
                AND column_name = '{column['name']}';
                """
                
                result = mysql_hook.get_first(check_column_query)
                column_exists = result[0] > 0
                
                if not column_exists:
                    print(f"Añadiendo columna '{column['name']}' a la tabla documents...")
                    add_column_query = f"""
                    ALTER TABLE documents 
                    ADD COLUMN {column['name']} {column['definition']};
                    """
                    mysql_hook.run(add_column_query)
                    print(f"Columna '{column['name']}' añadida exitosamente.")
            
            check_index_query = """
            SELECT COUNT(*)
            FROM information_schema.statistics
            WHERE table_schema = 'airflow_db'
            AND table_name = 'documents'
            AND index_name = 'idx_title';
            """
            
            result = mysql_hook.get_first(check_index_query)
            index_exists = result[0] > 0
            
            if not index_exists:
                print("Añadiendo índice para la columna 'title'...")
                add_index_query = """
                CREATE INDEX idx_title ON documents(title);
                """
                mysql_hook.run(add_index_query)
                print("Índice para 'title' añadido exitosamente.")
                
            return True
        else:
            print("Creando tabla 'documents'...")
            
            create_table_query = """
            CREATE TABLE documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                content LONGTEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE,
                processed_at TIMESTAMP NULL,
                file_path VARCHAR(255) NULL,
                page_count INT DEFAULT 1,
                file_hash VARCHAR(64) NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_title (title)
            ) ENGINE=InnoDB CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
            """
            
            mysql_hook.run(create_table_query)
            print("Tabla 'documents' creada exitosamente.")
            return True
        
    except Exception as e:
        print(f"Error al verificar/crear la tabla documents: {e}")
        raise

def extract_pdf_text(pdf_path):
    text = ""
    num_pages = 0
    
    try:
        with open(pdf_path, 'rb') as file:
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        print(f"No se pudo extraer texto de la página {page_num+1}")
            except AttributeError:
                file.seek(0)
                pdf_reader = PyPDF2.PdfFileReader(file)
                num_pages = pdf_reader.getNumPages()
                
                for page_num in range(num_pages):
                    page = pdf_reader.getPage(page_num)
                    page_text = page.extractText()
                    if page_text:
                        text += page_text + "\n"
                    else:
                        print(f"No se pudo extraer texto de la página {page_num+1}")
        
        if text.strip():
            print(f"Extracción exitosa del PDF {pdf_path} usando PyPDF2")
            return text, num_pages
    except Exception as e:
        print(f"Error en PyPDF2 para {pdf_path}: {e}")
    
    if pdfplumber is not None:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                num_pages = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if text.strip():
                print(f"Extracción exitosa del PDF {pdf_path} usando pdfplumber")
                return text, num_pages
        except Exception as e:
            print(f"Error en pdfplumber para {pdf_path}: {e}")
    
    print(f"No se pudo extraer texto del PDF {pdf_path} con ningún método")
    return f"[Documento sin texto extraíble: {os.path.basename(pdf_path)}]", max(1, num_pages)

def calculate_file_hash(file_path):    
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def scan_and_load_pdfs():
    pdf_directory = '/opt/airflow/data'
    print(f"Escaneando directorio: {pdf_directory}")
    
    check_documents_table()
    
    mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
    
    pdf_files = []
    for root, dirs, files in os.walk(pdf_directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                pdf_files.append(pdf_path)
    
    print(f"Encontrados {len(pdf_files)} archivos PDF")
    
    if not pdf_files:
        print("No se encontraron archivos PDF para procesar")
        return []
    
    docs_added_or_updated = []
    
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        file_hash = calculate_file_hash(pdf_path)
        
        check_query = """
        SELECT id, file_hash FROM documents 
        WHERE title = %s
        """
        
        result = mysql_hook.get_first(check_query, parameters=[file_name])
        existing_doc_id = None
        existing_hash = None
        
        if result:
            existing_doc_id, existing_hash = result
            
        if existing_doc_id is None or existing_hash != file_hash:
            content, page_count = extract_pdf_text(pdf_path)
            
            if not content:
                print(f"No se pudo extraer texto del archivo {file_name}")
                continue
            
            try:
                if existing_doc_id:
                    update_query = """
                    UPDATE documents 
                    SET content = %s, 
                        processed = 0, 
                        processed_at = NULL, 
                        file_path = %s, 
                        page_count = %s,
                        file_hash = %s
                    WHERE id = %s
                    """
                    
                    mysql_hook.run(update_query, parameters=[content, pdf_path, page_count, file_hash, existing_doc_id])
                    print(f"Actualizado documento existente {file_name} (ID: {existing_doc_id}) en la base de datos")
                    docs_added_or_updated.append(existing_doc_id)
                else:
                    insert_query = """
                    INSERT INTO documents (title, content, processed, file_path, page_count, file_hash) 
                    VALUES (%s, %s, 0, %s, %s, %s)
                    """
                    
                    mysql_hook.run(insert_query, parameters=[file_name, content, pdf_path, page_count, file_hash])
                    
                    get_id_query = "SELECT LAST_INSERT_ID()"
                    doc_id = mysql_hook.get_first(get_id_query)[0]
                    print(f"Añadido nuevo documento {file_name} (ID: {doc_id}) a la base de datos")
                    docs_added_or_updated.append(doc_id)
            
            except Exception as e:
                print(f"Error al insertar/actualizar documento {file_name}: {e}")
        else:
            print(f"El archivo {file_name} no ha cambiado desde la última vez (hash: {file_hash})")
    
    return docs_added_or_updated

def clean_duplicate_documents():
    """
    Limpia documentos duplicados de la base de datos basándose en el nombre del archivo
    """
    try:
        check_documents_table()
        
        mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
        
        find_duplicates_query = """
        SELECT title, COUNT(*) as count, GROUP_CONCAT(id ORDER BY id) as ids
        FROM documents
        GROUP BY title
        HAVING COUNT(*) > 1
        """
        
        duplicates = mysql_hook.get_records(find_duplicates_query)
        
        if not duplicates:
            print("No se encontraron documentos duplicados")
            return
        
        print(f"Encontrados {len(duplicates)} títulos de documentos con duplicados")
        
        for row in duplicates:
            title, count, ids = row
            id_list = ids.split(',')
            keep_id = max([int(id) for id in id_list])
            delete_ids = [id for id in id_list if int(id) != keep_id]
            
            if delete_ids:
                delete_ids_str = ','.join(delete_ids)
                delete_query = f"""
                DELETE FROM documents
                WHERE id IN ({delete_ids_str})
                """
                
                mysql_hook.run(delete_query)
                print(f"Eliminados {len(delete_ids)} duplicados para el documento '{title}', manteniendo ID {keep_id}")
        
    except Exception as e:
        print(f"Error al limpiar documentos duplicados: {e}")
        raise

def get_documents(ti):
    """
    Recupera documentos desde la base de datos MySQL
    """
    scan_and_load_pdfs()
    
    try:
        mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
        query = """
        SELECT id, title, content, created_at, file_path, page_count 
        FROM documents 
        WHERE processed = 0
        LIMIT 100
        """
        
        documents_df = mysql_hook.get_pandas_df(query)
        
        if documents_df.empty:
            print("No hay nuevos documentos para procesar")
            return []
        
        doc_ids = documents_df['id'].tolist()
        
        documents_df['created_at'] = documents_df['created_at'].astype(str)       
        documents = documents_df.to_dict('records')
        
        print(f"Recuperados {len(documents)} documentos para procesar")
        return {'documents': documents, 'doc_ids': doc_ids}
        
    except Exception as e:
        print(f"Error al recuperar documentos: {e}")
        raise

def process_documents(ti):
    """
    Limpia el texto de cada documento
    """
    task_data = ti.xcom_pull(task_ids='get_documents_task')
    
    if not task_data or not task_data.get('documents'):
        print("No hay documentos para procesar")
        return []
    
    documents = task_data['documents']
    processed_docs = []
    
    for doc in documents:
        clean_content = clean_text(doc['content'])      
        doc['clean_content'] = clean_content
        processed_docs.append(doc)
    
    print(f"Procesados {len(processed_docs)} documentos")
    
    return {'processed_docs': processed_docs, 'doc_ids': task_data['doc_ids']}

def generate_embeddings(ti):
    """
    Genera embeddings para cada documento usando el modelo de Ollama
    """
    task_data = ti.xcom_pull(task_ids='process_documents_task')
    
    if not task_data or not task_data.get('processed_docs'):
        print("No hay documentos procesados para generar embeddings")
        return []
    
    processed_docs = task_data['processed_docs']
    docs_with_embeddings = []
    
    ollama_url = os.environ.get('OLLAMA_API_URL', 'http://ollama:11434')
    embedding_endpoint = f"{ollama_url}/api/embeddings"
    
    embedding_model = "nomic-embed-text"
    
    for doc in processed_docs:
        try:
            payload = {
                "model": embedding_model,
                "prompt": doc['clean_content']
            }
            
            response = requests.post(embedding_endpoint, json=payload)
            
            if response.status_code == 200:
                embedding_data = response.json()
                embedding = embedding_data.get('embedding', [])
                
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                
                doc['embedding'] = embedding
                
                if 'created_at' in doc and not isinstance(doc['created_at'], str):
                    doc['created_at'] = str(doc['created_at'])
                
                docs_with_embeddings.append(doc)
                print(f"Generado embedding para documento {doc['id']}")
            else:
                print(f"Error al generar embedding para documento {doc['id']}: {response.text}")
        
        except Exception as e:
            print(f"Excepción al generar embedding para documento {doc['id']}: {e}")
    
    print(f"Generados embeddings para {len(docs_with_embeddings)} documentos")
    
    return {'docs_with_embeddings': docs_with_embeddings, 'doc_ids': task_data['doc_ids']}

def save_to_minio(ti):
    task_data = ti.xcom_pull(task_ids='generate_embeddings_task')
    
    if not task_data or not task_data.get('docs_with_embeddings'):
        print("No hay documentos con embeddings para guardar")
        return
    
    docs_with_embeddings = task_data['docs_with_embeddings']
    doc_ids = task_data['doc_ids']
    
    minio_endpoint = os.environ.get('MINIO_ENDPOINT', 'minio:9000')
    minio_access_key = os.environ.get('MINIO_ACCESS_KEY', 'minioadmin')
    minio_secret_key = os.environ.get('MINIO_SECRET_KEY', 'minioadmin')
    minio_bucket = 'llm-data'
    
    try:
        minio_client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False
        )
        
        if not minio_client.bucket_exists(minio_bucket):
            minio_client.make_bucket(minio_bucket)
            print(f"Bucket {minio_bucket} creado")
        
        try:
            metadata_path = "metadata/document_list.json"
            existing_docs = []
            
            try:
                response = minio_client.get_object(minio_bucket, metadata_path)
                existing_docs = json.loads(response.read().decode('utf-8'))
                response.close()
                response.release_conn()
            except Exception as e:
                print(f"No se encontró lista de documentos previa, creando nueva: {e}")
                
            for doc in docs_with_embeddings:
                try:
                    doc_id = str(doc['id'])
                    chunks = []
                    
                    chunks.append({
                        "chunk_id": 0,
                        "text": doc['clean_content'],
                        "embedding": doc['embedding']
                    })
                    
                    chunks_json = json.dumps(chunks).encode('utf-8')
                    embeddings_path = f"embeddings/{doc_id}_embeddings.json"
                    try:
                        minio_client.remove_object(minio_bucket, embeddings_path)
                        print(f"Eliminado embedding anterior para documento {doc_id}")
                    except:
                        pass
                    
                    minio_client.put_object(
                        bucket_name=minio_bucket,
                        object_name=embeddings_path,
                        data=BytesIO(chunks_json),
                        length=len(chunks_json),
                        content_type='application/json'
                    )
                    
                    processed_text = doc['clean_content'].encode('utf-8')
                    processed_path = f"processed/{doc_id}_processed.txt"
                    
                    try:
                        minio_client.remove_object(minio_bucket, processed_path)
                        print(f"Eliminado texto procesado anterior para documento {doc_id}")
                    except:
                        pass
                    
                    minio_client.put_object(
                        bucket_name=minio_bucket,
                        object_name=processed_path,
                        data=BytesIO(processed_text),
                        length=len(processed_text),
                        content_type='text/plain'
                    )
                    
                    original_filename = os.path.basename(doc.get('file_path', ''))
                    if not original_filename:
                        original_filename = f"{doc['title'].replace(' ', '_')}_{doc_id}.pdf"
                    
                    doc_metadata = {
                        "doc_id": doc_id,
                        "filename": original_filename,
                        "page_count": doc.get('page_count', 1),
                        "processed_date": datetime.now().isoformat()
                    }
                    
                    existing_docs = [d for d in existing_docs if d.get('filename') != original_filename]
                    
                    existing_docs.append(doc_metadata)
                    
                    print(f"Documento {doc_id} ({original_filename}) actualizado en MinIO")
                    
                except Exception as e:
                    print(f"Error al guardar documento {doc['id']} en MinIO: {e}")
            
            doc_list_json = json.dumps(existing_docs).encode('utf-8')
            minio_client.put_object(
                bucket_name=minio_bucket,
                object_name=metadata_path,
                data=BytesIO(doc_list_json),
                length=len(doc_list_json),
                content_type='application/json'
            )
            
            try:
                models_metadata_path = "metadata/models_metadata.json"
                
                try:
                    ollama_url = os.environ.get('OLLAMA_API_URL', 'http://ollama:11434')
                    response = requests.get(f"{ollama_url}/api/tags")
                    
                    if response.status_code == 200:
                        models_data = {
                            "models": [],
                            "current_llm": "llama3"
                        }
                        
                        for model in response.json().get("models", []):
                            models_data["models"].append({
                                "name": model.get("name", "unknown"),
                                "size": model.get("size", 0),
                                "modified": model.get("modified", ""),
                                "status": "downloaded"
                            })
                        
                        models_json = json.dumps(models_data).encode('utf-8')
                        
                        minio_client.put_object(
                            bucket_name=minio_bucket,
                            object_name=models_metadata_path,
                            data=BytesIO(models_json),
                            length=len(models_json),
                            content_type='application/json'
                        )
                        
                        print("Metadata de modelos guardada en MinIO")
                    
                except Exception as e:
                    print(f"Error al obtener/guardar metadata de modelos: {e}")
            
            except Exception as e:
                print(f"Error al guardar metadata de modelos: {e}")
        
        except Exception as e:
            print(f"Error al crear estructura de datos en MinIO: {e}")
            
        update_document_status(doc_ids)
        
    except S3Error as e:
        print(f"Error de MinIO: {e}")
        raise
    except Exception as e:
        print(f"Error al guardar en MinIO: {e}")
        raise

def update_document_status(doc_ids):
    if not doc_ids:
        print("No hay IDs de documentos para actualizar")
        return
    
    try:
        mysql_hook = MySqlHook(mysql_conn_id='mysql_default')
        ids_str = ','.join([str(doc_id) for doc_id in doc_ids])
        
        update_query = f"""
        UPDATE documents 
        SET processed = 1, 
            processed_at = NOW() 
        WHERE id IN ({ids_str})
        """
        
        mysql_hook.run(update_query)
        print(f"Actualizados {len(doc_ids)} documentos en la base de datos")
        
    except Exception as e:
        print(f"Error al actualizar estado de documentos: {e}")
        raise

with DAG(
    'pdf_processing_embedding_dag',
    default_args=default_args,
    description='DAG para procesar PDFs, generar embeddings y guardar en MinIO',
    schedule_interval=None,
    start_date=datetime(2025, 4, 14),
    catchup=False,
    tags=['pdf', 'nlp', 'embeddings', 'minio'],
) as dag:
    
    clean_duplicates_task = PythonOperator(
        task_id='clean_duplicates_task',
        python_callable=clean_duplicate_documents,
    )
    
    get_documents_task = PythonOperator(
        task_id='get_documents_task',
        python_callable=get_documents,
    )
    
    process_documents_task = PythonOperator(
        task_id='process_documents_task',
        python_callable=process_documents,
    )
    
    generate_embeddings_task = PythonOperator(
        task_id='generate_embeddings_task',
        python_callable=generate_embeddings,
    )
    
    save_to_minio_task = PythonOperator(
        task_id='save_to_minio_task',
        python_callable=save_to_minio,
    )
    
    clean_duplicates_task >> get_documents_task >> process_documents_task >> generate_embeddings_task >> save_to_minio_task