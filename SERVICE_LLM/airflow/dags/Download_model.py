from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import requests
import json
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

default_args = {
   'owner': 'airflow',
   'depends_on_past': False,
   'email_on_failure': False,
   'email_on_retry': False,
   'retries': 2,
   'retry_delay': timedelta(minutes=2),
}

models_dir = '/opt/airflow/models'
llm_model_dir = f'{models_dir}/llama3:8b'
version_file = f'{models_dir}/model_versions.json'
ollama_url = os.environ.get('OLLAMA_API_URL', 'http://ollama:11434')
embedding_endpoint = f"{ollama_url}/api/embeddings"
embedding_model = "nomic-embed-text"

def setup_environment():
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(llm_model_dir, exist_ok=True)
    
    if not os.path.exists(version_file):
        with open(version_file, 'w') as f:
            json.dump({
                'llm_model': {
                    'name': 'zephyr:7b',
                    'version': '1.0.0',
                    'last_update': datetime.now().isoformat()
                },
                'embedding_model': {
                    'name': embedding_model,
                    'version': '1.0.0',
                    'last_update': datetime.now().isoformat()
                }
            }, f, indent=2)
    
    logger.info("Environment setup completed")

def download_llm_model():
    model_name = "llama3:8b"
    
    try:
        tags_response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        tags_data = tags_response.json()
        
        model_exists = any(model['name'] == model_name for model in tags_data.get('models', []))
        
        if model_exists:
            logger.info(f"LLM model {model_name} already exists in Ollama")
        else:
            logger.info(f"Downloading LLM model {model_name}")
            pull_response = requests.post(
                f"{ollama_url}/api/pull",
                json={"name": model_name},
                timeout=1800
            )
            
            if pull_response.status_code == 200:
                logger.info(f"LLM model {model_name} downloaded successfully")
            else:
                logger.error(f"Error downloading model: {pull_response.status_code} - {pull_response.text}")
                raise Exception(f"Failed to pull model: {pull_response.text}")
    
    except Exception as e:
        logger.error(f"Error checking/downloading model: {str(e)}")
        raise
    
    with open(version_file, 'r') as f:
        versions = json.load(f)

    versions['llm_model'] = {
        'name': 'llama3:8b',
        'version': '1.0.0',
        'last_update': datetime.now().isoformat()
    }

    with open(version_file, 'w') as f:
        json.dump(versions, f, indent=2)

    logger.info("LLM model downloaded and version updated successfully")

def download_embedding_model():
    logger.info(f"Checking availability of embedding model: {embedding_model}")
    
    try:
        response = requests.post(
            embedding_endpoint,
            json={"model": embedding_model, "prompt": "test"},
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Embedding model {embedding_model} is already available")
        else:
            logger.warning(f"Embedding model is not available: {response.status_code}")
            
            logger.info(f"Downloading embedding model {embedding_model}")
            pull_response = requests.post(
                f"{ollama_url}/api/pull",
                json={"name": embedding_model},
                timeout=600
            )
            
            if pull_response.status_code == 200:
                logger.info(f"Download of embedding model {embedding_model} completed")
            else:
                logger.error(f"Error downloading model: {pull_response.status_code} - {pull_response.text}")
                raise Exception(f"Failed to pull model: {pull_response.text}")
        
        with open(version_file, 'r') as f:
            versions = json.load(f)
        
        versions['embedding_model'] = {
            'name': embedding_model,
            'version': '1.0.0',
            'last_update': datetime.now().isoformat()
        }
        
        with open(version_file, 'w') as f:
            json.dump(versions, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error checking/downloading embedding model: {str(e)}")
        raise

with DAG(
   'model_downloader_dag',
   default_args=default_args,
   description='DAG for downloading LLM and embedding models',
   schedule_interval=None,
   start_date=datetime(2023, 1, 1),
   catchup=False,
   tags=['llm', 'embeddings', 'download'],
) as dag:
   
   setup_task = PythonOperator(
       task_id='setup_environment',
       python_callable=setup_environment,
   )
   
   llm_download_task = PythonOperator(
       task_id='download_llm_model',
       python_callable=download_llm_model,
   )
   
   embedding_download_task = PythonOperator(
       task_id='download_embedding_model',
       python_callable=download_embedding_model,
   )
   
   setup_task >> [llm_download_task, embedding_download_task]