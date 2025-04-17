FROM apache/airflow:2.6.0-python3.9

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    libffi-dev \
    software-properties-common \
    python3-dev \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN mkdir -p /opt/airflow/huggingface_cache && \
    chown -R airflow:0 /opt/airflow/huggingface_cache && \
    chmod -R 777 /opt/airflow/huggingface_cache

USER airflow
ENV PATH="/home/airflow/.cargo/bin:${PATH}"
ENV TRANSFORMERS_CACHE="/opt/airflow/huggingface_cache"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir pandas==1.5.3 && \
    pip install --no-cache-dir scikit-learn==1.0.2

RUN pip install --no-cache-dir torch==2.2.0 accelerate>=0.28.0

RUN pip install --no-cache-dir "transformers>=4.35.0"

RUN pip install --no-cache-dir joblib apache-airflow-providers-mysql && \
    pip install --no-cache-dir pdfplumber minio contractions nltk PyPDF2 mysql-connector-python && \
    python -m nltk.downloader punkt

RUN pip install --no-cache-dir scipy peft==0.5.0 bitsandbytes==0.45.5

RUN python -c "import sys; import transformers; import torch; import numpy; print(f'Python: {sys.version}\nTransformers: {transformers.__version__}\nPyTorch: {torch.__version__}\nNumPy: {numpy.__version__}')"