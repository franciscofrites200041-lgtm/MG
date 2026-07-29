FROM python:3.11-slim

# antiword para .doc viejos; otras deps chicas del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
      antiword \
      ca-certificates \
      curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# El index.db y memory.db viven en /data (bind-mount desde el NAS).
VOLUME ["/data"]
ENV INDEX_DB_PATH=/data/index.db \
    MEMORY_DB_PATH=/data/memory.db \
    DOCS_OUT_DIR=/data/escritos_generados \
    REINDEX_LOG=/data/logs/reindex.log

CMD ["python", "main.py"]
