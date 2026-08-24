FROM python:3.11-slim

WORKDIR /app

# antiword (para .doc) + rclone (para on-demand NAS fetching)
RUN apt-get update && apt-get install -y --no-install-recommends antiword curl unzip && \
    curl -s https://rclone.org/install.sh | bash && \
    apt-get remove -y curl unzip && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Deps del gateway (autosuficiente: FastAPI + Gemini + Qdrant + embed)
COPY gateway/requirements.txt /tmp/gw-requirements.txt
# python-docx + pymupdf para que bg_worker pueda extraer
RUN pip install --no-cache-dir -r /tmp/gw-requirements.txt python-docx==1.1.2 pymupdf==1.24.9

# Codigo
COPY rag /app/rag
COPY gateway /app/gateway
COPY scripts /app/scripts

ENV PYTHONUNBUFFERED=1
ENV RAG_BACKEND=qdrant

EXPOSE 8000
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
