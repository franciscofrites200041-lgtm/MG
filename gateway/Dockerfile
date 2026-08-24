FROM python:3.11-slim

WORKDIR /app

# Deps del bot (para reusar rag/*)
COPY requirements.txt /tmp/bot-requirements.txt
COPY gateway/requirements.txt /tmp/gw-requirements.txt
RUN pip install --no-cache-dir -r /tmp/bot-requirements.txt -r /tmp/gw-requirements.txt

# Codigo
COPY rag /app/rag
COPY gateway /app/gateway

ENV PYTHONUNBUFFERED=1
ENV RAG_BACKEND=qdrant

EXPOSE 8000
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
