FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      antiword ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Reusa el requirements del bot (comparte deps: qdrant-client, sentence-transformers).
COPY requirements.txt ./bot-requirements.txt
COPY api/requirements.txt ./api-requirements.txt
RUN pip install --no-cache-dir -r ./bot-requirements.txt -r ./api-requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
