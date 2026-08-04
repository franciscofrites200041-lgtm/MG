"""API FastAPI para el frontend web.

Endpoints:
    GET  /health                    -> {"status": "ok", "backend": "qdrant", ...}
    POST /search   {query, top_k}   -> lista de hits con path/page/snippet
    POST /read_page {path, page}    -> texto completo de la pagina

CORS habilitado via API_CORS_ORIGINS (coma-separado en .env).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="MG Bot API", version="1.0")

origins = [o.strip() for o in os.getenv("API_CORS_ORIGINS", "").split(",") if o.strip()]
if not origins:
    origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SearchReq(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=20)
    carpeta: str | None = None


class ReadPageReq(BaseModel):
    path: str
    page: int = Field(..., ge=1)


@app.get("/health")
def health():
    from rag.search import stats_index
    return {"status": "ok", **stats_index()}


@app.post("/search")
def search(req: SearchReq):
    """Semantic search directo contra Qdrant (o SQLite si RAG_BACKEND=sqlite).
    Devuelve JSON estructurado en vez del texto formateado del bot."""
    if os.getenv("RAG_BACKEND", "sqlite") == "qdrant":
        from rag import qdrant_backend, reranker
        try:
            qvec = reranker.embed_query(req.query).tolist()
        except Exception as e:
            raise HTTPException(500, f"embed fallo: {e}")
        try:
            hits = qdrant_backend.search(qvec, limit=req.top_k, carpeta=req.carpeta)
        except Exception as e:
            raise HTTPException(500, f"qdrant fallo: {e}")
        return {"hits": hits, "count": len(hits)}

    # Fallback SQLite: devolvemos el texto formateado como esta.
    from rag.search import buscar_en_documentos
    txt = buscar_en_documentos(req.query, top_k=req.top_k)
    return {"text": txt}


@app.post("/read_page")
def read_page(req: ReadPageReq):
    from rag.search import leer_pagina_documento
    text = leer_pagina_documento(req.path, req.page)
    return {"path": req.path, "page": req.page, "text": text}


@app.get("/")
def root():
    return {"service": "mg-bot-api", "docs": "/docs"}
