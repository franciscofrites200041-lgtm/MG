"""RAG Gateway: expone endpoint OpenAI-compat que Open WebUI consume.

Flujo por request:
    1. Recibe chat completions request (mensajes + config)
    2. Toma el ultimo user message = query
    3. Embed + Qdrant search -> top-k chunks
    4. Arma prompt: system (con contexto RAG) + historial (compactado si excede) + query
    5. Llama OpenAI con streaming
    6. Devuelve chunks OpenAI-formatted al frontend

Auto-compactacion:
    Si el total_tokens estimado del historial supera COMPACT_THRESHOLD,
    los mensajes anteriores al ultimo turno se reemplazan por un resumen.

Endpoints:
    GET  /v1/models                    -> lista mock (Open WebUI necesita)
    POST /v1/chat/completions          -> streaming SSE compatible OpenAI
    GET  /health                       -> estado
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("RAG_BACKEND", "qdrant")

logger = logging.getLogger("gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_SUMMARIZE_MODEL = os.getenv("OPENAI_SUMMARIZE_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")  # vacio = default OpenAI
QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K", "8"))          # hits finales al LLM (post-dedupe)
QDRANT_POOL = int(os.getenv("QDRANT_POOL", "40"))           # candidatos pre-dedupe
SNIPPET_CHARS = int(os.getenv("SNIPPET_CHARS", "1500"))
COMPACT_THRESHOLD_TOKENS = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "8000"))
MODEL_NAME_EXPOSED = os.getenv("MODEL_NAME_EXPOSED", "mg-bot")

# Saludos triviales -> skip RAG (evita cold start + latencia + hits basura)
_TRIVIAL_PATTERNS = {"hola", "buenas", "buen dia", "buenos dias", "buenas tardes",
                     "buenas noches", "gracias", "muchas gracias", "chau", "adios",
                     "ok", "listo", "perfecto", "genial", "dale", "si", "no", "ola"}


app = FastAPI(title="MG RAG Gateway", version="1.0")


# --------------------- MODELS (OpenAI-compat) ---------------------


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionsReq(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


# --------------------- LLM CALL (OpenAI) ---------------------


def _openai_client():
    from openai import OpenAI
    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return OpenAI(**kwargs)


def _count_tokens(text: str) -> int:
    """Aprox: 1 token = 4 chars. Cheap y suficiente para el threshold."""
    return max(1, len(text) // 4)


def _total_tokens(messages: list[Message]) -> int:
    return sum(_count_tokens(m.content) for m in messages)


# --------------------- RAG ---------------------


def _is_trivial(query: str) -> bool:
    """Query trivial (saludo/agradecimiento) -> no vale RAG."""
    q = query.strip().lower().rstrip("?!.").strip()
    if not q or len(q) <= 2:
        return True
    if q in _TRIVIAL_PATTERNS:
        return True
    words = q.split()
    if len(words) <= 3 and all(w in _TRIVIAL_PATTERNS or len(w) <= 3 for w in words):
        return True
    return False


def _extract_keywords(query: str) -> list[str]:
    """Palabras candidatas para keyword boost. Filtra stopwords y palabras cortas."""
    stop = {"de","la","el","los","las","un","una","que","en","por","para","con","sin",
            "revisa","revisar","buscar","busca","buscá","dame","dime","quiero","necesito",
            "saber","informacion","informacion","informar","archivo","archivos","documento",
            "documentos","algo","sobre","del","al","es","son","hay","por favor","favor"}
    words = [w.strip(".,;:¿?¡!\"'()").lower() for w in query.split()]
    return [w for w in words if len(w) >= 4 and w not in stop]


def _rag_context(query: str) -> tuple[str, list[dict]]:
    """Retrieval con dedupe por (path, page) + keyword boost.

    Flujo:
        1. Trae POOL candidatos (default 40)
        2. Boost score si filename/snippet matchean keywords del query
        3. Dedupe: 1 hit por (path, page), el de mayor score
        4. Devuelve top TOP_K
    """
    from rag import qdrant_backend, reranker
    qvec = reranker.embed_query(query).tolist()
    hits = qdrant_backend.search(qvec, limit=QDRANT_POOL)
    if not hits:
        return "", []

    kws = _extract_keywords(query)
    if kws:
        for h in hits:
            hay = (h.get("filename", "") + " " + h.get("snippet", "")).lower()
            match_count = sum(1 for k in kws if k in hay)
            h["score"] = h.get("score", 0) + 0.1 * match_count  # boost 0.1 por keyword

    # Dedupe: 1 hit por (path, page), quedarse con mejor score
    best: dict[tuple, dict] = {}
    for h in hits:
        key = (h.get("path"), h.get("page"))
        if key not in best or h["score"] > best[key]["score"]:
            best[key] = h
    hits = sorted(best.values(), key=lambda x: x["score"], reverse=True)[:QDRANT_TOP_K]

    lines = ["## Contexto (documentos del estudio)\n"]
    for i, h in enumerate(hits, 1):
        fn = h.get("filename", "?")
        pg = h.get("page", "?")
        snip = (h.get("snippet") or "").strip()[:SNIPPET_CHARS]
        lines.append(f"[{i}] {fn} (pag. {pg}, score {h.get('score', 0):.2f})\n{snip}\n")
    lines.append("\n## Instruccion\nRespondes en espanol, tono profesional legal argentino. "
                 "Analiza CADA snippet del contexto antes de decir que no encontras nada. "
                 "Si algo del contexto es relevante aunque sea parcialmente, usalo y citalo. "
                 "Formato de cita OBLIGATORIO: (archivo.pdf, pag. X). "
                 "Solo decis 'no encuentro' si ningun snippet contiene el tema.\n")
    return "\n".join(lines), hits


# --------------------- COMPACTACION ---------------------


def _summarize(messages: list[Message]) -> str:
    """Resume una lista de mensajes en 1-2 parrafos. Llamada sincrona a OpenAI."""
    if not messages:
        return ""
    body = "\n\n".join(f"[{m.role}]: {m.content}" for m in messages)
    prompt = ("Resumi esta conversacion en 1-2 parrafos claros, preservando datos concretos "
              "(nombres, fechas, montos, decisiones tomadas). Zero relleno.\n\n"
              f"{body}\n\nResumen:")
    try:
        client = _openai_client()
        r = client.chat.completions.create(
            model=OPENAI_SUMMARIZE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        logger.exception("Fallo summarize: %s", e)
        return "[Resumen no disponible por error interno.]"


def _maybe_compact(messages: list[Message]) -> list[Message]:
    """Si total_tokens > threshold, resume todos menos los ultimos 4 turnos."""
    total = _total_tokens(messages)
    if total <= COMPACT_THRESHOLD_TOKENS:
        return messages

    system_msgs = [m for m in messages if m.role == "system"]
    non_system = [m for m in messages if m.role != "system"]
    if len(non_system) <= 4:
        return messages
    to_summarize = non_system[:-4]
    tail = non_system[-4:]
    logger.info("Compactando %d mensajes (~%d tokens)", len(to_summarize), total)
    resumen = _summarize(to_summarize)
    system_extra = Message(
        role="system",
        content=f"## Resumen de conversacion anterior\n{resumen}",
    )
    return system_msgs + [system_extra] + tail


# --------------------- STREAMING SSE (OpenAI compat) ---------------------


def _sse_chunk(model: str, chat_id: str, delta_text: str, finish_reason: str | None = None) -> str:
    payload = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": delta_text} if delta_text else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_openai(messages: list[Message], model: str) -> AsyncIterator[str]:
    client = _openai_client()
    msgs_dict = [{"role": m.role, "content": m.content} for m in messages]
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    try:
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs_dict,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content or ""
            if text:
                yield _sse_chunk(model, chat_id, text)
        yield _sse_chunk(model, chat_id, "", finish_reason="stop")
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.exception("Stream OpenAI fallo")
        err_msg = f"\n\n[Error interno del gateway: {type(e).__name__}: {e}]"
        yield _sse_chunk(model, chat_id, err_msg, finish_reason="stop")
        yield "data: [DONE]\n\n"


# --------------------- ENDPOINTS ---------------------


@app.get("/health")
def health():
    ok_qdrant = False
    n_points = 0
    try:
        from rag import qdrant_backend
        info = qdrant_backend.stats()
        ok_qdrant = "error" not in info
        n_points = info.get("points", 0)
    except Exception as e:
        logger.warning("Qdrant stats err: %s", e)
    return {
        "status": "ok",
        "qdrant_ok": ok_qdrant,
        "qdrant_points": n_points,
        "openai_configured": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
    }


@app.get("/v1/models")
def list_models():
    """Open WebUI llama a /v1/models al arrancar."""
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME_EXPOSED,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "mg-bot",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsReq):
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY no configurada")
    if not req.messages:
        raise HTTPException(400, "messages vacio")

    last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
    if not last_user:
        raise HTTPException(400, "sin user message")

    # ponytail: saludo trivial -> skip RAG (evita cold start MiniLM + latencia + basura)
    if _is_trivial(last_user.content):
        logger.info("Query trivial, skip RAG: %r", last_user.content)
        ctx_block, hits = "", []
    else:
        try:
            ctx_block, hits = _rag_context(last_user.content)
        except Exception as e:
            logger.exception("RAG fallo")
            ctx_block, hits = "", []

    messages = list(req.messages)
    if ctx_block:
        messages = [Message(role="system", content=ctx_block)] + messages
    messages = _maybe_compact(messages)

    logger.info("Chat: hits=%d, tokens_in=~%d, stream=%s",
                len(hits), _total_tokens(messages), req.stream)

    if not req.stream:
        client = _openai_client()
        msgs_dict = [{"role": m.role, "content": m.content} for m in messages]
        r = client.chat.completions.create(model=OPENAI_MODEL, messages=msgs_dict)
        text = r.choices[0].message.content or ""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
        }

    return StreamingResponse(
        _stream_openai(messages, req.model),
        media_type="text/event-stream",
    )


@app.get("/")
def root():
    return {"service": "mg-rag-gateway", "docs": "/docs"}
