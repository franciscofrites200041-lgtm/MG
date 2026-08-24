"""RAG Gateway: expone endpoint OpenAI-compatible que Open WebUI consume.

Flujo por request:
    1. Recibe chat completions request (mensajes + config)
    2. Toma el ultimo user message = query
    3. Embed + Qdrant search -> top-k chunks
    4. Arma prompt: system (con contexto RAG) + historial (compactado si excede) + query
    5. Llama Gemini con streaming
    6. Devuelve chunks OpenAI-formatted al frontend

Auto-compactacion:
    Si el total_tokens estimado del historial supera COMPACT_THRESHOLD,
    los mensajes anteriores al ultimo turno se reemplazan por un resumen
    (llamando a Gemini con prompt de resumen). Se guarda como system extra.

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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("RAG_BACKEND", "qdrant")

logger = logging.getLogger("gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
GEMINI_SUMMARIZE_MODEL = os.getenv("GEMINI_SUMMARIZE_MODEL", "gemini-2.0-flash-exp")
QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K", "8"))
COMPACT_THRESHOLD_TOKENS = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "8000"))
MODEL_NAME_EXPOSED = os.getenv("MODEL_NAME_EXPOSED", "mg-bot-gemini")


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


# --------------------- LLM CALL (Gemini) ---------------------


def _gemini_client():
    from google import genai  # google-genai package
    return genai.Client(api_key=GEMINI_API_KEY)


def _to_gemini_contents(messages: list[Message]) -> tuple[str | None, list[dict]]:
    """Extrae system prompt y convierte al formato Gemini (role user/model)."""
    system_prompt = None
    contents = []
    for m in messages:
        if m.role == "system":
            system_prompt = (system_prompt + "\n\n" + m.content) if system_prompt else m.content
        else:
            role = "user" if m.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m.content}]})
    return system_prompt, contents


def _count_tokens(text: str) -> int:
    """Aprox: 1 token = 4 chars. Cheap y suficiente para el threshold."""
    return max(1, len(text) // 4)


def _total_tokens(messages: list[Message]) -> int:
    return sum(_count_tokens(m.content) for m in messages)


# --------------------- RAG ---------------------


def _rag_context(query: str) -> tuple[str, list[dict]]:
    """Devuelve (bloque_de_contexto_para_system, hits_json)."""
    from rag import qdrant_backend, reranker
    qvec = reranker.embed_query(query).tolist()
    hits = qdrant_backend.search(qvec, limit=QDRANT_TOP_K)
    if not hits:
        return "", []
    lines = ["## Contexto (documentos del estudio)\n"]
    for i, h in enumerate(hits, 1):
        fn = h.get("filename", "?")
        pg = h.get("page", "?")
        snip = (h.get("snippet") or "").strip()
        lines.append(f"[{i}] {fn} (pag. {pg})\n{snip}\n")
    lines.append("\n## Instruccion\nRespondes en espanol. "
                 "Citas SIEMPRE con el formato (archivo.pdf, pag. X) cuando uses "
                 "un dato del contexto. Si el contexto no cubre la pregunta, decilo.\n")
    return "\n".join(lines), hits


# --------------------- COMPACTACION ---------------------


def _summarize(messages: list[Message]) -> str:
    """Resume una lista de mensajes en 1-2 parrafos. Llamada sincrona a Gemini."""
    if not messages:
        return ""
    body = "\n\n".join(f"[{m.role}]: {m.content}" for m in messages)
    prompt = ("Resumi esta conversacion en 1-2 parrafos claros, preservando datos concretos "
              "(nombres, fechas, montos, decisiones tomadas). Zero relleno.\n\n"
              f"{body}\n\nResumen:")
    try:
        client = _gemini_client()
        r = client.models.generate_content(
            model=GEMINI_SUMMARIZE_MODEL,
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
        )
        return (r.text or "").strip()
    except Exception as e:
        logger.exception("Fallo summarize: %s", e)
        return "[Resumen no disponible por error interno.]"


def _maybe_compact(messages: list[Message]) -> list[Message]:
    """Si total_tokens > threshold, resume todos menos los ultimos 4 turnos.

    Devuelve nueva lista: [system(s) originales, system_extra(resumen), ...ultimos_4].
    """
    total = _total_tokens(messages)
    if total <= COMPACT_THRESHOLD_TOKENS:
        return messages

    # ponytail: separar por rol, no por indice, para no romper si vienen desordenados
    system_msgs = [m for m in messages if m.role == "system"]
    non_system = [m for m in messages if m.role != "system"]
    if len(non_system) <= 4:
        return messages  # muy corto, no vale la pena
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


async def _stream_gemini(messages: list[Message], model: str) -> AsyncIterator[str]:
    from google import genai
    from google.genai import types as gtypes
    client = _gemini_client()
    system_prompt, contents = _to_gemini_contents(messages)
    config = gtypes.GenerateContentConfig(system_instruction=system_prompt) if system_prompt else None
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    try:
        stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=contents,
            config=config,
        )
        for chunk in stream:
            text = getattr(chunk, "text", "") or ""
            if text:
                yield _sse_chunk(model, chat_id, text)
        yield _sse_chunk(model, chat_id, "", finish_reason="stop")
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.exception("Stream Gemini fallo")
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
        "gemini_configured": bool(GEMINI_API_KEY),
        "model": GEMINI_MODEL,
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
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY no configurada")
    if not req.messages:
        raise HTTPException(400, "messages vacio")

    # 1) ultimo user message -> query RAG
    last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
    if not last_user:
        raise HTTPException(400, "sin user message")

    # 2) RAG context
    try:
        ctx_block, hits = _rag_context(last_user.content)
    except Exception as e:
        logger.exception("RAG fallo")
        ctx_block, hits = "", []

    # 3) armar messages finales: system(RAG) + historial (compactado si excede)
    messages = list(req.messages)
    if ctx_block:
        messages = [Message(role="system", content=ctx_block)] + messages
    messages = _maybe_compact(messages)

    logger.info("Chat: hits=%d, tokens_in=~%d, stream=%s",
                len(hits), _total_tokens(messages), req.stream)

    if not req.stream:
        # Modo no-streaming: coleccionar todo y devolver
        from google.genai import types as gtypes
        client = _gemini_client()
        system_prompt, contents = _to_gemini_contents(messages)
        config = gtypes.GenerateContentConfig(system_instruction=system_prompt) if system_prompt else None
        r = client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
        text = r.text or ""
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
        _stream_gemini(messages, req.model),
        media_type="text/event-stream",
    )


@app.get("/")
def root():
    return {"service": "mg-rag-gateway", "docs": "/docs"}
