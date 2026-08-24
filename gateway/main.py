"""RAG Gateway: expone endpoint OpenAI-compat que Open WebUI consume.

Flujo por request:
    1. Recibe chat completions request (mensajes + config)
    2. Toma el ultimo user message = query
    3. Si es trivial -> skip RAG y contesta directo
    4. Sino: query expansion + vector search + filename hybrid + rerank + dedupe
    5. Contexto ampliado por (path, pag +/- CONTEXT_RADIUS) -> mas texto por hit
    6. System prompt legal argentino riguroso
    7. Auto-compactacion si historial >THRESHOLD
    8. Stream a OpenAI, chunks OpenAI-compat de vuelta

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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")               # calidad legal, no mini
OPENAI_SUMMARIZE_MODEL = os.getenv("OPENAI_SUMMARIZE_MODEL", "gpt-4o-mini")
OPENAI_QUERY_MODEL = os.getenv("OPENAI_QUERY_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K", "10"))              # hits finales al LLM
QDRANT_POOL = int(os.getenv("QDRANT_POOL", "60"))                # candidatos pre-dedupe
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "30"))              # top-N que pasan por cross-encoder
SNIPPET_CHARS = int(os.getenv("SNIPPET_CHARS", "1800"))
CONTEXT_RADIUS = int(os.getenv("CONTEXT_RADIUS", "1"))           # +/- N paginas alrededor de cada hit
COMPACT_THRESHOLD_TOKENS = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "12000"))
ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "1") not in ("0", "false", "False")
ENABLE_CROSS_ENCODER = os.getenv("ENABLE_CROSS_ENCODER", "1") not in ("0", "false", "False")
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
MODEL_NAME_EXPOSED = os.getenv("MODEL_NAME_EXPOSED", "mg-bot")

_TRIVIAL_PATTERNS = {"hola", "buenas", "buen dia", "buenos dias", "buenas tardes",
                     "buenas noches", "gracias", "muchas gracias", "chau", "adios",
                     "ok", "listo", "perfecto", "genial", "dale", "si", "no", "ola"}


# --------------------- SYSTEM PROMPT LEGAL ---------------------

SYSTEM_PROMPT_LEGAL = """Sos asistente juridico senior del Estudio Montoya-Gherzi, especializado en derecho argentino (civil, comercial, laboral, seguros, danos y perjuicios).

## PRINCIPIOS INVIOLABLES

1. **PRECISION ABSOLUTA**. Un detalle mal citado o inventado puede arruinar un juicio. Nunca inventes fechas, montos, nombres, articulos, jurisdicciones ni jurisprudencia. Si no lo ves textual en el contexto que se te entrega, no lo afirmas.

2. **CITA OBLIGATORIA Y VERIFICABLE**. Todo dato factico DEBE ir acompanado de su fuente en el formato EXACTO:
   `(nombre_archivo.ext, pag. X)`
   - El `nombre_archivo.ext` debe coincidir textualmente con el filename del CONTEXTO (case y espacios incluidos)
   - El numero de pagina debe existir en el CONTEXTO
   - Si el mismo dato aparece en varios documentos, citas TODOS
   - Prohibido inventar filenames o paginas

3. **FIDELIDAD TEXTUAL**. Cuando reproduces clausulas, articulos o parrafos, transcribis ENTRE COMILLAS tal cual estan en el snippet, sin parafrasear. Si necesitas resumir, primero cita textual y despues tu resumen entre corchetes: "[resumen: ...]".

4. **RECONOCER LIMITES**. Si el contexto no cubre el tema o esta incompleto, lo decis explicitamente:
   > "En los documentos indexados no consta [tema]. Puede estar en archivos no indexados aun o requerir verificacion humana."
   Nunca rellenes huecos con conocimiento general de codigos legales.

5. **ESPANOL RIOPLATENSE FORMAL**. Tono profesional legal argentino, terminologia procesal correcta. Nunca anglicismos ni jerga.

## RAZONAMIENTO PREVIO (Chain of Thought silencioso)

Antes de responder, mentalmente:
- (a) Lei CADA snippet del contexto de arriba a abajo
- (b) Identifique los snippets relevantes al tema (no los descarte por semantica pobre)
- (c) Detecte contradicciones entre documentos, si las hay
- (d) Ordene los datos citables con su path + pag exactos
- (e) Marcar que falta o requiere verificacion humana

## FORMATO DE RESPUESTA CONSULTA/ANALISIS

**Respuesta directa** (1-2 oraciones al inicio).

**Fundamento**: citas textuales entre comillas con `(archivo, pag)`, agrupadas por tema.

**Contradicciones o dudas**: si dos documentos dicen cosas distintas, las marcas.

**Datos faltantes**: lista de cosas que necesitarias del expediente/cliente para dar respuesta completa.

## FORMATO DE ESCRITOS LEGALES (demanda, contestacion, cedula, oficio, alegato, apelacion, memorial)

Cuando el pedido es un escrito, generas el documento completo con este esquema (procesal argentino):

```
Sr. Juez / Sra. Jueza a cargo del [FALTA: juzgado]

Autos: "[FALTA: caratula]" - Expte. Nro. [FALTA: numero]

[Titulo del escrito EN MAYUSCULA]

I. OBJETO. En X caracter, por medio del presente vengo a [objeto].

II. HECHOS. [con citas si vienen del contexto].

III. DERECHO. [normas aplicables citadas].

IV. PRUEBA. [ofrece prueba documental, informativa, testimonial, pericial].

V. PETITORIO. Por lo expuesto, solicito: 1) [...]; 2) [...]; 3) [...].

Provea V.S. de conformidad,
SERA JUSTICIA.

[FALTA: firma abogado, matricula, domicilio constituido]
```

REGLAS ESCRITOS:
- Usas EXCLUSIVAMENTE datos del contexto para nombres, expedientes, hechos, articulos citados
- Cualquier dato faltante -> `[FALTA: descripcion breve]` para que el abogado complete
- No inventes hechos ni normas
- Tono formal procesal

## EJEMPLO DE RESPUESTA CORRECTA

Consulta: "que dice la clausula de exclusion por alcoholemia"

Respuesta:
La poliza excluye cobertura si el conductor tenia alcohol en sangre superior al limite legal al momento del siniestro.

**Fundamento textual:**
"Quedan excluidos de la cobertura los siniestros producidos en ocasion o consecuencia del estado de ebriedad del conductor o de haber ingerido cualquier tipo de estupefaciente" (POLIZA (3).pdf, pag. 40).

"La compania no indemnizara al asegurado cuando el conductor circulara con una alcoholemia superior a 0,50 gramos por litro de sangre" (ACEPTA CITACION MEED.pdf, pag. 52).

**Faltantes:** para casos concretos verificar el resultado del test de alcoholemia del acta policial.
"""


# Detecta si el user pide un ESCRITO LEGAL (demanda, contestacion, cedula, etc)
_ESCRITO_KEYWORDS = {
    "redacta", "redactar", "redactame", "redactame", "hace", "hacé", "haceme",
    "escribi", "escribir", "escribime", "genera", "generar", "generame",
    "arma", "armar", "armame", "prepara", "preparar", "preparame",
    "borrador", "modelo",
}
_ESCRITO_TIPOS = {
    "demanda", "contestacion", "contestación", "cedula", "cédula", "oficio",
    "alegato", "apelacion", "apelación", "memorial", "escrito", "carta documento",
    "notificacion", "notificación", "recurso", "peticion", "petición",
    "reconvencion", "reconvención", "excepcion", "excepción", "denuncia",
}


def _is_escrito_request(query: str) -> bool:
    """True si el user pide un escrito legal formal (dispara template especifico)."""
    q = query.lower()
    return any(kw in q for kw in _ESCRITO_KEYWORDS) and any(t in q for t in _ESCRITO_TIPOS)


app = FastAPI(title="MG RAG Gateway", version="2.0")


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionsReq(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


# --------------------- LLM ---------------------


def _openai_client():
    from openai import OpenAI
    kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL
    return OpenAI(**kwargs)


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _total_tokens(messages: list[Message]) -> int:
    return sum(_count_tokens(m.content) for m in messages)


# --------------------- QUERY UTILS ---------------------


def _is_trivial(query: str) -> bool:
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
    stop = {"de","la","el","los","las","un","una","que","en","por","para","con","sin",
            "revisa","revisar","buscar","busca","buscá","dame","dime","quiero","necesito",
            "saber","informacion","informacion","informar","archivo","archivos","documento",
            "documentos","algo","sobre","del","al","es","son","hay","por favor","favor",
            "dice","este","esta","estas","estos","como","cuando","donde","cual","cuales",
            "hacer","haces","cita","citar","transcribir","copiar","texto","dato","datos"}
    words = [w.strip(".,;:¿?¡!\"'()").lower() for w in query.split()]
    # min_len 2 para no perder siglas legales (AZ, IOL, ADR, CSJ, SCJ, etc)
    return [w for w in words if len(w) >= 2 and w not in stop and not w.isdigit()]


def _expand_query(query: str) -> list[str]:
    """Pide al LLM 2-3 variantes semanticas de la query para ampliar el retrieval.
    Cada variante es una reformulacion o palabras clave alternativas.
    """
    if not ENABLE_QUERY_EXPANSION or not OPENAI_API_KEY:
        return [query]
    prompt = (
        "Eres un asistente que reformula queries de busqueda legal argentina.\n"
        "Genera 3 variantes semanticas de la siguiente consulta (una por linea, sin numerar, sin comentarios).\n"
        "Las variantes deben ser palabras clave o frases cortas que capturen el TEMA CENTRAL.\n"
        "Ej. query: 'que dice el contrato de alquiler de Perez'\n"
        "-> alquiler Perez\n"
        "-> contrato locacion Perez\n"
        "-> obligaciones inquilino Perez\n\n"
        f"Query: {query}\n"
        "Variantes:"
    )
    try:
        client = _openai_client()
        r = client.chat.completions.create(
            model=OPENAI_QUERY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        text = (r.choices[0].message.content or "").strip()
        variants = [ln.strip("-•* ").strip() for ln in text.splitlines() if ln.strip()]
        variants = [v for v in variants if 3 <= len(v) <= 200]
        # Original siempre primero
        return [query] + variants[:3]
    except Exception as e:
        logger.warning("Query expansion fallo: %s", str(e)[:100])
        return [query]


# --------------------- RETRIEVAL ---------------------


_ce_model = None


def _cross_encoder():
    """Carga lazy del cross-encoder (multilingual). Compartido entre requests."""
    global _ce_model
    if _ce_model is not None:
        return _ce_model
    if not ENABLE_CROSS_ENCODER:
        return None
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Cargando cross-encoder %s...", CROSS_ENCODER_MODEL)
        _ce_model = CrossEncoder(CROSS_ENCODER_MODEL)
        return _ce_model
    except Exception as e:
        logger.warning("Cross-encoder no disponible: %s", str(e)[:100])
        return None


def _rerank(query: str, hits: list[dict], top_n: int) -> list[dict]:
    """Rerank con cross-encoder. Devuelve top_n reordenados por relevancia real query-snippet."""
    ce = _cross_encoder()
    if ce is None or not hits:
        return hits[:top_n]
    pairs = [(query, (h.get("filename", "") + " " + (h.get("snippet") or ""))[:2000]) for h in hits]
    try:
        scores = ce.predict(pairs, show_progress_bar=False)
        for h, s in zip(hits, scores):
            h["_ce_score"] = float(s)
        return sorted(hits, key=lambda x: x.get("_ce_score", 0), reverse=True)[:top_n]
    except Exception as e:
        logger.warning("Rerank fallo: %s", str(e)[:100])
        return hits[:top_n]


def _filename_hits(keywords: list[str], limit: int = 20) -> list[dict]:
    """Hybrid: hits por keyword match en filename (indice TEXT en Qdrant).

    Marca hits con `_source='filename'` para que el pipeline los proteja del rerank.
    Score = 1.0 + 0.1 * keywords_matched (multiples matches = mas confianza).
    """
    if not keywords:
        return []
    from rag import qdrant_backend
    from qdrant_client.models import Filter, FieldCondition, MatchText
    c = qdrant_backend.get_client()

    # Contar matches por (path, page) para saber cuantos keywords matchean
    match_count: dict[tuple, int] = {}
    payload_by_key: dict[tuple, dict] = {}
    for k in keywords[:8]:
        try:
            batch, _ = c.scroll(
                collection_name=qdrant_backend.COLLECTION,
                scroll_filter=Filter(must=[FieldCondition(key="filename", match=MatchText(text=k))]),
                limit=limit, with_payload=True,
            )
            for p in batch:
                key = (p.payload.get("path"), p.payload.get("page"))
                match_count[key] = match_count.get(key, 0) + 1
                payload_by_key[key] = p.payload
        except Exception as e:
            logger.warning("filename_hits fallo kw=%r: %s", k, str(e)[:80])

    results = []
    for key, cnt in sorted(match_count.items(), key=lambda x: -x[1]):
        d = dict(payload_by_key[key])
        d["score"] = 1.0 + 0.1 * cnt              # score alto para competir con vector
        d["_source"] = "filename"
        d["_kw_matches"] = cnt
        results.append(d)
    return results[:limit]


def _expand_context(hit: dict, radius: int) -> str:
    """Trae texto de paginas adyacentes (pag +/- radius) del mismo path."""
    if radius <= 0:
        return hit.get("snippet") or ""
    from rag import qdrant_backend
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
    path = hit.get("path")
    page = hit.get("page")
    if not path or page is None:
        return hit.get("snippet") or ""
    try:
        c = qdrant_backend.get_client()
        batch, _ = c.scroll(
            collection_name=qdrant_backend.COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="path", match=MatchValue(value=path)),
                FieldCondition(key="page", range=Range(gte=page - radius, lte=page + radius)),
            ]),
            limit=32, with_payload=True,
        )
        by_pg: dict[int, list[str]] = {}
        for p in batch:
            pg = int(p.payload.get("page", 0))
            snip = (p.payload.get("snippet") or "").strip()
            if snip:
                by_pg.setdefault(pg, []).append(snip)
        parts = []
        for pg in sorted(by_pg.keys()):
            joined = " ".join(by_pg[pg])[:SNIPPET_CHARS]
            marker = " [PAG ACTUAL]" if pg == page else ""
            parts.append(f"--- Pag. {pg}{marker} ---\n{joined}")
        return "\n\n".join(parts) if parts else (hit.get("snippet") or "")
    except Exception as e:
        logger.warning("expand_context fallo path=%r pg=%s: %s", path, page, str(e)[:80])
        return hit.get("snippet") or ""


def _rag_context(query: str) -> tuple[str, list[dict]]:
    """Retrieval multi-fase:
        1. Expandir query (LLM -> 3 variantes)
        2. Cada variante: vector search + filename hybrid
        3. Merge, keyword boost, dedupe por (path, page)
        4. Top-K con contexto ampliado (paginas +/- radius)
    """
    from rag import qdrant_backend, reranker

    variants = _expand_query(query)
    logger.info("Query variants (%d): %r", len(variants), variants)

    kws = _extract_keywords(query)
    for v in variants[1:]:
        kws.extend(_extract_keywords(v))
    kws = list(dict.fromkeys(kws))  # dedupe preservando orden

    all_hits: list[dict] = []
    for v in variants:
        try:
            qvec = reranker.embed_query(v).tolist()
            all_hits.extend(qdrant_backend.search(qvec, limit=QDRANT_POOL))
        except Exception as e:
            logger.warning("Search v=%r fallo: %s", v[:40], str(e)[:80])

    # Boost por keyword en filename/snippet
    if kws:
        for h in all_hits:
            hay = (h.get("filename", "") + " " + h.get("snippet", "")).lower()
            match_count = sum(1 for k in kws if k in hay)
            h["score"] = h.get("score", 0) + 0.08 * match_count

    # Filename hybrid
    all_hits.extend(_filename_hits(kws, limit=25))

    if not all_hits:
        return "", []

    # Dedupe por (path, page). Si hay conflicto vector vs filename, PROTEGER filename.
    best: dict[tuple, dict] = {}
    for h in all_hits:
        key = (h.get("path"), h.get("page"))
        # Filename hit siempre gana (queremos protegerlo del rerank)
        if key not in best:
            best[key] = h
        elif h.get("_source") == "filename" and best[key].get("_source") != "filename":
            best[key] = h
        elif h.get("_source") != "filename" and best[key].get("_source") == "filename":
            pass  # mantener el filename hit
        elif h["score"] > best[key]["score"]:
            best[key] = h

    # Separar filename-hits (protegidos) de semantic-hits (rerankeables)
    fn_protected = [h for h in best.values() if h.get("_source") == "filename"]
    semantic = [h for h in best.values() if h.get("_source") != "filename"]
    fn_protected.sort(key=lambda x: (-x.get("_kw_matches", 0), -x.get("score", 0)))
    semantic.sort(key=lambda x: -x["score"])
    semantic = semantic[:RERANK_TOP_N]

    # Rerank SOLO los semantic (los filename ya son alta confianza)
    semantic_reranked = _rerank(query, semantic, QDRANT_TOP_K)

    # Merge: reservar hasta N_FILENAME_SLOTS slots para filename-hits + resto semantic
    n_fn_slots = min(len(fn_protected), max(3, QDRANT_TOP_K // 2))
    top = fn_protected[:n_fn_slots] + semantic_reranked[:QDRANT_TOP_K - n_fn_slots]

    logger.info("Retrieval: %d cand -> dedup=%d (fn=%d, sem=%d) -> rerank sem -> top=%d (fn_slots=%d)",
                len(all_hits), len(best), len(fn_protected), len(semantic), len(top), n_fn_slots)
    for i, h in enumerate(top[:6], 1):
        src = h.get("_source", "vec")
        logger.info("  [%d] src=%s score=%.2f ce=%.2f kw=%d  %s p.%s",
                    i, src, h.get("score", 0), h.get("_ce_score", 0),
                    h.get("_kw_matches", 0), h.get("filename", "?"), h.get("page", "?"))

    # Armar contexto con paginas adyacentes
    lines = ["## CONTEXTO — Extractos de documentos del Estudio\n"]
    for i, h in enumerate(top, 1):
        fn = h.get("filename", "?")
        pg = h.get("page", "?")
        path = h.get("path", "?")
        expanded = _expand_context(h, CONTEXT_RADIUS)
        lines.append(
            f"\n### [{i}] {fn} — pag. {pg} (score {h.get('score', 0):.2f})\n"
            f"_Ruta: {path}_\n\n{expanded}\n"
        )
    lines.append("\n---\n")
    return "\n".join(lines), top


# --------------------- COMPACTACION ---------------------


def _summarize(messages: list[Message]) -> str:
    if not messages:
        return ""
    body = "\n\n".join(f"[{m.role}]: {m.content}" for m in messages)
    prompt = ("Resumi esta conversacion legal en 1-2 parrafos claros, preservando datos concretos "
              "(nombres de partes, expedientes, fechas, montos, articulos citados, decisiones tomadas). "
              "Zero relleno.\n\n"
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


# --------------------- STREAMING ---------------------


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


async def _stream_openai(messages: list[Message], model: str, hits: list[dict], active_model: str = None) -> AsyncIterator[str]:
    """Stream la respuesta y al final valida las citas contra los hits reales."""
    client = _openai_client()
    msgs_dict = [{"role": m.role, "content": m.content} for m in messages]
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    buffered = []
    llm_model = active_model or OPENAI_MODEL
    try:
        stream = client.chat.completions.create(
            model=llm_model,
            messages=msgs_dict,
            stream=True,
            temperature=0.1,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            text = chunk.choices[0].delta.content or ""
            if text:
                buffered.append(text)
                yield _sse_chunk(model, chat_id, text)

        # Validacion de citas post-stream (agrega footer si hay invalidas)
        if hits:
            full = "".join(buffered)
            v = _validate_citations(full, hits)
            if v["invalidas"]:
                warning = (
                    "\n\n---\n"
                    f"⚠️ **Verificación de citas**: {len(v['invalidas'])} cita(s) "
                    "no coincide(n) con archivos del contexto:\n"
                    + "\n".join(f"  - `{c}`" for c in v["invalidas"])
                    + "\n\nRevisar antes de usar. Esto puede ser tipeo del LLM o alucinación."
                )
                logger.warning("Citas invalidas: %s", v["invalidas"])
                yield _sse_chunk(model, chat_id, warning)
            else:
                logger.info("Citas OK: %d validas", v["total"])

        yield _sse_chunk(model, chat_id, "", finish_reason="stop")
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.exception("Stream OpenAI fallo")
        err_msg = f"\n\n[Error interno del gateway: {type(e).__name__}: {e}]"
        yield _sse_chunk(model, chat_id, err_msg, finish_reason="stop")
        yield "data: [DONE]\n\n"


# --------------------- ENDPOINTS ---------------------


def _validate_citations(answer: str, hits: list[dict]) -> dict:
    """Extrae citas del texto y verifica que los filenames existan en hits.
    Formato buscado: (archivo.ext, pag. N) o variantes.
    Devuelve estadisticas: total_citas, validas, invalidas (filenames que no estan en hits).
    """
    import re
    valid_files = {(h.get("filename") or "").lower() for h in hits}
    # Regex acepta filenames con parentesis internos tipo "POLIZA (3).pdf"
    pat = re.compile(
        r"\(([^,()]*(?:\([^)]*\))?[^,()]*\.(?:pdf|docx|doc|txt))[,\s]+p[aá]g\.?\s*(\d+)\)",
        re.IGNORECASE,
    )
    matches = pat.findall(answer)
    validas = []
    invalidas = []
    for fn, pg in matches:
        fn_clean = fn.strip().lower()
        (validas if fn_clean in valid_files else invalidas).append(f"{fn} p.{pg}")
    return {
        "total": len(matches),
        "validas": validas,
        "invalidas": invalidas,
    }


@app.post("/debug/query")
async def debug_query(req: dict):
    """Endpoint de debug: recibe {'query': '...'} y devuelve hits + citas si querés testear.
    Uso: docker exec mg-gateway python -c "import urllib.request,json; ..."
    """
    q = req.get("query", "").strip()
    if not q:
        raise HTTPException(400, "query vacia")
    if _is_trivial(q):
        return {"trivial": True, "hits": []}
    ctx, hits = _rag_context(q)
    return {
        "trivial": False,
        "n_hits": len(hits),
        "top_hits": [
            {
                "filename": h.get("filename"),
                "page": h.get("page"),
                "score": h.get("score"),
                "ce_score": h.get("_ce_score"),
                "snippet_preview": (h.get("snippet") or "")[:200],
            }
            for h in hits
        ],
        "context_chars": len(ctx),
    }


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
        "query_expansion": ENABLE_QUERY_EXPANSION,
        "context_radius": CONTEXT_RADIUS,
    }


@app.get("/v1/models")
def list_models():
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

    intent = "consulta"
    if _is_trivial(last_user.content):
        intent = "trivial"
        logger.info("Query trivial, skip RAG: %r", last_user.content)
        ctx_block, hits = "", []
    else:
        if _is_escrito_request(last_user.content):
            intent = "escrito"
        try:
            ctx_block, hits = _rag_context(last_user.content)
        except Exception as e:
            logger.exception("RAG fallo")
            ctx_block, hits = "", []
    logger.info("Intent detectado: %s", intent)

    # ponytail: para triviales usar modelo rapido + prompt minimo
    active_model = OPENAI_QUERY_MODEL if intent == "trivial" else OPENAI_MODEL
    if intent == "trivial":
        SYSTEM_ACTIVE = (
            "Sos asistente breve del Estudio Montoya-Gherzi. Responde el saludo/agradecimiento "
            "en 1 oracion, tono profesional. Ofrece ayuda si corresponde."
        )
    else:
        SYSTEM_ACTIVE = SYSTEM_PROMPT_LEGAL

    # System prompt (segun intent) PRIMERO, despues contexto RAG, despues intent hint, despues historial
    system_stack: list[Message] = [Message(role="system", content=SYSTEM_ACTIVE)]
    if ctx_block:
        system_stack.append(Message(role="system", content=ctx_block))
    if intent == "escrito":
        system_stack.append(Message(
            role="system",
            content=("## INTENCION DETECTADA: GENERAR ESCRITO LEGAL FORMAL\n"
                     "El usuario pide un escrito procesal. Aplicas el formato de escritos "
                     "(Sr. Juez, Autos, Objeto, Hechos, Derecho, Prueba, Petitorio, Sera Justicia). "
                     "Datos faltantes -> [FALTA: descripcion]. NO redactes preambulo conversacional; "
                     "empieza directo con el encabezado del escrito."),
        ))

    # Remover systems ya presentes en req para no duplicar
    user_msgs = [m for m in req.messages if m.role != "system"]
    messages = system_stack + user_msgs
    messages = _maybe_compact(messages)

    logger.info("Chat: hits=%d, tokens_in=~%d, stream=%s, model=%s",
                len(hits), _total_tokens(messages), req.stream, OPENAI_MODEL)

    if not req.stream:
        client = _openai_client()
        msgs_dict = [{"role": m.role, "content": m.content} for m in messages]
        r = client.chat.completions.create(
            model=active_model, messages=msgs_dict, temperature=0.1,
        )
        text = r.choices[0].message.content or ""
        # Validar citas
        if hits:
            v = _validate_citations(text, hits)
            if v["invalidas"]:
                text += (
                    "\n\n---\n"
                    f"⚠️ **Verificación de citas**: {len(v['invalidas'])} no coincide(n) "
                    "con archivos del contexto:\n"
                    + "\n".join(f"  - `{c}`" for c in v["invalidas"])
                )
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
        _stream_openai(messages, req.model, hits, active_model=active_model),
        media_type="text/event-stream",
    )


@app.get("/")
def root():
    return {"service": "mg-rag-gateway", "docs": "/docs"}
