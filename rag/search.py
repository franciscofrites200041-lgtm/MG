"""Busqueda RAG con backend switcheable.

RAG_BACKEND=qdrant (produccion VPS):
    query -> embed -> Qdrant top-N por cosine -> reranker (opcional, rerank cerrado)
    -> top_k final formateado con path/pagina/snippet.

RAG_BACKEND=sqlite (legacy / dev local):
    query -> FTS5 top-N -> promote previews on-demand -> re-FTS5
          -> embed query + cosine con chunk_embeddings -> reordenar -> top_k final.

leer_pagina_documento / leer_rango_documento en Qdrant reconstruyen la pagina
concatenando chunks por (path, page) ordenados por chunk_idx.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

logger = logging.getLogger("rag.search")

RAG_BACKEND = os.getenv("RAG_BACKEND", "sqlite").lower()
INDEX_DB = os.getenv("INDEX_DB_PATH", "./index.db")
NAS_ROOT = os.getenv("NAS_ROOT", "")
RERANK_POOL = int(os.getenv("RERANK_POOL", "20"))
QDRANT_SEARCH_LIMIT = int(os.getenv("QDRANT_SEARCH_LIMIT", "20"))
MAX_PROMOTIONS_PER_QUERY = int(os.getenv("MAX_PROMOTIONS_PER_QUERY", "5"))
NO_HIT_LOG = os.getenv("NO_HIT_LOG", "/data/logs/queries_no_hit.jsonl")

_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]{2,}")


def _sanitize_query(q: str) -> str:
    tokens = _TOKEN_RE.findall(q or "")
    return " ".join(tokens)


def _normalize_for_citation(s: str) -> str:
    """Lowercase + strip acentos + collapse spaces. Uso para comparar filenames tolerante."""
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str.lower().strip())


def citation_matches(cited_filename: str, context_paths: list[str]) -> bool:
    """True si el filename citado matchea (normalizado) algun path del contexto.

    Uso desde el verificador de citas del pipeline superior. Tolera mayusculas,
    acentos, y espacios extra.
    """
    cited_norm = _normalize_for_citation(cited_filename)
    if not cited_norm:
        return False
    for p in context_paths:
        pname = _normalize_for_citation(Path(p).name)
        if cited_norm == pname or cited_norm in pname or pname in cited_norm:
            return True
    return False


def _log_no_hit(query: str, analyzed, sugerencias: list[dict]) -> None:
    """Persiste queries fallidas para poder skillificarlas como evals despues."""
    try:
        line = json.dumps({
            "ts": time.time(),
            "query": query[:500],
            "carpeta": analyzed.carpeta,
            "expedientes": analyzed.expedientes,
            "nombres": analyzed.nombres_propios,
            "intent": analyzed.intent,
            "sugerencias": [s.get("path") for s in sugerencias[:5]],
        }, ensure_ascii=False)
        p = Path(NO_HIT_LOG)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.debug("no_hit log fallo: %s", e)


# =============== QDRANT PATH ===============

def _buscar_qdrant(query: str, top_k: int, prev_context=None) -> str:
    """Hybrid search sobre Qdrant: semantic + filename MatchText + path MatchText.

    Flujo:
    1. analyze_query -> extrae carpeta, expedientes, nombres, intent, negaciones
    2. hybrid retrieval combinando 3 canales
    3. Si intent='caso', reordena filenames que parecen expedientes primero
    4. Si zero hits, fallback: sugerencias por filename similarity
    """
    from rag import qdrant_backend, reranker
    from rag.query_analyzer import analyze_query

    # 1. Analisis. Enriquecido con carpetas conocidas para fuzzy match.
    try:
        carpetas = qdrant_backend.list_carpetas_top()
    except Exception as e:
        logger.warning("list_carpetas_top fallo, sigo sin fuzzy: %s", e)
        carpetas = []
    analyzed = analyze_query(query, carpetas_conocidas=carpetas, prev_context=prev_context)

    # 2. Embedding sobre texto limpio (sin numeros de expte, sin scope)
    try:
        qvec = reranker.embed_query(analyzed.text_semantic or query).tolist()
    except Exception as e:
        logger.exception("No pude embeddear query: %s", e)
        return f"Error del reranker: {e}"

    pool = max(top_k, QDRANT_SEARCH_LIMIT)

    # Filename terms: nombres propios + expedientes (los mas discriminativos)
    filename_terms = list(analyzed.nombres_propios) + list(analyzed.expedientes)

    try:
        hits = qdrant_backend.search_hybrid(
            qvec, limit=pool,
            carpeta=analyzed.carpeta,
            filename_terms=filename_terms,
            expedientes=analyzed.expedientes,
        )
    except Exception as e:
        logger.exception("Qdrant search_hybrid fallo: %s", e)
        return f"Error del index Qdrant: {e}"

    # 3. Intent-aware reorder: si el user pidio "casos", priorizar filenames tipo expediente
    if analyzed.intent == "caso":
        hits = _reorder_by_caso_signal(hits)

    # 4. Optional cross-encoder
    if reranker.is_enabled() and os.getenv("RERANK_CROSS_ENCODER") == "1":
        try:
            hits = _cross_encoder_rerank(analyzed.text, hits)
        except Exception as e:
            logger.warning("cross-encoder rerank fallo, cae a orden Qdrant: %s", e)

    hits = hits[:top_k]

    if not hits:
        # 5. Fallback: sugerir archivos por filename similarity
        sugerencias = []
        if filename_terms:
            try:
                sugerencias = qdrant_backend.search_filenames_similar(filename_terms, limit=3)
            except Exception as e:
                logger.warning("filename similar fallo: %s", e)
        _log_no_hit(query, analyzed, sugerencias)
        if sugerencias:
            lines = ["Sin resultados directos. Archivos con nombre parecido:"]
            for i, s in enumerate(sugerencias, 1):
                lines.append(f"[{i}] {s.get('filename', '?')}\n    path: {s.get('path')}")
            if analyzed.carpeta:
                lines.append(f"(scope aplicado: carpeta '{analyzed.carpeta}')")
            return "\n".join(lines)
        return "Sin resultados." + (
            f" (scope: carpeta '{analyzed.carpeta}')" if analyzed.carpeta else ""
        )

    lines = []
    for i, h in enumerate(hits, start=1):
        marker = f" (pag. {h.get('page', '?')})"
        snippet = (h.get("snippet") or "")[:300]
        canal = h.get("_channel", "")
        canal_hint = f" [{canal}]" if canal and canal != "semantic" else ""
        lines.append(
            f"[{i}] {h.get('filename', '?')}{marker}{canal_hint} - {snippet}\n"
            f"    path: {h.get('path')}"
        )
    return "\n".join(lines)


_RE_EXPTE_EN_FILENAME = re.compile(
    r"(?:N[°º]?\.?\s*\d|Expte|Exp\.|FMZ|\d{5,})|(?:\bc/\b)|(?:\bvs?\b)",
    re.IGNORECASE,
)


def _reorder_by_caso_signal(hits: list[dict]) -> list[dict]:
    """Cuando intent='caso', promueve hits cuyo filename parece un expediente concreto
    (contiene N°, Expte, apellidos c/ demandado, etc). Mantiene el orden relativo dentro
    de cada grupo.
    """
    con_signal = []
    sin_signal = []
    for h in hits:
        fn = h.get("filename", "") or ""
        if _RE_EXPTE_EN_FILENAME.search(fn):
            con_signal.append(h)
        else:
            sin_signal.append(h)
    return con_signal + sin_signal


def _cross_encoder_rerank(query: str, hits: list[dict]) -> list[dict]:
    """Hook para futuro cross-encoder. Hoy no-op: devuelve el mismo orden."""
    return hits


def _leer_pagina_qdrant(path: str, pagina: int) -> str:
    """Reconstruye una pagina concatenando chunks (path, page) por chunk_idx."""
    from rag import qdrant_backend
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    c = qdrant_backend.get_client()
    q_filter = Filter(must=[
        FieldCondition(key="path", match=MatchValue(value=path)),
        FieldCondition(key="page", match=MatchValue(value=pagina)),
    ])
    # scroll: no vector, solo payload
    scrolled, _ = c.scroll(
        collection_name=qdrant_backend.COLLECTION,
        scroll_filter=q_filter,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    if not scrolled:
        return f"No hay pagina {pagina} indexada para {path}."
    ordenados = sorted(scrolled, key=lambda p: p.payload.get("chunk_idx", 0))
    return "\n".join(p.payload.get("snippet", "") for p in ordenados)


def _leer_rango_qdrant(path: str, pagina_inicio: int, pagina_fin: int) -> str:
    from rag import qdrant_backend
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

    c = qdrant_backend.get_client()
    q_filter = Filter(must=[
        FieldCondition(key="path", match=MatchValue(value=path)),
        FieldCondition(key="page", range=Range(gte=pagina_inicio, lte=pagina_fin)),
    ])
    scrolled, _ = c.scroll(
        collection_name=qdrant_backend.COLLECTION,
        scroll_filter=q_filter,
        limit=500,
        with_payload=True,
        with_vectors=False,
    )
    if not scrolled:
        return f"Sin paginas en el rango {pagina_inicio}-{pagina_fin} para {path}."
    por_pagina: dict[int, list] = {}
    for p in scrolled:
        pg = p.payload.get("page", 0)
        por_pagina.setdefault(pg, []).append(p)
    out = []
    for pg in sorted(por_pagina):
        chunks = sorted(por_pagina[pg], key=lambda p: p.payload.get("chunk_idx", 0))
        out.append(f"[Pagina {pg}]\n" + "\n".join(p.payload.get("snippet", "") for p in chunks))
    return "\n\n".join(out)


def _stats_qdrant() -> dict:
    from rag import qdrant_backend
    return qdrant_backend.stats()


# =============== SQLITE PATH (legacy) ===============

def _rerank_rows(query_raw: str, rows: list[tuple], conn: sqlite3.Connection, top_k: int) -> list[tuple]:
    try:
        from rag import reranker
        if not reranker.is_enabled():
            return rows[:top_k]
        rowids = [r[0] for r in rows]
        placeholders = ",".join("?" * len(rowids))
        emb_rows = conn.execute(
            f"SELECT chunk_rowid, vec FROM chunk_embeddings WHERE chunk_rowid IN ({placeholders})",
            rowids,
        ).fetchall()
        if not emb_rows:
            return rows[:top_k]
        candidatos = [(rid, reranker.blob_to_vec(b)) for rid, b in emb_rows]
        qvec = reranker.embed_query(query_raw)
        orden = reranker.rerank_by_cosine(qvec, candidatos)
        by_rowid = {r[0]: r for r in rows}
        rerankeados = [by_rowid[rid] for rid in orden if rid in by_rowid]
        con_emb = {r[0] for r in emb_rows}
        cola = [r for r in rows if r[0] not in con_emb]
        return (rerankeados + cola)[:top_k]
    except Exception as e:
        logger.exception("Reranker fallo, cae a FTS5: %s", e)
        return rows[:top_k]


def _promote_preview_hits(conn: sqlite3.Connection, rows: list[tuple]) -> int:
    if not rows or MAX_PROMOTIONS_PER_QUERY <= 0:
        return 0
    seen: set[str] = set()
    to_promote: list[str] = []
    for _rowid, path, _fn, page, _snip in rows:
        if page == 0 and path not in seen:
            seen.add(path)
            to_promote.append(path)
            if len(to_promote) >= MAX_PROMOTIONS_PER_QUERY:
                break
    if not to_promote:
        return 0
    from rag.extractor import promote_to_full
    n = 0
    for path in to_promote:
        try:
            r = promote_to_full(conn, path)
            if r["status"] == "ok" and r["n_chunks"] > 0 and not r.get("ya_estaba"):
                n += 1
        except Exception as e:
            logger.warning("promote_to_full fallo para %s: %s", path, e)
    return n


def _buscar_sqlite(query: str, top_k: int) -> str:
    q = _sanitize_query(query)
    if not q:
        return "Sin resultados. Query vacia despues de sanitizar."
    pool = max(top_k, RERANK_POOL)
    try:
        conn = sqlite3.connect(INDEX_DB)

        def _fts5(limit: int) -> list[tuple]:
            return conn.execute(
                f"""
                SELECT rowid, path, filename, page,
                       snippet(chunks, 3, '<<', '>>', ' ... ', 20) AS snip
                FROM chunks
                WHERE chunks MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (q, limit),
            ).fetchall()

        rows = _fts5(pool)
        promoted = _promote_preview_hits(conn, rows)
        if promoted:
            logger.info("Promovidos %d archivos on-demand para query '%s'", promoted, query[:80])
            rows = _fts5(pool)
        if rows:
            rows = _rerank_rows(query, rows, conn, top_k)
        conn.close()
    except sqlite3.OperationalError as e:
        logger.error("FTS5 error: %s", e)
        return f"Error del index: {e}"
    if not rows:
        return "Sin resultados."
    lines = []
    for i, (_rowid, path, fn, page, snip) in enumerate(rows, start=1):
        marker = " (preview)" if page == 0 else f" (pag. {page})"
        lines.append(f"[{i}] {fn}{marker} - {snip}\n    path: {path}")
    return "\n".join(lines)


def _leer_pagina_sqlite(path: str, pagina: int) -> str:
    try:
        conn = sqlite3.connect(INDEX_DB)
        row = conn.execute(
            "SELECT text FROM chunks WHERE path = ? AND page = ?",
            (path, pagina),
        ).fetchone()
        conn.close()
    except sqlite3.OperationalError as e:
        return f"Error del index: {e}"
    if not row:
        return f"No hay pagina {pagina} indexada para {path}."
    return row[0]


def _leer_rango_sqlite(path: str, pagina_inicio: int, pagina_fin: int) -> str:
    try:
        conn = sqlite3.connect(INDEX_DB)
        rows = conn.execute(
            "SELECT page, text FROM chunks WHERE path = ? AND page BETWEEN ? AND ? ORDER BY page",
            (path, pagina_inicio, pagina_fin),
        ).fetchall()
        conn.close()
    except sqlite3.OperationalError as e:
        return f"Error del index: {e}"
    if not rows:
        return f"Sin paginas en el rango {pagina_inicio}-{pagina_fin} para {path}."
    return "\n\n".join(f"[Pagina {pg}]\n{txt}" for pg, txt in rows)


def _stats_sqlite() -> dict:
    try:
        conn = sqlite3.connect(INDEX_DB)
        n_files = conn.execute("SELECT COUNT(*) FROM files WHERE status='ok'").fetchone()[0]
        n_err = conn.execute("SELECT COUNT(*) FROM files WHERE status='error'").fetchone()[0]
        n_sin_texto = conn.execute("SELECT COUNT(*) FROM files WHERE status='sin_texto'").fetchone()[0]
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        try:
            n_emb = conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
        except sqlite3.OperationalError:
            n_emb = 0
        conn.close()
        return {"files_ok": n_files, "files_err": n_err, "files_sin_texto": n_sin_texto,
                "chunks": n_chunks, "embeddings": n_emb, "db_path": INDEX_DB}
    except sqlite3.OperationalError:
        return {"files_ok": 0, "files_err": 0, "files_sin_texto": 0,
                "chunks": 0, "embeddings": 0, "db_path": INDEX_DB}


# =============== API PUBLICA (dispatch) ===============

def buscar_en_documentos(query: str, top_k: int = 5, prev_context: dict | None = None) -> str:
    """Busca en la base interna del estudio (RAG) con hybrid retrieval.

    Usa esta tool ANTES de responder cualquier pregunta sobre polizas, contratos, expedientes,
    fallos internos o cualquier documento propio del estudio Montoya-Gherzi.

    La query se analiza para extraer automaticamente: nombre de carpeta ('en la carpeta X'),
    numeros de expediente ('N° 12345', 'FMZ 25156/2024'), nombres propios en mayusculas
    (BENAVIDEZ), e intencion (caso vs concepto). Estos se aplican como filtros/boost
    ademas del semantic search.

    Args:
        query: Palabras clave o pregunta en lenguaje natural. Si conoces la carpeta o
            numero de expediente, mencionalos explicitos ('carpeta ARABELA', 'N° 170495').
        top_k: Cantidad de resultados (default 5, max 10).
        prev_context: opcional. Dict con {carpeta, expedientes, nombres_propios} del
            turno anterior para heredar en follow-ups ('y el segundo?', 'ampliá').
    """
    top_k = max(1, min(top_k, 10))
    ctx = None
    if prev_context:
        from rag.query_analyzer import QueryContext
        ctx = QueryContext(
            carpeta=prev_context.get("carpeta"),
            expedientes=list(prev_context.get("expedientes") or []),
            nombres_propios=list(prev_context.get("nombres_propios") or []),
        )
    if RAG_BACKEND == "qdrant":
        return _buscar_qdrant(query, top_k, prev_context=ctx)
    return _buscar_sqlite(query, top_k)


def leer_pagina_documento(path: str, pagina: int) -> str:
    """Devuelve el texto completo de una pagina especifica de un documento ya indexado."""
    if RAG_BACKEND == "qdrant":
        return _leer_pagina_qdrant(path, pagina)
    return _leer_pagina_sqlite(path, pagina)


def leer_rango_documento(path: str, pagina_inicio: int, pagina_fin: int) -> str:
    """Devuelve el texto de un rango de paginas. Cap 20 paginas por llamada."""
    if pagina_fin - pagina_inicio > 19:
        pagina_fin = pagina_inicio + 19
    if RAG_BACKEND == "qdrant":
        return _leer_rango_qdrant(path, pagina_inicio, pagina_fin)
    return _leer_rango_sqlite(path, pagina_inicio, pagina_fin)


def stats_index() -> dict:
    """Reporta stats del backend activo."""
    if RAG_BACKEND == "qdrant":
        return {"backend": "qdrant", **_stats_qdrant()}
    return {"backend": "sqlite", **_stats_sqlite()}


def demo() -> None:
    """Self-check del backend SQLite. Qdrant se testea en qdrant_backend.demo."""
    import tempfile
    from pathlib import Path

    os.environ["RERANKER_MOCK"] = "1"
    os.environ["USE_RERANKER"] = "1"
    os.environ["RAG_BACKEND"] = "sqlite"
    from rag import reranker
    reranker._encoder = None
    from rag.extractor import index_root

    global INDEX_DB, RAG_BACKEND
    RAG_BACKEND = "sqlite"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "poliza_zurich_2023.txt"
        f.write_text(
            "POLIZA de seguro. Clausula de exclusion por alcoholemia superior a 0.5 g/l. "
            "El asegurado Juan Perez conducia en estado de ebriedad.",
            encoding="utf-8",
        )
        db_path = str(Path(d) / "index.db")
        index_root(d, db_path, workers=1)

        INDEX_DB = db_path
        r = buscar_en_documentos("alcoholemia Perez")
        assert "poliza_zurich_2023" in r, f"esperaba match, got: {r}"

        r2 = leer_pagina_documento(str(f), 1)
        assert "alcoholemia" in r2.lower(), f"pagina no leida: {r2}"

        s = stats_index()
        assert s["backend"] == "sqlite" and s["files_ok"] == 1, s

    print("search.demo OK")


if __name__ == "__main__":
    demo()
