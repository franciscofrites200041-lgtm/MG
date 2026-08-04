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

import logging
import os
import re
import sqlite3

logger = logging.getLogger("rag.search")

RAG_BACKEND = os.getenv("RAG_BACKEND", "sqlite").lower()
INDEX_DB = os.getenv("INDEX_DB_PATH", "./index.db")
NAS_ROOT = os.getenv("NAS_ROOT", "")
RERANK_POOL = int(os.getenv("RERANK_POOL", "20"))
QDRANT_SEARCH_LIMIT = int(os.getenv("QDRANT_SEARCH_LIMIT", "20"))
MAX_PROMOTIONS_PER_QUERY = int(os.getenv("MAX_PROMOTIONS_PER_QUERY", "5"))

_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]{2,}")


def _sanitize_query(q: str) -> str:
    tokens = _TOKEN_RE.findall(q or "")
    return " ".join(tokens)


# =============== QDRANT PATH ===============

def _buscar_qdrant(query: str, top_k: int) -> str:
    """Semantic search puro sobre Qdrant + rerank opcional sobre top-N."""
    from rag import qdrant_backend, reranker

    try:
        qvec = reranker.embed_query(query).tolist()
    except Exception as e:
        logger.exception("No pude embeddear query: %s", e)
        return f"Error del reranker: {e}"

    pool = max(top_k, QDRANT_SEARCH_LIMIT)
    try:
        hits = qdrant_backend.search(qvec, limit=pool)
    except Exception as e:
        logger.exception("Qdrant search fallo: %s", e)
        return f"Error del index Qdrant: {e}"

    if not hits:
        return "Sin resultados."

    # ponytail: Qdrant ya devuelve por cosine, el rerank explicito con MiniLM sobre
    # el mismo modelo seria redundante. Se deja hook por si mas adelante se usa un
    # cross-encoder pesado (bge-reranker) que si aporta calidad extra al reordenar.
    if reranker.is_enabled() and os.getenv("RERANK_CROSS_ENCODER") == "1":
        try:
            hits = _cross_encoder_rerank(query, hits)
        except Exception as e:
            logger.warning("cross-encoder rerank fallo, cae a orden Qdrant: %s", e)

    hits = hits[:top_k]

    lines = []
    for i, h in enumerate(hits, start=1):
        marker = f" (pag. {h.get('page', '?')})"
        snippet = (h.get("snippet") or "")[:300]
        lines.append(f"[{i}] {h.get('filename', '?')}{marker} - {snippet}\n    path: {h.get('path')}")
    return "\n".join(lines)


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

def buscar_en_documentos(query: str, top_k: int = 5) -> str:
    """Busca en la base interna del estudio (RAG). Devuelve los N chunks mas relevantes con path y pagina.

    Usa esta tool ANTES de responder cualquier pregunta sobre polizas, contratos, expedientes,
    fallos internos o cualquier documento propio del estudio Montoya-Gherzi.

    Args:
        query: Palabras clave o pregunta en lenguaje natural.
        top_k: Cantidad de resultados (default 5, max 10).
    """
    top_k = max(1, min(top_k, 10))
    if RAG_BACKEND == "qdrant":
        return _buscar_qdrant(query, top_k)
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
