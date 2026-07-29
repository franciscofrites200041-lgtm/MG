"""Busqueda FTS5 con re-ranker semantico opcional, y lectura de paginas.

Pipeline por default (USE_RERANKER=1):
    query -> FTS5 top-N (RERANK_POOL, default 20)
          -> embed query + cosine con chunk_embeddings
          -> reordenar -> top_k final

Si USE_RERANKER=0 o no hay embeddings poblados, cae solo a FTS5.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3

logger = logging.getLogger("rag.search")

INDEX_DB = os.getenv("INDEX_DB_PATH", "./index.db")
NAS_ROOT = os.getenv("NAS_ROOT", "")  # opcional, solo para chequeo de existencia
RERANK_POOL = int(os.getenv("RERANK_POOL", "20"))

_TOKEN_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]{2,}")


def _sanitize_query(q: str) -> str:
    """FTS5 no tolera puntuación arbitraria. Extrae tokens y los une con espacio (AND implicito)."""
    tokens = _TOKEN_RE.findall(q or "")
    return " ".join(tokens)


def _rerank_rows(query_raw: str, rows: list[tuple], conn: sqlite3.Connection, top_k: int) -> list[tuple]:
    """Reordena rows por similitud semantica. Si no puede, devuelve los top_k originales."""
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
            logger.info("Sin embeddings poblados; devolviendo FTS5 puro.")
            return rows[:top_k]
        candidatos = [(rid, reranker.blob_to_vec(b)) for rid, b in emb_rows]
        qvec = reranker.embed_query(query_raw)
        orden = reranker.rerank_by_cosine(qvec, candidatos)
        by_rowid = {r[0]: r for r in rows}
        rerankeados = [by_rowid[rid] for rid in orden if rid in by_rowid]
        # Si sobraban rows sin embedding, los apendeo al final.
        con_emb = {r[0] for r in emb_rows}
        cola = [r for r in rows if r[0] not in con_emb]
        return (rerankeados + cola)[:top_k]
    except Exception as e:
        logger.exception("Reranker fallo, cae a FTS5: %s", e)
        return rows[:top_k]


def buscar_en_documentos(query: str, top_k: int = 5) -> str:
    """Busca en la base interna del estudio (RAG). Devuelve los N chunks mas relevantes con path y pagina.

    Usa esta tool ANTES de responder cualquier pregunta sobre polizas, contratos, expedientes,
    fallos internos o cualquier documento propio del estudio Montoya-Gherzi.

    El sistema combina busqueda lexica (FTS5) con re-ranker semantico basado en embeddings:
    la query puede usar sinonimos o parafrasis y aun asi encontrar chunks relevantes.

    Args:
        query: Palabras clave o pregunta en lenguaje natural. Ambos funcionan.
        top_k: Cantidad de resultados a devolver (default 5, max 10).

    Returns:
        Texto plano con los resultados en formato:
        [1] archivo.pdf (pag. 12) - snippet...
        [2] otro.docx (pag. 3) - snippet...
        O "Sin resultados." si la busqueda no devuelve nada.
    """
    q = _sanitize_query(query)
    if not q:
        return "Sin resultados. Query vacia despues de sanitizar."
    top_k = max(1, min(top_k, 10))
    pool = max(top_k, RERANK_POOL)

    try:
        conn = sqlite3.connect(INDEX_DB)
        rows = conn.execute(
            f"""
            SELECT rowid, path, filename, page,
                   snippet(chunks, 3, '<<', '>>', ' ... ', 20) AS snip
            FROM chunks
            WHERE chunks MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (q, pool),
        ).fetchall()

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
        lines.append(f"[{i}] {fn} (pag. {page}) - {snip}\n    path: {path}")
    return "\n".join(lines)


def leer_pagina_documento(path: str, pagina: int) -> str:
    """Devuelve el texto completo de una pagina especifica de un documento ya indexado.

    Usa esta tool cuando `buscar_en_documentos` te devolvio un resultado y necesitas
    leer el texto completo de esa pagina (no solo el snippet).

    Args:
        path: Ruta absoluta al documento tal como aparece en el resultado de busqueda.
        pagina: Numero de pagina (1-based).

    Returns:
        Texto completo de la pagina, o mensaje de error.
    """
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


def leer_rango_documento(path: str, pagina_inicio: int, pagina_fin: int) -> str:
    """Devuelve el texto de un rango de paginas de un documento indexado.

    Args:
        path: Ruta absoluta al documento.
        pagina_inicio: Primera pagina (1-based, inclusive).
        pagina_fin: Ultima pagina (inclusive). Maximo 20 paginas de una vez.

    Returns:
        Texto concatenado con separadores por pagina.
    """
    if pagina_fin - pagina_inicio > 19:
        pagina_fin = pagina_inicio + 19  # ponytail: hard cap para no volar el context
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


def stats_index() -> dict:
    """Reporta cuantos archivos, chunks y embeddings tiene el index."""
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
        return {
            "files_ok": n_files,
            "files_err": n_err,
            "files_sin_texto": n_sin_texto,
            "chunks": n_chunks,
            "embeddings": n_emb,
            "db_path": INDEX_DB,
        }
    except sqlite3.OperationalError:
        return {"files_ok": 0, "files_err": 0, "files_sin_texto": 0,
                "chunks": 0, "embeddings": 0, "db_path": INDEX_DB}


def demo() -> None:
    """Self-check contra un index temporal (reranker en MOCK)."""
    import tempfile
    from pathlib import Path
    from rag.extractor import index_root

    os.environ["RERANKER_MOCK"] = "1"
    os.environ["USE_RERANKER"] = "1"
    from rag import reranker

    reranker._encoder = None

    global INDEX_DB
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
        assert s["files_ok"] == 1 and s["embeddings"] >= 1, s

    print("search.demo OK")


if __name__ == "__main__":
    demo()
