"""Exporta index.db (SQLite) -> Qdrant local.

Corre en la PC despues del procesamiento full con bg_worker + drain_pending.
Lee files + chunks + chunk_embeddings, mapea a puntos Qdrant con payload
{path, filename, ext, page, chunk_idx, snippet, mtime, size} y hace upsert
en batches al Qdrant local (Docker).

Requisitos:
    - Qdrant corriendo en localhost:6333 (docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant)
    - index.db con embeddings poblados

Uso:
    python scripts/export_to_qdrant.py \\
        --db ./data/index.db \\
        --qdrant-url http://localhost:6333 \\
        --collection mg_docs \\
        --batch 512

Al terminar, se puede sacar snapshot con:
    curl -X POST http://localhost:6333/collections/mg_docs/snapshots
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("RAG_BACKEND", "qdrant")

logger = logging.getLogger("export_to_qdrant")


def _chunk_idx_from_row(rowid: int, path: str, page: int, seen: dict[tuple[str, int], int]) -> int:
    """Cada (path, page) puede tener varios chunks. Le asignamos chunk_idx incremental
    en el orden en que aparecen en el DB (rowid ASC)."""
    key = (path, page)
    idx = seen.get(key, 0)
    seen[key] = idx + 1
    return idx


def export(db_path: str, qdrant_url: str, api_key: str | None,
           collection: str, batch: int, limit: int | None) -> None:
    os.environ["QDRANT_URL"] = qdrant_url
    if api_key:
        os.environ["QDRANT_API_KEY"] = api_key
    os.environ["QDRANT_COLLECTION"] = collection

    from rag import qdrant_backend
    from rag.qdrant_backend import ChunkPoint

    qdrant_backend.ensure_collection()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Total para reporte de progreso
    total = conn.execute(
        "SELECT COUNT(*) FROM chunks c JOIN chunk_embeddings e ON e.chunk_rowid = c.rowid"
    ).fetchone()[0]
    if limit:
        total = min(total, limit)
    logger.info("A exportar: %d chunks con embedding", total)

    # Payload de files para enriquecer con ext/size
    files_meta: dict[str, sqlite3.Row] = {}
    for r in conn.execute("SELECT path, ext, mtime, size FROM files"):
        files_meta[r["path"]] = r

    query = """
        SELECT c.rowid, c.path, c.filename, c.page, c.text, e.vec, e.dim
        FROM chunks c
        JOIN chunk_embeddings e ON e.chunk_rowid = c.rowid
        WHERE c.page > 0
        ORDER BY c.path, c.page, c.rowid
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    import numpy as np

    seen_chunk_idx: dict[tuple[str, int], int] = {}
    buffer: list[ChunkPoint] = []
    exportados = 0
    saltados = 0
    t0 = time.time()

    for r in conn.execute(query):
        path = r["path"]
        meta = files_meta.get(path)
        if not meta:
            saltados += 1
            continue

        vec = np.frombuffer(r["vec"], dtype=np.float32).tolist()
        if len(vec) != qdrant_backend.VECTOR_SIZE:
            saltados += 1
            continue

        idx = _chunk_idx_from_row(r["rowid"], path, r["page"], seen_chunk_idx)
        snippet = (r["text"] or "")[:800]  # cap para no inflar Qdrant

        buffer.append(ChunkPoint(
            path=path,
            filename=r["filename"],
            ext=meta["ext"],
            page=r["page"],
            chunk_idx=idx,
            snippet=snippet,
            mtime=meta["mtime"],
            size=meta["size"],
            vector=vec,
        ))

        if len(buffer) >= batch:
            n = qdrant_backend.upsert_chunks(buffer, batch_size=batch)
            exportados += n
            buffer = []
            dt = time.time() - t0
            rate = exportados / dt if dt > 0 else 0
            eta_min = (total - exportados) / rate / 60 if rate > 0 else 0
            logger.info("Exportados %d/%d (%.0f/s) ETA %.1f min saltados=%d",
                        exportados, total, rate, eta_min, saltados)

    if buffer:
        n = qdrant_backend.upsert_chunks(buffer, batch_size=batch)
        exportados += n

    conn.close()
    dt = time.time() - t0
    logger.info("Fin. Exportados=%d saltados=%d en %.1fs", exportados, saltados, dt)

    stats = qdrant_backend.stats()
    logger.info("Coleccion final: %s", stats)


def snapshot(qdrant_url: str, collection: str, out_dir: str) -> str:
    """Dispara un snapshot de la coleccion y descarga el .tar a out_dir. Devuelve el path."""
    import httpx

    with httpx.Client(timeout=1800) as client:
        resp = client.post(f"{qdrant_url}/collections/{collection}/snapshots")
        resp.raise_for_status()
        data = resp.json()["result"]
        snap_name = data["name"]
        logger.info("Snapshot creado: %s (size=%d bytes)", snap_name, data.get("size", 0))

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / snap_name

        url = f"{qdrant_url}/collections/{collection}/snapshots/{snap_name}"
        logger.info("Descargando snapshot a %s", dest)
        with client.stream("GET", url) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
        logger.info("Snapshot descargado: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
        return str(dest)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="./data/index.db")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--api-key", default=None)
    p.add_argument("--collection", default="mg_docs")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--limit", type=int, default=None, help="Cortar tras N chunks (debug)")
    p.add_argument("--snapshot-dir", default=None, help="Si se pasa, dispara snapshot post-export y baja el .tar")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    export(args.db, args.qdrant_url, args.api_key, args.collection, args.batch, args.limit)

    if args.snapshot_dir:
        snapshot(args.qdrant_url, args.collection, args.snapshot_dir)


if __name__ == "__main__":
    main()
