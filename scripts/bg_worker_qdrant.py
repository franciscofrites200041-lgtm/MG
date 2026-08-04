"""Worker de background para el VPS: procesa archivos nuevos del NAS mirror y los mete en Qdrant.

Flujo:
    1. Escanea RCLONE_LOCAL_MIRROR recursivo.
    2. Para cada archivo: si (path, mtime) ya esta en Qdrant -> skip.
    3. Si no: extrae texto (fitz/docx/etc), chunkea, embeddea, upsert Qdrant.
    4. Si el archivo cambio (path existe pero mtime distinto) -> delete_by_path + reindex.

Uso:
    python scripts/bg_worker_qdrant.py --root /data/nas_mirror --workers 2

Cron: cada 6h, despues del rclone sync.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("RAG_BACKEND", "qdrant")

from rag.extractor import walk_files, process_one  # noqa: E402
from rag import qdrant_backend  # noqa: E402
from rag.qdrant_backend import ChunkPoint  # noqa: E402

logger = logging.getLogger("bg_worker_qdrant")


def _map_to_bot_path(local_path: str, source_root: str, virtual_root: str) -> str:
    """Traduce path local /data/nas_mirror/foo/x.pdf -> /volume1/Publico/Estudio/foo/x.pdf.

    Asi el bot cita el path que el usuario reconoce, no el path interno del VPS.
    """
    rel = os.path.relpath(local_path, source_root).replace(os.sep, "/")
    return virtual_root.rstrip("/") + "/" + rel


def _needs_reindex(path: str, mtime: float) -> tuple[bool, bool]:
    """Devuelve (necesita_reindex, ya_existia_con_otro_mtime)."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

    c = qdrant_backend.get_client()
    # Mismo path + mismo mtime -> ya esta.
    if qdrant_backend.exists_by_path_mtime(path, mtime):
        return False, False
    # Mismo path pero cualquier mtime -> archivo cambio.
    q_filter = Filter(must=[FieldCondition(key="path", match=MatchValue(value=path))])
    result = c.count(collection_name=qdrant_backend.COLLECTION, count_filter=q_filter, exact=False)
    return True, result.count > 0


def _process_and_upsert(local_path: str, virtual_path: str) -> tuple[str, int]:
    """Devuelve (status, n_chunks_upserted)."""
    doc = process_one(local_path)
    if doc.status != "ok" or not doc.chunks:
        return doc.status, 0

    from rag.reranker import embed_texts
    texts = [txt for _pg, txt in doc.chunks]
    vecs = embed_texts(texts)

    # chunk_idx incremental por (path, page)
    seen_idx: dict[int, int] = {}
    points = []
    for (page, text), vec in zip(doc.chunks, vecs):
        idx = seen_idx.get(page, 0)
        seen_idx[page] = idx + 1
        points.append(ChunkPoint(
            path=virtual_path,
            filename=doc.filename,
            ext=doc.ext,
            page=page,
            chunk_idx=idx,
            snippet=text[:800],
            mtime=doc.mtime,
            size=doc.size,
            vector=vec.astype("float32").tolist(),
        ))

    n = qdrant_backend.upsert_chunks(points, batch_size=128)
    return "ok", n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.getenv("RCLONE_LOCAL_MIRROR", "/data/nas_mirror"))
    p.add_argument("--virtual-root", default=os.getenv("RCLONE_SOURCE_PATH", "/volume1/Publico/Estudio"),
                   help="Prefijo con el que el bot cita paths (visible al usuario)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=1,
                   help="Serial por default; subir solo si CPU sobra")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    qdrant_backend.ensure_collection()

    logger.info("Escaneando %s...", args.root)
    todos = list(walk_files(args.root))
    if args.limit:
        todos = todos[:args.limit]
    logger.info("Encontrados %d archivos totales", len(todos))

    nuevos = 0
    actualizados = 0
    skipped = 0
    errores = 0
    t0 = time.time()

    for i, local_path in enumerate(todos, start=1):
        try:
            st = os.stat(local_path)
            virtual = _map_to_bot_path(str(local_path), args.root, args.virtual_root)
            necesita, existia = _needs_reindex(virtual, st.st_mtime)
            if not necesita:
                skipped += 1
                continue
            if existia:
                qdrant_backend.delete_by_path(virtual)

            status, n = _process_and_upsert(str(local_path), virtual)
            if status != "ok":
                errores += 1
            elif existia:
                actualizados += 1
            else:
                nuevos += 1
        except Exception as e:
            logger.exception("Fallo %s: %s", local_path, e)
            errores += 1

        if i % 20 == 0:
            dt = time.time() - t0
            rate = i / dt if dt > 0 else 0
            logger.info("Progreso %d/%d nuevos=%d act=%d skip=%d err=%d (%.1f/s)",
                        i, len(todos), nuevos, actualizados, skipped, errores, rate)

    dt = time.time() - t0
    logger.info("Fin en %.1fs: nuevos=%d actualizados=%d skipped=%d errores=%d",
                dt, nuevos, actualizados, skipped, errores)


if __name__ == "__main__":
    main()
