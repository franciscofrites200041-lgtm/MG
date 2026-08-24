"""Worker paralelo: extrae, embedea y upsertea directo a Qdrant (sin JSONs).

Basado en bg_worker.py + upsert directo. 4 workers -> cada uno con su cliente
Qdrant HTTP. Sin drop-folder, sin drain, checkpoint reanudable.

Uso:
    python scripts/bg_worker_qdrant_parallel.py \
        --source-root C:\\Publico \
        --virtual-root /volume1/Publico/Estudio \
        --qdrant-url http://localhost:6333 \
        --collection mg_docs \
        --workers 4 \
        --checkpoint C:\\Users\\franc\\.claude\\jobs\\e21051fc\\worker.checkpoint

Cada worker crea su propio QdrantClient (multiprocessing spawn en Windows).
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("RAG_BACKEND", "qdrant")

from rag.extractor import walk_files, process_one  # noqa: E402

logger = logging.getLogger("bg_worker_qdrant_parallel")


def _load_checkpoint(cp: Path) -> set[str]:
    if not cp.exists():
        return set()
    return {ln.strip() for ln in cp.read_text(encoding="utf-8").splitlines() if ln.strip()}


def _append_checkpoint(cp: Path, path: str) -> None:
    cp.parent.mkdir(parents=True, exist_ok=True)
    with cp.open("a", encoding="utf-8") as f:
        f.write(path + "\n")


def _map_to_virtual(local_path: str, source_root: str, virtual_root: str) -> str:
    rel = os.path.relpath(local_path, source_root).replace(os.sep, "/")
    return virtual_root.rstrip("/") + "/" + rel


def _worker(args_tuple):
    """Corre en un worker process. Extrae + embed + upsert. Devuelve (path, status, n_chunks, err)."""
    local_path, source_root, virtual_root, qurl, collection = args_tuple

    # Cada proceso setea sus vars antes de importar qdrant_backend (que lee env).
    os.environ["QDRANT_URL"] = qurl
    os.environ["QDRANT_COLLECTION"] = collection

    from rag import qdrant_backend
    from rag.qdrant_backend import ChunkPoint

    try:
        doc = process_one(local_path)
        if doc.status != "ok" or not doc.chunks:
            return (local_path, doc.status, 0, doc.error)

        from rag.reranker import embed_texts
        texts = [txt for _pg, txt in doc.chunks]
        vecs = embed_texts(texts)

        virtual = _map_to_virtual(local_path, source_root, virtual_root)
        seen_idx: dict[int, int] = {}
        points = []
        for (page, text), vec in zip(doc.chunks, vecs):
            idx = seen_idx.get(page, 0)
            seen_idx[page] = idx + 1
            points.append(ChunkPoint(
                path=virtual, filename=doc.filename, ext=doc.ext,
                page=page, chunk_idx=idx, snippet=text[:800],
                mtime=doc.mtime, size=doc.size,
                vector=vec.astype("float32").tolist(),
            ))
        n = qdrant_backend.upsert_chunks(points, batch_size=128)
        return (local_path, "ok", n, None)
    except Exception as e:
        return (local_path, "err_worker", 0, f"{type(e).__name__}: {e}"[:300])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", required=True)
    p.add_argument("--virtual-root", required=True,
                   help="Prefijo con el que el bot cita paths (ej: /volume1/Publico/Estudio)")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--collection", default="mg_docs")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    os.environ["QDRANT_URL"] = args.qdrant_url
    os.environ["QDRANT_COLLECTION"] = args.collection
    from rag import qdrant_backend  # main process ensure_collection una vez
    qdrant_backend.ensure_collection()

    source_root = str(Path(args.source_root).resolve())
    cp_path = Path(args.checkpoint)
    procesados = _load_checkpoint(cp_path)
    logger.info("Checkpoint: %d ya procesados", len(procesados))

    todos: list[str] = []
    for p_local in walk_files(source_root):
        s = str(p_local)
        if s in procesados:
            continue
        todos.append(s)
        if args.limit and len(todos) >= args.limit:
            break
    logger.info("A procesar: %d archivos", len(todos))

    if not todos:
        logger.info("Nada que hacer.")
        return

    tarea = [(pth, source_root, args.virtual_root, args.qdrant_url, args.collection) for pth in todos]
    ok = err = sin_texto = 0
    total_points = 0
    t0 = time.time()

    with mp.Pool(processes=args.workers) as pool:
        for i, (pth, status, n, e) in enumerate(pool.imap_unordered(_worker, tarea, chunksize=4), start=1):
            _append_checkpoint(cp_path, pth)
            if status == "ok":
                ok += 1
                total_points += n
            elif status == "sin_texto":
                sin_texto += 1
            else:
                err += 1
                if err <= 20:
                    logger.warning("Err (%s): %s -> %s", status, pth, e)
            if i % 50 == 0:
                dt = time.time() - t0
                rate = i / dt if dt > 0 else 0
                eta_h = (len(todos) - i) / rate / 3600 if rate > 0 else 0
                logger.info(
                    "Progreso %d/%d ok=%d err=%d sin_texto=%d puntos=%d rate=%.1f/s ETA=%.1fh",
                    i, len(todos), ok, err, sin_texto, total_points, rate, eta_h,
                )

    dt = time.time() - t0
    logger.info(
        "Terminado en %.1fs: ok=%d err=%d sin_texto=%d puntos_upsertados=%d",
        dt, ok, err, sin_texto, total_points,
    )


if __name__ == "__main__":
    main()
