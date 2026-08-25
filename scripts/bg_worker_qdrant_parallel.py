"""Worker paralelo resiliente: extrae, embedea y upsertea directo a Qdrant.

Refuerzos vs version anterior:
- Skip si el archivo ya esta en Qdrant (path + mtime) -> evita re-hacer trabajo
  cuando el checkpoint se perdio o se cambio de PC
- Retry con backoff exponencial (3 intentos por archivo) en errores transitorios
  de Qdrant / red / embed
- Checkpoint atomico: buffered en RAM + flush cada N archivos o cada M segundos,
  escrito con temp+rename para nunca quedar truncado
- Heartbeat: log cada 60s aunque no haya avance ("sigo vivo, procesando X")
- Signal handler: SIGINT/SIGTERM hacen flush del checkpoint antes de salir
- Progreso persistente en worker.progress.json para monitoreo externo
- Ignora archivos que colgaron previamente si aparecen en --skip-file

Uso:
    python scripts/bg_worker_qdrant_parallel.py \
        --source-root Z:\\Publico \
        --virtual-root /volume1/Publico/Estudio \
        --qdrant-url http://VPS_IP:6333 \
        --collection mg_docs \
        --workers 4 \
        --checkpoint E:\\MG_drop\\worker.checkpoint
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("RAG_BACKEND", "qdrant")

from rag.extractor import walk_files, process_one  # noqa: E402

logger = logging.getLogger("bg_worker_qdrant_parallel")

CHECKPOINT_FLUSH_EVERY_N = 50
CHECKPOINT_FLUSH_EVERY_SEC = 30.0
HEARTBEAT_EVERY_SEC = 60.0
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2.0  # 2s, 4s, 8s


class Checkpoint:
    """Checkpoint atomico buffered.
    Formato: un path por linea, UTF-8. flush() escribe .tmp + rename."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        if path.exists():
            self._done = {
                ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
            }
        self._buffer: list[str] = []
        self._last_flush = time.time()

    def has(self, p: str) -> bool:
        return p in self._done

    def add(self, p: str) -> None:
        if p in self._done:
            return
        self._done.add(p)
        self._buffer.append(p)
        if (len(self._buffer) >= CHECKPOINT_FLUSH_EVERY_N or
                (time.time() - self._last_flush) >= CHECKPOINT_FLUSH_EVERY_SEC):
            self.flush()

    def flush(self) -> None:
        if not self._buffer and self.path.exists():
            return
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        # Escritura completa del set (idempotente y robusto vs append truncado)
        with tmp.open("w", encoding="utf-8") as f:
            for p in sorted(self._done):
                f.write(p + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self.path)
        self._buffer.clear()
        self._last_flush = time.time()

    def __len__(self) -> int:
        return len(self._done)


def _map_to_virtual(local_path: str, source_root: str, virtual_root: str) -> str:
    rel = os.path.relpath(local_path, source_root).replace(os.sep, "/")
    return virtual_root.rstrip("/") + "/" + rel


def _worker(args_tuple):
    """Corre en un worker process. Extrae + embed + upsert con retry.
    Devuelve (path, status, n_chunks, err).
    """
    local_path, source_root, virtual_root, qurl, collection = args_tuple

    os.environ["QDRANT_URL"] = qurl
    os.environ["QDRANT_COLLECTION"] = collection

    from rag import qdrant_backend
    from rag.qdrant_backend import ChunkPoint
    from rag.reranker import embed_texts

    # 1. Skip fuerte: si ya esta en Qdrant con este mtime, no re-hacer
    try:
        st = os.stat(local_path)
        virtual = _map_to_virtual(local_path, source_root, virtual_root)
        if qdrant_backend.exists_by_path_mtime(virtual, st.st_mtime):
            return (local_path, "skip_already_indexed", 0, None)
    except Exception:
        pass  # si falla el check, seguimos e intentamos procesar

    last_err = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            doc = process_one(local_path)
            if doc.status == "sin_texto" or (doc.status == "ok" and not doc.chunks):
                return (local_path, "sin_texto", 0, None)
            if doc.status != "ok":
                return (local_path, doc.status, 0, doc.error)

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
            last_err = f"{type(e).__name__}: {e}"[:300]
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                continue
            # ultimo intento fallo
            return (local_path, "err_worker", 0, last_err + f" (attempts={attempt})")
    return (local_path, "err_worker", 0, last_err or "unknown")


def _load_skip_file(p: Path | None) -> set[str]:
    if not p or not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}


def _write_progress(path: Path, data: dict) -> None:
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        logger.debug("progress write fallo: %s", e)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", required=True)
    p.add_argument("--virtual-root", required=True,
                   help="Prefijo con el que el bot cita paths (ej: /volume1/Publico/Estudio)")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--collection", default="mg_docs")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--skip-file", default=None,
                   help="Archivo con paths a ignorar (uno por linea). Uso: PDFs corruptos que cuelgan.")
    p.add_argument("--progress-file", default=None,
                   help="JSON de progreso escrito cada heartbeat. Default: <checkpoint>.progress.json")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    os.environ["QDRANT_URL"] = args.qdrant_url
    os.environ["QDRANT_COLLECTION"] = args.collection
    from rag import qdrant_backend
    qdrant_backend.ensure_collection()

    source_root = str(Path(args.source_root).resolve())
    cp_path = Path(args.checkpoint)
    checkpoint = Checkpoint(cp_path)
    skip_paths = _load_skip_file(Path(args.skip_file) if args.skip_file else None)
    progress_path = Path(args.progress_file) if args.progress_file else cp_path.with_suffix(".progress.json")

    logger.info("Checkpoint: %d ya procesados", len(checkpoint))
    if skip_paths:
        logger.info("Skip file: %d paths a ignorar", len(skip_paths))

    todos: list[str] = []
    for p_local in walk_files(source_root):
        s = str(p_local)
        if checkpoint.has(s) or s in skip_paths:
            continue
        todos.append(s)
        if args.limit and len(todos) >= args.limit:
            break
    logger.info("A procesar: %d archivos", len(todos))

    if not todos:
        logger.info("Nada que hacer.")
        return

    # Signal handler para flush limpio en Ctrl+C / SIGTERM
    stop_flag = {"stop": False}

    def _handle_signal(signum, _frame):
        logger.warning("Signal %d recibido, flushing checkpoint y saliendo...", signum)
        stop_flag["stop"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            pass  # Windows no soporta SIGTERM en todos los contextos

    tarea = [(pth, source_root, args.virtual_root, args.qdrant_url, args.collection) for pth in todos]
    ok = err = sin_texto = skipped = 0
    total_points = 0
    t0 = time.time()
    last_heartbeat = t0

    pool = mp.Pool(processes=args.workers)
    try:
        for i, (pth, status, n, e) in enumerate(pool.imap_unordered(_worker, tarea, chunksize=4), start=1):
            checkpoint.add(pth)
            if status == "ok":
                ok += 1
                total_points += n
            elif status == "sin_texto":
                sin_texto += 1
            elif status == "skip_already_indexed":
                skipped += 1
            else:
                err += 1
                if err <= 20 or err % 100 == 0:
                    logger.warning("Err (%s): %s -> %s", status, pth, e)

            now = time.time()
            if i % 50 == 0 or (now - last_heartbeat) >= HEARTBEAT_EVERY_SEC:
                dt = now - t0
                rate = i / dt if dt > 0 else 0
                eta_h = (len(todos) - i) / rate / 3600 if rate > 0 else 0
                logger.info(
                    "Progreso %d/%d ok=%d err=%d sin_texto=%d skip=%d pts=%d rate=%.1f/s ETA=%.1fh",
                    i, len(todos), ok, err, sin_texto, skipped, total_points, rate, eta_h,
                )
                _write_progress(progress_path, {
                    "ts": now, "processed": i, "total": len(todos),
                    "ok": ok, "err": err, "sin_texto": sin_texto, "skipped": skipped,
                    "points": total_points, "rate_per_s": round(rate, 2),
                    "eta_hours": round(eta_h, 2),
                })
                last_heartbeat = now

            if stop_flag["stop"]:
                logger.warning("Stop flag activo, terminando pool")
                pool.terminate()
                break
    finally:
        try:
            pool.close()
            pool.join()
        except Exception:
            pass
        checkpoint.flush()
        _write_progress(progress_path, {
            "ts": time.time(), "final": True,
            "ok": ok, "err": err, "sin_texto": sin_texto, "skipped": skipped,
            "points": total_points, "elapsed_s": round(time.time() - t0, 1),
        })

    dt = time.time() - t0
    logger.info(
        "Terminado en %.1fs: ok=%d err=%d sin_texto=%d skip=%d puntos=%d",
        dt, ok, err, sin_texto, skipped, total_points,
    )


def demo() -> None:
    """Self-check del Checkpoint atomico."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cp = Checkpoint(Path(d) / "test.cp")
        assert len(cp) == 0
        cp.add("a.pdf"); cp.add("b.pdf")
        assert cp.has("a.pdf") and cp.has("b.pdf")
        cp.add("a.pdf")  # dup
        assert len(cp) == 2
        cp.flush()
        # Reload
        cp2 = Checkpoint(Path(d) / "test.cp")
        assert cp2.has("a.pdf") and cp2.has("b.pdf")
        assert len(cp2) == 2
        # Nunca queda .tmp
        assert not (Path(d) / "test.cp.tmp").exists()
    print("bg_worker.demo OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        main()
