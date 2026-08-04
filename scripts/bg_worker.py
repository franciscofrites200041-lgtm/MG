"""Worker de background: procesa full-extraction en la PC y escribe JSONs a un drop-folder.

Uso:
    python scripts/bg_worker.py \
        --source-root C:\\mg_nas_local \
        --nas-root /volume1/Publico \
        --drop-dir \\\\100.120.66.105\\Publico\\.mg_bot\\pending \
        --workers 8 \
        --checkpoint ./data/bg_worker.checkpoint

- source-root: carpeta local en el SSD con la copia del NAS (rclone).
- nas-root: el prefijo de path que usara el bot en produccion (para que match
  con los paths ya indexados en modo lite). Reemplaza source-root en los paths
  antes de serializar a JSON.
- drop-dir: carpeta compartida al NAS donde se van dropeando los JSONs.
- checkpoint: archivo local con paths ya procesados (append-only). Reanudable.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import uuid
from pathlib import Path

# Import lazy del extractor: agregar bot/ al sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.extractor import EXTRACTORS, walk_files, process_one  # noqa: E402

logger = logging.getLogger("bg_worker")


def _load_checkpoint(cp_path: Path) -> set[str]:
    if not cp_path.exists():
        return set()
    return {line.strip() for line in cp_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _append_checkpoint(cp_path: Path, path: str) -> None:
    cp_path.parent.mkdir(parents=True, exist_ok=True)
    with cp_path.open("a", encoding="utf-8") as f:
        f.write(path + "\n")


def _map_to_nas_path(local_path: str, source_root: str, nas_root: str) -> str:
    """Traduce C:\\mg_nas_local\\ANDY-1\\file.pdf -> /volume1/Publico/ANDY-1/file.pdf."""
    rel = os.path.relpath(local_path, source_root)
    # Normalizar separadores a POSIX (el bot en el NAS corre Linux).
    rel_posix = rel.replace(os.sep, "/")
    nas = nas_root.rstrip("/") + "/" + rel_posix
    return nas


def _process_and_serialize(args_tuple: tuple[str, str, str, bool]) -> dict:
    """Corre en un worker process. Devuelve dict con el resultado listo a serializar."""
    local_path, source_root, nas_root, do_embed = args_tuple
    doc = process_one(local_path)
    nas_path = _map_to_nas_path(doc.path, source_root, nas_root)
    payload: dict = {
        "path": nas_path,
        "filename": doc.filename,
        "ext": doc.ext,
        "mtime": doc.mtime,
        "size": doc.size,
        "status": doc.status,
        "error": doc.error,
        "chunks": [{"page": pg, "text": txt} for pg, txt in doc.chunks],
        "embeddings": [],
    }
    if do_embed and doc.status == "ok" and doc.chunks:
        try:
            from rag.reranker import MODEL_NAME, embed_texts
            texts = [txt for _pg, txt in doc.chunks]
            vecs = embed_texts(texts)
            dim = int(vecs.shape[1])
            embeddings = []
            for (pg, _txt), vec in zip(doc.chunks, vecs):
                b = vec.astype("float32").tobytes()
                embeddings.append({
                    "page": pg,
                    "model": MODEL_NAME,
                    "dim": dim,
                    "vec_b64": base64.b64encode(b).decode("ascii"),
                })
            payload["embeddings"] = embeddings
        except Exception as e:
            payload["error"] = f"embed: {e}"[:500]
    payload["_local_path"] = local_path  # para el checkpoint, no se serializa al drop
    return payload


def _write_json_atomic(drop_dir: Path, payload: dict) -> Path:
    """Escribe {uuid}.json.tmp y hace rename atomico a .json. El drain solo lee .json."""
    drop_dir.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex
    tmp = drop_dir / f"{uid}.json.tmp"
    final = drop_dir / f"{uid}.json"
    to_dump = {k: v for k, v in payload.items() if not k.startswith("_")}
    tmp.write_text(json.dumps(to_dump, ensure_ascii=False), encoding="utf-8")
    tmp.replace(final)
    return final


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", required=True, help="Carpeta local con la copia del NAS")
    p.add_argument("--nas-root", required=True, help="Prefijo que usa el bot en produccion (ej: /volume1/Publico)")
    p.add_argument("--drop-dir", required=True, help="Carpeta destino para los JSONs (share del NAS)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--checkpoint", default="./data/bg_worker.checkpoint")
    p.add_argument("--no-embed", action="store_true", help="Omitir embeddings (mas rapido, sin reranker)")
    p.add_argument("--limit", type=int, default=None, help="Cortar tras N archivos (debug)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    source_root = Path(args.source_root).resolve()
    drop_dir = Path(args.drop_dir)
    cp_path = Path(args.checkpoint)
    procesados = _load_checkpoint(cp_path)
    logger.info("Checkpoint tiene %d archivos ya procesados", len(procesados))

    do_embed = not args.no_embed

    # Armar la lista de archivos a procesar (respetando checkpoint).
    todos: list[str] = []
    for p_local in walk_files(str(source_root)):
        s = str(p_local)
        if s in procesados:
            continue
        todos.append(s)
        if args.limit and len(todos) >= args.limit:
            break
    logger.info("A procesar: %d archivos", len(todos))

    if not todos:
        logger.info("Nada que hacer. Bye.")
        return

    ok, err, sin_texto = 0, 0, 0
    tarea = [(pth, str(source_root), args.nas_root, do_embed) for pth in todos]

    t0 = time.time()
    with mp.Pool(processes=args.workers) as pool:
        for i, payload in enumerate(pool.imap_unordered(_process_and_serialize, tarea, chunksize=4), start=1):
            try:
                _write_json_atomic(drop_dir, payload)
                _append_checkpoint(cp_path, payload["_local_path"])
                if payload["status"] == "ok":
                    ok += 1
                elif payload["status"] == "sin_texto":
                    sin_texto += 1
                else:
                    err += 1
            except Exception as e:
                logger.exception("Fallo al escribir JSON para %s: %s", payload.get("path"), e)
                err += 1
            if i % 50 == 0:
                dt = time.time() - t0
                rate = i / dt if dt > 0 else 0
                eta_min = (len(todos) - i) / rate / 60 if rate > 0 else 0
                logger.info(
                    "Progreso: %d/%d (ok=%d err=%d sin_texto=%d) - %.1f arch/s - ETA %.1f min",
                    i, len(todos), ok, err, sin_texto, rate, eta_min,
                )

    dt = time.time() - t0
    logger.info(
        "Terminado en %.1fs: ok=%d err=%d sin_texto=%d total=%d",
        dt, ok, err, sin_texto, len(todos),
    )


if __name__ == "__main__":
    main()
