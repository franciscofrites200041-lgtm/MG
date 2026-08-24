"""Procesa archivos del NAS on-demand sin mirror local persistente.

Flujo:
    1. rclone lsjson nas:Publico --recursive -> lista (path, mtime, size)
    2. Filtra: check en Qdrant si (virtual_path, mtime) ya existe -> skip
    3. Para los faltantes: baja archivo a /tmp/one_by_one, procesa, borra
    4. Sin --max-age: procesa cualquier archivo nuevo/no-indexado

Uso:
    docker exec mg-gateway python /app/scripts/process_nas_ondemand.py

Cron cada 6h (host):
    0 */6 * * * docker exec mg-gateway python /app/scripts/process_nas_ondemand.py >> /data/logs/nas_ondemand.log 2>&1

Requisitos en el container gateway:
    - rclone binary
    - rclone.conf montado en /root/.config/rclone/rclone.conf (o pasarlo via env)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("RAG_BACKEND", "qdrant")

logger = logging.getLogger("nas_ondemand")

REMOTE = os.getenv("RCLONE_REMOTE", "nas")
SRC = os.getenv("RCLONE_SOURCE_PATH", "Publico")
VIRTUAL_ROOT = os.getenv("VIRTUAL_ROOT", "/volume1/Publico/Estudio")
TMP_DIR = Path(os.getenv("TMP_DIR", "/tmp/nas_ondemand"))
EXCLUDE_PREFIXES = ("Publico.zip", "#recycle/", "#snapshot/", "@eaDir/", ".DS_Store", "Thumbs.db")
SUPPORTED_EXT = {".pdf", ".docx", ".doc", ".txt"}


def _rclone_list(remote_path: str) -> list[dict]:
    """rclone lsjson recursivo. Devuelve [{Path, ModTime, Size, IsDir}]."""
    logger.info("rclone lsjson %s (puede tardar unos minutos)...", remote_path)
    t0 = time.time()
    r = subprocess.run(
        ["rclone", "lsjson", remote_path, "--recursive", "--files-only", "--fast-list"],
        capture_output=True, text=True, timeout=1800,
    )
    if r.returncode != 0:
        raise RuntimeError(f"rclone lsjson fallo: {r.stderr[:500]}")
    data = json.loads(r.stdout)
    logger.info("rclone lsjson: %d archivos en %.1fs", len(data), time.time() - t0)
    return data


def _is_supported(path: str) -> bool:
    p = path.lower()
    if any(prefix.lower() in p.lower() for prefix in EXCLUDE_PREFIXES):
        return False
    ext = "." + p.rsplit(".", 1)[-1] if "." in p else ""
    return ext in SUPPORTED_EXT


def _mtime_epoch(mod_time: str) -> float:
    """rclone ModTime: '2025-01-15T10:30:00.000000000-03:00'. Devuelve epoch."""
    from datetime import datetime
    try:
        # Normalizar +00:00 vs -03:00 y ns opcionales
        ts = mod_time.split(".")[0]
        tz = ""
        for sep in ("+", "-"):
            if sep in mod_time[10:]:
                tz = mod_time[10:][mod_time[10:].rindex(sep):]
                break
        if tz:
            ts = ts + tz
        dt = datetime.fromisoformat(ts)
        return dt.timestamp()
    except Exception:
        return 0.0


def _virtual_path(rel_path: str) -> str:
    return VIRTUAL_ROOT.rstrip("/") + "/" + rel_path.replace("\\", "/")


def _needs_process(virtual: str, mtime: float) -> bool:
    from rag import qdrant_backend
    try:
        return not qdrant_backend.exists_by_path_mtime(virtual, mtime)
    except Exception as e:
        logger.warning("qdrant check %s fallo: %s", virtual, str(e)[:80])
        return True  # ante duda, procesar


def _rclone_download(remote_path: str, dest: Path) -> bool:
    r = subprocess.run(
        ["rclone", "copyto", remote_path, str(dest), "--timeout", "5m"],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode != 0:
        logger.warning("download fallo %s: %s", remote_path, r.stderr[:200])
        return False
    return dest.exists() and dest.stat().st_size > 0


def _process_and_upsert(tmp_file: Path, virtual_path: str, mtime: float, size: int) -> str:
    from rag.extractor import process_one
    from rag import qdrant_backend
    from rag.qdrant_backend import ChunkPoint
    from rag.reranker import embed_texts, MODEL_NAME

    doc = process_one(str(tmp_file))
    if doc.status != "ok" or not doc.chunks:
        return doc.status

    texts = [txt for _pg, txt in doc.chunks]
    vecs = embed_texts(texts)
    seen_idx: dict[int, int] = {}
    points = []
    for (page, text), vec in zip(doc.chunks, vecs):
        idx = seen_idx.get(page, 0)
        seen_idx[page] = idx + 1
        points.append(ChunkPoint(
            path=virtual_path,
            filename=os.path.basename(virtual_path),
            ext="." + virtual_path.rsplit(".", 1)[-1],
            page=page, chunk_idx=idx,
            snippet=text[:800],
            mtime=mtime, size=size,
            vector=vec.astype("float32").tolist(),
        ))
    qdrant_backend.upsert_chunks(points, batch_size=64)
    return "ok"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="Cortar tras N archivos (debug)")
    p.add_argument("--dry-run", action="store_true", help="Solo listar candidatos, no bajar")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Listar NAS
    remote_path = f"{REMOTE}:{SRC}"
    try:
        listing = _rclone_list(remote_path)
    except Exception as e:
        logger.error("rclone lsjson fallo: %s", e)
        return

    # 2. Filtrar procesables + no indexados
    logger.info("Filtrando contra Qdrant (%d archivos)...", len(listing))
    t0 = time.time()
    candidates: list[tuple[str, str, float, int]] = []  # (rel_path, virtual, mtime, size)
    checked = 0
    for entry in listing:
        rel = entry["Path"]
        if not _is_supported(rel):
            continue
        mtime = _mtime_epoch(entry.get("ModTime", ""))
        size = int(entry.get("Size", 0))
        virtual = _virtual_path(rel)
        checked += 1
        if _needs_process(virtual, mtime):
            candidates.append((rel, virtual, mtime, size))
        if checked % 5000 == 0:
            logger.info("  chequeados %d/%d -> %d faltantes", checked, len(listing), len(candidates))

    logger.info("Filter fin en %.1fs: %d chequeados, %d faltantes en Qdrant",
                time.time() - t0, checked, len(candidates))

    if args.limit:
        candidates = candidates[:args.limit]
        logger.info("--limit %d aplicado", args.limit)

    if args.dry_run:
        for rel, v, m, s in candidates[:20]:
            logger.info("  DRY: %s (%.0fMB)", rel, s / 1e6)
        logger.info("--dry-run: %d candidatos", len(candidates))
        return

    # 3. Procesar uno por uno
    ok = err = 0
    total = len(candidates)
    for i, (rel, virtual, mtime, size) in enumerate(candidates, 1):
        # Extension del temp segun original
        ext = "." + rel.rsplit(".", 1)[-1]
        tmp = TMP_DIR / f"{uuid.uuid4().hex}{ext}"
        try:
            if not _rclone_download(f"{remote_path}/{rel}", tmp):
                err += 1
                continue
            status = _process_and_upsert(tmp, virtual, mtime, size)
            if status == "ok":
                ok += 1
            else:
                err += 1
        except Exception as e:
            err += 1
            logger.warning("proceso %s fallo: %s", rel, str(e)[:150])
        finally:
            tmp.unlink(missing_ok=True)

        if i % 20 == 0:
            logger.info("  [%d/%d] ok=%d err=%d", i, total, ok, err)

    logger.info("Fin: ok=%d err=%d de %d candidatos", ok, err, total)


if __name__ == "__main__":
    main()
