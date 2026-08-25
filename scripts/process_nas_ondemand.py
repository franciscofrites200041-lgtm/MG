"""Procesa archivos del NAS on-demand. Stream por chunks de top-level dir.

Flujo:
    1. rclone lsf top-level dirs de nas:Publico -> N chunks
    2. Por cada chunk: rclone lsjson recursivo (timeout por chunk, no global)
    3. Filtro on-the-fly contra Qdrant via exists_by_path_mtime
    4. Baja + procesa + borra uno por uno

El streaming por chunks evita cargar 500k+ archivos en un solo lsjson (que reventaba
el timeout de 30 min con --fast-list). Cada chunk es un top-level dir independiente.

Uso:
    docker exec mg-gateway python /app/scripts/process_nas_ondemand.py [--preflight|--dry-run] [--limit N]

Cron cada hora (host) con flock para no apilar corridas:
    0 * * * * flock -n /tmp/nas_ondemand.lock docker exec mg-gateway python /app/scripts/process_nas_ondemand.py >> /data/logs/nas_ondemand.log 2>&1

Requisitos en el container gateway:
    - rclone binary
    - rclone.conf montado en /root/.config/rclone/rclone.conf
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
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("RAG_BACKEND", "qdrant")

logger = logging.getLogger("nas_ondemand")

REMOTE = os.getenv("RCLONE_REMOTE", "nas")
NAS_SHARE_PATH = os.getenv("NAS_SHARE_PATH", "Publico")
VIRTUAL_ROOT = os.getenv("VIRTUAL_ROOT", "/volume1/Publico/Estudio")
TMP_DIR = Path(os.getenv("TMP_DIR", "/tmp/nas_ondemand"))
CHUNK_TIMEOUT = int(os.getenv("RCLONE_CHUNK_TIMEOUT", "1800"))
EXCLUDE_PREFIXES = ("Publico.zip", "#recycle/", "#snapshot/", "@eaDir/", ".DS_Store", "Thumbs.db")
SUPPORTED_EXT = {".pdf", ".docx", ".doc", ".txt"}


def _rclone_list(remote_path: str) -> Iterator[dict]:
    """Stream por top-level dir. Timeout por chunk, no global."""
    logger.info("rclone lsf top-level dirs %s...", remote_path)
    r = subprocess.run(
        ["rclone", "lsf", remote_path, "--max-depth", "1", "--dirs-only"],
        capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        raise RuntimeError(f"rclone lsf dirs fallo: {r.stderr[:500]}")
    dirs = [d.rstrip("/") for d in r.stdout.strip().splitlines() if d.strip()]
    logger.info("  %d dirs top-level", len(dirs))

    try:
        r = subprocess.run(
            ["rclone", "lsjson", remote_path, "--files-only", "--max-depth", "1"],
            capture_output=True, text=True, timeout=CHUNK_TIMEOUT,
        )
        if r.returncode == 0 and r.stdout.strip():
            top = json.loads(r.stdout)
            if top:
                logger.info("  top-level files: %d", len(top))
                for entry in top:
                    yield entry
    except Exception as e:
        logger.warning("  top-level files fallo: %s", e)

    for i, d in enumerate(dirs, 1):
        subpath = f"{remote_path}/{d}"
        t0 = time.time()
        try:
            r = subprocess.run(
                ["rclone", "lsjson", subpath, "--recursive", "--files-only", "--fast-list"],
                capture_output=True, text=True, timeout=CHUNK_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            logger.warning("  [%d/%d] %s TIMEOUT >%ds", i, len(dirs), d, CHUNK_TIMEOUT)
            continue
        if r.returncode != 0:
            logger.warning("  [%d/%d] %s FAIL: %s", i, len(dirs), d, r.stderr[:200])
            continue
        try:
            chunk = json.loads(r.stdout)
        except json.JSONDecodeError as e:
            logger.warning("  [%d/%d] %s JSON parse fail: %s", i, len(dirs), d, e)
            continue
        logger.info("  [%d/%d] %s: %d files (%.1fs)", i, len(dirs), d, len(chunk), time.time() - t0)
        for entry in chunk:
            entry["Path"] = f"{d}/{entry['Path']}"
            yield entry


def _is_supported(path: str) -> bool:
    p = path.lower()
    if any(prefix.lower() in p for prefix in EXCLUDE_PREFIXES):
        return False
    ext = "." + p.rsplit(".", 1)[-1] if "." in p else ""
    return ext in SUPPORTED_EXT


def _mtime_epoch(mod_time: str) -> float:
    from datetime import datetime
    try:
        ts = mod_time.split(".")[0]
        tz = ""
        for sep in ("+", "-"):
            if sep in mod_time[10:]:
                tz = mod_time[10:][mod_time[10:].rindex(sep):]
                break
        if tz:
            ts = ts + tz
        return datetime.fromisoformat(ts).timestamp()
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
        return True


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
    from rag.reranker import embed_texts

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


def _preflight() -> bool:
    print("=== PREFLIGHT ===")
    all_ok = True
    try:
        r = subprocess.run(["rclone", "version"], capture_output=True, text=True, timeout=10)
        print(f"  [{'OK' if r.returncode==0 else 'FAIL'}] rclone: {r.stdout.splitlines()[0] if r.stdout else r.stderr[:100]}")
        if r.returncode != 0: all_ok = False
    except Exception as e:
        print(f"  [FAIL] rclone: {e}"); return False
    try:
        r = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True, timeout=10)
        if f"{REMOTE}:" in r.stdout.strip().splitlines():
            print(f"  [OK] remote '{REMOTE}'")
        else:
            print(f"  [FAIL] remote '{REMOTE}' no encontrado"); all_ok = False
    except Exception as e:
        print(f"  [FAIL] listremotes: {e}"); all_ok = False
    remote_path = f"{REMOTE}:{NAS_SHARE_PATH}"
    try:
        r = subprocess.run(["rclone", "lsf", remote_path, "--max-depth", "1"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            print(f"  [OK] NAS: {len(r.stdout.strip().splitlines())} entradas top-level")
        else:
            print(f"  [FAIL] NAS: {r.stderr[:200]}"); all_ok = False
    except Exception as e:
        print(f"  [FAIL] NAS lsf: {e}"); all_ok = False
    try:
        from rag import qdrant_backend
        info = qdrant_backend.stats()
        if "error" not in info:
            print(f"  [OK] Qdrant: {info.get('points', 0):,} pts")
        else:
            print(f"  [FAIL] Qdrant: {info['error']}"); all_ok = False
    except Exception as e:
        print(f"  [FAIL] Qdrant: {e}"); all_ok = False
    try:
        from rag.reranker import embed_texts
        v = embed_texts(["ok"])
        print(f"  [OK] Embedder: shape {v.shape}")
    except Exception as e:
        print(f"  [FAIL] Embedder: {e}"); all_ok = False
    try:
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        tf = TMP_DIR / f".preflight_{uuid.uuid4().hex}"
        tf.write_bytes(b"ok"); tf.unlink()
        print(f"  [OK] TMP_DIR {TMP_DIR}")
    except Exception as e:
        print(f"  [FAIL] TMP_DIR: {e}"); all_ok = False
    print("=== PREFLIGHT %s ===" % ("OK" if all_ok else "FAIL"))
    return all_ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()

    if args.preflight:
        sys.exit(0 if _preflight() else 1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    remote_path = f"{REMOTE}:{NAS_SHARE_PATH}"

    logger.info("Streaming NAS + filtrando contra Qdrant...")
    t0 = time.time()
    candidates: list[tuple[str, str, float, int]] = []
    listed = checked = 0
    hit_limit = False
    try:
        for entry in _rclone_list(remote_path):
            listed += 1
            rel = entry["Path"]
            if not _is_supported(rel):
                continue
            mtime = _mtime_epoch(entry.get("ModTime", ""))
            size = int(entry.get("Size", 0))
            virtual = _virtual_path(rel)
            checked += 1
            if _needs_process(virtual, mtime):
                candidates.append((rel, virtual, mtime, size))
                if args.limit and len(candidates) >= args.limit:
                    hit_limit = True
                    break
            if checked % 5000 == 0:
                logger.info("  progreso: listados %d, chequeados %d, faltantes %d",
                           listed, checked, len(candidates))
    except Exception as e:
        logger.error("listado fallo: %s", e)
        return

    logger.info("Filter fin en %.1fs: %d listados, %d chequeados, %d faltantes%s",
                time.time() - t0, listed, checked, len(candidates),
                " (limit)" if hit_limit else "")

    if args.dry_run:
        for rel, v, m, s in candidates[:20]:
            logger.info("  DRY: %s (%.0fMB)", rel, s / 1e6)
        logger.info("--dry-run: %d candidatos", len(candidates))
        return

    ok = err = 0
    total = len(candidates)
    for i, (rel, virtual, mtime, size) in enumerate(candidates, 1):
        ext = "." + rel.rsplit(".", 1)[-1]
        tmp = TMP_DIR / f"{uuid.uuid4().hex}{ext}"
        try:
            if not _rclone_download(f"{remote_path}/{rel}", tmp):
                err += 1; continue
            status = _process_and_upsert(tmp, virtual, mtime, size)
            if status == "ok": ok += 1
            else: err += 1
        except Exception as e:
            err += 1
            logger.warning("proceso %s fallo: %s", rel, str(e)[:150])
        finally:
            tmp.unlink(missing_ok=True)
        if i % 20 == 0:
            logger.info("  [%d/%d] ok=%d err=%d", i, total, ok, err)

    logger.info("Fin: ok=%d err=%d de %d candidatos", ok, err, total)

    try:
        ckpt = Path("/data/logs/nas_ondemand_last.json")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(json.dumps({
            "timestamp": time.time(), "listados": listed, "chequeados": checked,
            "candidatos": total, "procesados_ok": ok, "procesados_err": err,
        }, indent=2))
    except Exception as e:
        logger.warning("checkpoint fallo: %s", e)


if __name__ == "__main__":
    main()
