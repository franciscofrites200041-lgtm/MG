"""Drena JSONs del drop-folder e importa a la DB del bot.

Corre en el NAS (o donde este la DB). Puede correr como job periodico (cron/task)
o llamarse antes de cada query desde el bot.

JSON esperado (producido por scripts/bg_worker.py):
    {
      "path": "/volume1/Publico/ANDY-1/DEMANDA.pdf",
      "filename": "DEMANDA.pdf",
      "ext": ".pdf",
      "mtime": 1234567890.0,
      "size": 12345,
      "status": "ok",
      "error": null,
      "chunks": [{"page": 1, "text": "..."}, ...],
      "embeddings": [{"page": 1, "vec_b64": "...", "model": "...", "dim": 384}, ...]
    }

Uso:
    python scripts/drain_pending.py --drop-dir /volume1/Publico/.mg_bot/pending --db ./data/index.db
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.extractor import init_db  # noqa: E402

logger = logging.getLogger("drain")


def import_payload(conn: sqlite3.Connection, payload: dict) -> dict:
    """Inserta chunks + embeddings y marca fully_extracted=1. Idempotente."""
    path = payload["path"]
    filename = payload["filename"]
    ext = payload["ext"]
    mtime = payload["mtime"]
    size = payload["size"]
    status = payload.get("status", "ok")
    chunks = payload.get("chunks") or []
    embeddings = payload.get("embeddings") or []

    # Wipe cualquier estado previo (preview lite, o full anterior).
    conn.execute(
        "DELETE FROM chunk_embeddings WHERE chunk_rowid IN "
        "(SELECT rowid FROM chunks WHERE path = ?)",
        (path,),
    )
    conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
    conn.execute("DELETE FROM files WHERE path = ?", (path,))

    fully = 1 if chunks else 0
    conn.execute(
        "INSERT INTO files (path, filename, ext, mtime, size, n_chunks, indexed_at, "
        "status, preview, fully_extracted) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (path, filename, ext, mtime, size, len(chunks), time.time(),
         status, "", fully),
    )
    if not chunks:
        return {"chunks": 0, "embeddings": 0}

    # Insertar chunks preservando el orden (page).
    conn.executemany(
        "INSERT INTO chunks (path, filename, page, text) VALUES (?,?,?,?)",
        [(path, filename, c["page"], c["text"]) for c in chunks],
    )

    # Mapear embeddings por page -> rowid.
    if embeddings:
        by_page = {c["page"]: c for c in chunks}
        page_to_rowid = dict(conn.execute(
            "SELECT page, rowid FROM chunks WHERE path = ?", (path,)
        ).fetchall())
        emb_rows: list[tuple] = []
        for e in embeddings:
            pg = e["page"]
            if pg not in page_to_rowid or pg not in by_page:
                continue
            vec = np.frombuffer(base64.b64decode(e["vec_b64"]), dtype=np.float32)
            emb_rows.append((page_to_rowid[pg], e["model"], e["dim"], vec.tobytes()))
        if emb_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO chunk_embeddings (chunk_rowid, model, dim, vec) "
                "VALUES (?,?,?,?)",
                emb_rows,
            )
    return {"chunks": len(chunks), "embeddings": len(embeddings)}


def drain_once(drop_dir: Path, db_path: str) -> dict:
    """Procesa todos los .json presentes en drop_dir. Borra los ya importados."""
    if not drop_dir.exists():
        return {"scanned": 0, "imported": 0, "failed": 0}
    jsons = sorted(drop_dir.glob("*.json"))
    if not jsons:
        return {"scanned": 0, "imported": 0, "failed": 0}
    conn = init_db(db_path)
    imported, failed = 0, 0
    for j in jsons:
        try:
            payload = json.loads(j.read_text(encoding="utf-8"))
            import_payload(conn, payload)
            conn.commit()
            j.unlink()
            imported += 1
        except Exception as e:
            logger.exception("Fallo al importar %s: %s", j.name, e)
            # ponytail: mover a .failed en vez de dejar loop infinito.
            try:
                j.rename(j.with_suffix(".json.failed"))
            except Exception:
                pass
            failed += 1
    conn.close()
    return {"scanned": len(jsons), "imported": imported, "failed": failed}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--drop-dir", required=True)
    p.add_argument("--db", default="./data/index.db")
    p.add_argument("--loop", action="store_true", help="Correr en bucle infinito (poll cada --interval)")
    p.add_argument("--interval", type=int, default=30, help="Segundos entre passes en modo loop")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    drop_dir = Path(args.drop_dir)

    if not args.loop:
        r = drain_once(drop_dir, args.db)
        logger.info("Drain: %s", r)
        return

    logger.info("Loop drain, interval=%ds, drop_dir=%s", args.interval, drop_dir)
    while True:
        try:
            r = drain_once(drop_dir, args.db)
            if r["scanned"] > 0:
                logger.info("Drain: %s", r)
        except Exception as e:
            logger.exception("Fallo el pass: %s", e)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
