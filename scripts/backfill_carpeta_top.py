"""Backfill del payload field 'carpeta_top' en todos los puntos existentes.

Contexto: hasta este fix, _carpeta_top() calculaba mal y quedaba "volume1" para
todos los paths (el primer segmento absoluto). El fix nuevo descuenta VIRTUAL_ROOT
para dar la carpeta real (ARABELA, CARO, ASOCIART, etc).

Este script scrollea toda la coleccion, recalcula carpeta_top desde el path, y
hace set_payload SOLO para los que quedaron mal. Idempotente: si ya esta bien,
lo saltea.

Uso en el VPS:
    docker exec mg-gateway python /app/scripts/backfill_carpeta_top.py [--dry-run] [--batch-size 500]

Snapshot: se guarda /tmp/backfill_carpeta_top_snapshot.jsonl con [{id, path, carpeta_top_viejo}]
para poder revertir si hace falta.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("RAG_BACKEND", "qdrant")

logger = logging.getLogger("backfill_carpeta_top")

SNAPSHOT_PATH = Path("/tmp/backfill_carpeta_top_snapshot.jsonl")


def _iter_all_points(c, collection: str, batch: int = 500):
    """Scroll paginado por toda la coleccion. Yields lists de puntos."""
    offset = None
    while True:
        chunk, offset = c.scroll(
            collection_name=collection, limit=batch, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not chunk:
            return
        yield chunk
        if offset is None:
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Solo cuenta cuantos requieren fix")
    ap.add_argument("--batch-size", type=int, default=500)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from rag import qdrant_backend
    from qdrant_client.models import PointStruct

    c = qdrant_backend.get_client()
    collection = qdrant_backend.COLLECTION

    total = 0
    fix_needed = 0
    fixed = 0
    snapshot_f = None
    if not args.dry_run:
        snapshot_f = SNAPSHOT_PATH.open("w", encoding="utf-8")
        logger.info("snapshot -> %s", SNAPSHOT_PATH)

    t0 = time.time()
    to_update: list[tuple] = []  # (id, path, new_carpeta_top)

    try:
        for batch in _iter_all_points(c, collection, batch=args.batch_size):
            for p in batch:
                total += 1
                payload = p.payload or {}
                path = payload.get("path") or ""
                old = payload.get("carpeta_top") or ""
                new = qdrant_backend._carpeta_top(path)
                if new != old:
                    fix_needed += 1
                    if snapshot_f:
                        snapshot_f.write(json.dumps({
                            "id": str(p.id), "path": path, "carpeta_top_viejo": old,
                        }, ensure_ascii=False) + "\n")
                    to_update.append((p.id, new))

            # Flush batch
            if to_update and not args.dry_run and len(to_update) >= args.batch_size:
                _flush_updates(c, collection, to_update)
                fixed += len(to_update)
                to_update = []

            if total % 5000 == 0:
                dt = time.time() - t0
                rate = total / max(dt, 0.1)
                logger.info("  progreso: %d escaneados, %d requieren fix, %d aplicados (%.0f pts/s)",
                            total, fix_needed, fixed, rate)

        # Ultimo flush
        if to_update and not args.dry_run:
            _flush_updates(c, collection, to_update)
            fixed += len(to_update)
    finally:
        if snapshot_f:
            snapshot_f.close()

    dt = time.time() - t0
    logger.info("=" * 60)
    logger.info("Total escaneados: %d en %.1fs", total, dt)
    logger.info("Requieren fix: %d", fix_needed)
    if args.dry_run:
        logger.info("--dry-run: no se aplico ningun cambio")
    else:
        logger.info("Aplicados: %d", fixed)
        logger.info("Snapshot para revert: %s", SNAPSHOT_PATH)


def _flush_updates(c, collection: str, updates: list[tuple]) -> None:
    """Aplica set_payload en batch (mas barato que upsert full)."""
    from qdrant_client.models import SetPayload

    # Agrupar por carpeta_top nuevo para minimizar llamadas
    by_carpeta: dict[str, list] = {}
    for pid, new in updates:
        by_carpeta.setdefault(new, []).append(pid)

    for carpeta, ids in by_carpeta.items():
        try:
            c.set_payload(
                collection_name=collection,
                payload={"carpeta_top": carpeta},
                points=ids,
                wait=False,
            )
        except Exception as e:
            logger.warning("set_payload fallo para carpeta='%s' (%d ids): %s",
                           carpeta, len(ids), str(e)[:200])


if __name__ == "__main__":
    main()
