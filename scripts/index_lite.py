"""CLI para indexar en modo lazy: solo preview (~2000 chars) por archivo.

Es el primer paso del pipeline lazy. Corre en horas en vez de dias. Los archivos
quedan con fully_extracted=0. La extraccion completa la hacen despues:
- El bot en produccion, on-demand al consultarse (rag.search).
- El worker de background en la PC (scripts/bg_worker.py).

Uso:
    python scripts/index_lite.py --root /volume1/Publico --db ./data/index.db --workers 4
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.extractor import index_root_lite  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--db", default="./data/index.db")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--no-incremental", action="store_true",
                   help="Reprocesa todo (por default salta lo ya indexado por mtime+size)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    stats = index_root_lite(
        args.root, args.db, args.workers,
        incremental=not args.no_incremental,
        limit=args.limit,
    )
    dt = time.time() - t0
    logging.info("Lite terminado en %.1fs: %s", dt, stats)


if __name__ == "__main__":
    main()
