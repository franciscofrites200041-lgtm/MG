"""Reindex incremental: pensado para el cron nocturno del NAS.

Solo procesa archivos con mtime nuevo o size distinto al del index.

Uso:
    python scripts/reindex_incremental.py
    (lee NAS_ROOT y INDEX_DB_PATH del entorno)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.extractor import index_root  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.getenv("REINDEX_LOG", "/tmp/reindex.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("reindex")


def main() -> int:
    root = os.getenv("NAS_ROOT")
    db = os.getenv("INDEX_DB_PATH", "./index.db")
    workers = int(os.getenv("REINDEX_WORKERS", "2"))  # ponytail: NAS chico, pocos workers

    if not root:
        log.error("NAS_ROOT no seteado.")
        return 1
    if not Path(root).exists():
        log.error("NAS_ROOT no existe: %s", root)
        return 2

    t0 = time.time()
    log.info("Reindex incremental arrancando. root=%s db=%s workers=%d", root, db, workers)
    stats = index_root(root, db, workers=workers, incremental=True)
    log.info("Reindex incremental listo en %.1fs: %s", time.time() - t0, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
