"""Wrapper CLI para la extracción total desde la PC potente.

Uso típico:
    python scripts/index_full.py --root "Z:\\estudio_juridico" --db ./index.db --workers 8
    (después copiar el .db resultante al NAS)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Permite correr sin pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.extractor import _cli  # noqa: E402

if __name__ == "__main__":
    _cli()
