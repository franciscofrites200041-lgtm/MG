"""Cuenta archivos indexables del NAS usando el MISMO walker que el extractor.
Excluye #recycle, @eaDir, lockfiles ~$. Da total, size, desglose por extensión."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.extractor import walk_files, SUPPORTED_EXT  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/count_nas.py <ruta_nas>")
        sys.exit(1)
    root = sys.argv[1]
    print(f"Contando archivos indexables en {root} (con exclusiones aplicadas)...")
    print("Progreso cada 500 archivos:")
    t0 = time.time()
    total = 0
    total_size = 0
    por_ext: dict[str, int] = {ext: 0 for ext in SUPPORTED_EXT}
    for p in walk_files(root):
        total += 1
        try:
            total_size += p.stat().st_size
        except OSError:
            pass
        por_ext[p.suffix.lower()] = por_ext.get(p.suffix.lower(), 0) + 1
        if total % 500 == 0:
            dt = time.time() - t0
            print(f"  {total} archivos ({total_size / (1024**3):.2f} GB) - {dt:.0f}s")
    dt = time.time() - t0
    print("---")
    print(f"Walk termino en {dt:.1f}s")
    print(f"Total archivos indexables: {total}")
    print(f"Tamano total: {total_size / (1024**3):.2f} GB")
    print("Desglose por extension:")
    for ext, n in sorted(por_ext.items(), key=lambda x: -x[1]):
        if n > 0:
            print(f"  {ext}: {n}")


if __name__ == "__main__":
    main()
