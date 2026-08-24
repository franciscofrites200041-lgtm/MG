"""Suite de queries reales para verificar precision del RAG.

Corre 6 consultas legales tipicas y muestra:
    - Top hits (con score vector + score cross-encoder)
    - Si un filename esperado aparece en top-K
    - Latencia por query

Uso (dentro del container gateway):
    docker exec mg-gateway python /app/gateway/test_precision.py

Uso local:
    python gateway/test_precision.py
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QDRANT_URL", os.getenv("QDRANT_URL", "http://qdrant:6333"))

from gateway.main import _rag_context, _is_trivial, _extract_keywords, _expand_query


QUERIES = [
    {
        "q": "que dice el archivo 1ALLIANZ.doc",
        "esperado": "1ALLIANZ.doc",
        "descripcion": "Busqueda por nombre exacto de archivo",
    },
    {
        "q": "cedula AZ demanda y prueba",
        "esperado": "CEDULA AZ",
        "descripcion": "Busqueda por keywords del filename",
    },
    {
        "q": "convenio Allianz Mapfre",
        "esperado": "ALLIANZ",
        "descripcion": "Busqueda temática (convenios entre aseguradoras)",
    },
    {
        "q": "informe de juicios patrimoniales mayor a tres millones",
        "esperado": "JUICIOS",
        "descripcion": "Busqueda semantica larga",
    },
    {
        "q": "hola",
        "esperado": None,
        "descripcion": "Trivial (no dispara RAG)",
    },
    {
        "q": "poliza vehicular contra terceros con clausula de alcoholemia",
        "esperado": None,
        "descripcion": "Busqueda semantica compuesta",
    },
]


def _color(txt, code):
    return f"\033[{code}m{txt}\033[0m"


def run():
    print(_color("=" * 80, "36"))
    print(_color("SUITE DE PRECISION - RAG Gateway", "36"))
    print(_color("=" * 80, "36"))

    for i, tc in enumerate(QUERIES, 1):
        q = tc["q"]
        exp = tc["esperado"]
        print(f"\n{_color(f'[{i}/{len(QUERIES)}]', '33')} {tc['descripcion']}")
        print(f"  Query: {_color(q, '37')}")

        t0 = time.time()
        if _is_trivial(q):
            print(f"  {_color('SKIP RAG (query trivial)', '90')}")
            print(f"  Latencia: {time.time() - t0:.2f}s")
            continue

        kws = _extract_keywords(q)
        print(f"  Keywords: {kws}")

        try:
            variants = _expand_query(q)
            print(f"  Variantes ({len(variants)}):")
            for v in variants:
                print(f"    - {v}")
        except Exception as e:
            print(f"  {_color(f'Expansion fallo: {e}', '31')}")

        try:
            ctx, hits = _rag_context(q)
        except Exception as e:
            print(f"  {_color(f'FALLO retrieval: {e}', '31')}")
            continue

        dt = time.time() - t0
        print(f"  Latencia: {dt:.2f}s  |  hits={len(hits)}  |  contexto={len(ctx)} chars")

        found = False
        for j, h in enumerate(hits[:8], 1):
            fn = h.get("filename", "?")
            pg = h.get("page", "?")
            sc = h.get("score", 0)
            ce = h.get("_ce_score", 0)
            match = exp and exp.lower() in fn.lower()
            if match:
                found = True
            marker = _color("<-- ESPERADO", "32") if match else ""
            print(f"    [{j}] score={sc:.2f} ce={ce:.2f}  {fn} p.{pg} {marker}")

        if exp:
            if found:
                print(f"  {_color('PASA: filename esperado en top hits', '32')}")
            else:
                print(f"  {_color(f'FALLA: no se encontró {exp!r} en top-8', '31')}")


if __name__ == "__main__":
    run()
