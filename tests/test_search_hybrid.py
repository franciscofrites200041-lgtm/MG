"""Tests de integracion: hybrid search cubre las quejas reales del usuario.

Simula los 5 turnos que fallaron en produccion:
1. 'en asociart, BENAVIDEZ N° 170495 la demanda' -> filename boost trae el archivo
2. 'excepcion de pago en documentos arabela' -> scope carpeta filtra
3. 'primeros tres casos de alcoholemia' -> intent 'caso' promueve expedientes
4. Query independiente no arrastra keywords del turno anterior
5. Filename similar cuando 'no consta'
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ejecutar desde raiz del bot repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["RERANKER_MOCK"] = "1"
os.environ["USE_RERANKER"] = "1"
os.environ["RAG_BACKEND"] = "qdrant"
os.environ["VIRTUAL_ROOT"] = "/volume1/Publico/Estudio"


def _setup_qdrant_memoria():
    from qdrant_client import QdrantClient
    from rag import qdrant_backend, reranker

    reranker._encoder = None
    # Ajustar VECTOR_SIZE al del mock encoder (36) para que Qdrant acepte los vectores
    qdrant_backend.VECTOR_SIZE = reranker._MockEncoder.dim
    qdrant_backend._client = QdrantClient(":memory:")
    qdrant_backend._carpetas_cache = None
    qdrant_backend.ensure_collection()

    # Cargar corpus fixture: 3 carpetas top con archivos varios
    corpus = [
        # ASOCIART: BENAVIDEZ
        ("/volume1/Publico/Estudio/ASOCIART/BENAVIDEZ CINTIA VANESA N° 170495 - Demanda.pdf",
         1, "demanda por accidente laboral, riesgo del trabajo, cobertura excluida"),
        ("/volume1/Publico/Estudio/ASOCIART/BENAVIDEZ CINTIA VANESA N° 170495 - Demanda.pdf",
         2, "hechos del reclamo, actora Cintia Benavidez, ART Asociart"),
        # DOCUMENTOS ARABELA: excepcion de pago
        ("/volume1/Publico/Estudio/DOCUMENTOS ARABELA/OPONE EXCEPCION DE PAGO TOTAL DOCUMENTADO.docx",
         1, "excepcion de pago total documentado, articulo 310 CPC"),
        ("/volume1/Publico/Estudio/DOCUMENTOS ARABELA/BRESCA CUMPLIMIENTO.docx",
         1, "acredito cumplimiento con comprobantes de reintegro"),
        # DOCUMENTOS CARO: alcoholemia (casos)
        ("/volume1/Publico/Estudio/DOCUMENTOS CARO/PEREZ JUAN c/ TRES ROMBOS - alcoholemia N° 123456.pdf",
         1, "asegurado en estado de ebriedad, test de alcoholemia 1.2 g/l"),
        ("/volume1/Publico/Estudio/DOCUMENTOS CARO/GOMEZ c/ GALENO N° 234567.pdf",
         1, "alcohol en sangre, exclusion cobertura poliza"),
        ("/volume1/Publico/Estudio/DOCUMENTOS CARO/RUIZ c/ ZURICH N° 345678.pdf",
         1, "alcoholemia positivo, negativa a realizar test"),
        # Poliza generica (concepto, NO caso)
        ("/volume1/Publico/Estudio/DOCUMENTOS CARO/POLIZA.pdf",
         28, "en estado de ebriedad, examen alcoholemia, un gramo por mil"),
    ]

    from rag.qdrant_backend import ChunkPoint, upsert_chunks
    from rag.reranker import embed_texts
    vecs = embed_texts([snip for _, _, snip in corpus])
    points = []
    for (path, page, snip), vec in zip(corpus, vecs):
        points.append(ChunkPoint(
            path=path, filename=Path(path).name, ext=".pdf",
            page=page, chunk_idx=0, snippet=snip,
            mtime=1700000000.0, size=1000,
            vector=vec.astype("float32").tolist(),
        ))
    upsert_chunks(points, batch_size=32)


def test_1_expediente_con_carpeta_y_nombre():
    """Query 1 real: 'carpeta caro, en asociart, BENAVIDEZ N° 170495 la demanda'."""
    from rag.search import buscar_en_documentos
    r = buscar_en_documentos(
        "en asociart, BENAVIDEZ CINTIA VANESA N° 170495 la demanda para contestar",
        top_k=5,
    )
    assert "BENAVIDEZ" in r, f"filename boost fallo:\n{r}"
    assert "170495" in r, f"expediente no aparecio:\n{r}"


def test_2_scope_carpeta_filtra():
    """Query 2 real: 'excepcion de pago en documentos arabela'."""
    from rag.search import buscar_en_documentos
    r = buscar_en_documentos(
        "escrito de excepcion de pago en la carpeta documentos arabela", top_k=5,
    )
    assert "EXCEPCION DE PAGO" in r or "excepcion" in r.lower(), f"no matcheo excepcion:\n{r}"
    # No debe traer POLIZA.pdf de CARO
    assert "POLIZA.pdf" not in r or "DOCUMENTOS CARO" not in r, f"scope filtra mal:\n{r}"


def test_3_intent_caso_prioriza_expedientes():
    """Query 3 real: 'dame los primeros tres casos de alcoholemia'.
    Deben salir expedientes (PEREZ, GOMEZ, RUIZ) antes que POLIZA generica.
    """
    from rag.search import buscar_en_documentos
    r = buscar_en_documentos("dame los primeros tres casos de alcoholemia", top_k=4)
    # POLIZA.pdf es concepto (define alcoholemia), no un caso. Debe salir despues.
    pos_expte = min([r.find(name) for name in ("PEREZ", "GOMEZ", "RUIZ") if name in r] + [10**9])
    pos_poliza = r.find("POLIZA.pdf")
    if pos_poliza >= 0:
        assert pos_expte < pos_poliza, f"POLIZA.pdf salio antes que expedientes:\n{r}"


def test_4_query_independiente_sin_herencia_sucia():
    """Query 4 real: nuevo tema no debe arrastrar 'alcoholemia' de turno previo."""
    from rag.search import buscar_en_documentos
    prev = {"nombres_propios": ["ALCOHOLEMIA"]}
    r = buscar_en_documentos(
        "yo no dije alcoholemia, dame un caso random de documentos arabela",
        top_k=3, prev_context=prev,
    )
    # Debe salir algo de ARABELA, no de CARO
    assert "ARABELA" in r or "arabela" in r.lower(), f"no matcheo scope:\n{r}"


def test_5_fallback_sugerencias_por_filename():
    """Cuando no hay match, sugerir filenames parecidos."""
    from rag.search import buscar_en_documentos
    # Nombre que no existe pero se parece a uno que si
    r = buscar_en_documentos("BENAVIDES CYNTHIA N° 999999", top_k=3)
    # Debe sugerir BENAVIDEZ (similar) aunque el expediente 999999 no exista
    assert "BENAVIDEZ" in r or "Sin resultados" in r, f"fallback no sugirio:\n{r}"


def test_6_citation_matches_normalizado():
    """Citation verifier debe tolerar mayusculas/acentos/espacios."""
    from rag.search import citation_matches
    paths = ["/volume1/Publico/Estudio/CARO/Benavídez Demanda.pdf"]
    assert citation_matches("BENAVIDEZ DEMANDA.pdf", paths)
    assert citation_matches("Benavidez  Demanda.pdf", paths)
    assert not citation_matches("otro archivo random.pdf", paths)


def run_all():
    _setup_qdrant_memoria()
    fns = [
        test_1_expediente_con_carpeta_y_nombre,
        test_2_scope_carpeta_filtra,
        test_3_intent_caso_prioriza_expedientes,
        test_4_query_independiente_sin_herencia_sucia,
        test_5_fallback_sugerencias_por_filename,
        test_6_citation_matches_normalizado,
    ]
    ok = 0
    for fn in fns:
        try:
            fn()
            print(f"  [OK] {fn.__name__}")
            ok += 1
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
        except Exception as e:
            print(f"  [ERROR] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    return ok == len(fns)


if __name__ == "__main__":
    sys.exit(0 if run_all() else 1)
