"""Smoke tests con asserts. Sin frameworks. Corren en <2s.

Uso:
    python tests/test_smoke.py

El re-ranker corre en modo MOCK (RERANKER_MOCK=1) para no bajar el modelo real.
Para validar el modelo real: `python -m rag.reranker` con las deps instaladas.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Activa el mock ANTES de importar cualquier cosa de rag
os.environ["RERANKER_MOCK"] = "1"
os.environ["USE_RERANKER"] = "1"

from rag.extractor import index_root  # noqa: E402
from rag import search as search_mod  # noqa: E402


def _reset_reranker():
    from rag import reranker
    reranker._encoder = None


def test_extractor_indexa_txt_y_reencuentra():
    _reset_reranker()
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "poliza_zurich.txt"
        f.write_text(
            "POLIZA. Exclusion por alcoholemia superior a 0.5. Asegurado Juan Perez.",
            encoding="utf-8",
        )
        db = str(Path(d) / "idx.db")
        stats = index_root(d, db, workers=1)
        assert stats["ok"] == 1, stats
        assert stats["embedded"] >= 1, stats
        search_mod.INDEX_DB = db
        r = search_mod.buscar_en_documentos("alcoholemia Perez")
        assert "poliza_zurich" in r, r


def test_extractor_incremental_no_reprocesa():
    _reset_reranker()
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "a.txt"
        f.write_text("contenido de prueba", encoding="utf-8")
        db = str(Path(d) / "idx.db")
        s1 = index_root(d, db, workers=1)
        assert s1["ok"] == 1
        s2 = index_root(d, db, workers=1, incremental=True)
        assert s2["ok"] == 0 and s2["saltados"] == 1, s2


def test_sanitize_query_saca_puntuacion():
    from rag.search import _sanitize_query
    assert _sanitize_query("Gomez, c/ Perez p/ Danos") == "Gomez Perez Danos"
    assert _sanitize_query("!!!!") == ""


def test_elegir_modelo_rutea_a_heavy_cuando_pide_redactar():
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    from agent import elegir_modelo, MODEL_FAST, MODEL_HEAVY
    assert elegir_modelo("Buscame jurisprudencia sobre alcoholemia") == MODEL_FAST
    assert elegir_modelo("Redacta la contestacion de demanda") == MODEL_HEAVY
    assert elegir_modelo("Analisis de sentencia del fallo X") == MODEL_HEAVY


def test_agent_tool_schemas_matchean_tool_map():
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    from agent import TOOL_MAP, TOOL_SCHEMAS
    nombres_schema = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert nombres_schema == set(TOOL_MAP.keys()), (nombres_schema, set(TOOL_MAP.keys()))
    # Cada schema tiene los required declarados
    for s in TOOL_SCHEMAS:
        params = s["function"]["parameters"]
        assert params["type"] == "object"
        for req in params.get("required", []):
            assert req in params["properties"], (s["function"]["name"], req)


def test_gdocs_fallback_local_genera_docx():
    os.environ.pop("GDOCS_TEMPLATE_ID", None)
    os.environ.pop("GOOGLE_SA_JSON", None)
    with tempfile.TemporaryDirectory() as d:
        os.environ["DOCS_OUT_DIR"] = d
        from tools.gdocs_tools import generar_escrito
        r = generar_escrito("CONTESTACION TEST", "SENOR JUEZ:\n\nVengo a contestar.")
        assert r.startswith("DOCUMENTO_GENERADO:"), r
        p = Path(r.split(":", 1)[1].strip())
        assert p.exists() and p.stat().st_size > 100, p


def test_text_cleaner_saca_markdown_y_divide():
    from utils.text_cleaner import dividir_mensaje, limpiar_texto
    t = "**Hola** *mundo*\n# Titulo\n[ref](http://x)"
    lim = limpiar_texto(t)
    assert "**" not in lim and "*" not in lim and "http" not in lim, lim
    assert "Titulo" in lim and "ref" in lim, lim
    partes = dividir_mensaje("a" * 8000)
    assert all(len(p) <= 3800 for p in partes), [len(p) for p in partes]


def test_reranker_embeddings_se_pueblan_al_indexar():
    """Verifica que el pipeline extractor -> embeddings queda con la tabla llena."""
    _reset_reranker()
    import sqlite3
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a.txt").write_text("primer texto sobre alcoholemia y aviso", encoding="utf-8")
        (Path(d) / "b.txt").write_text("otro texto sobre milanesas y cocina", encoding="utf-8")
        db = str(Path(d) / "idx.db")
        stats = index_root(d, db, workers=1)
        assert stats["ok"] == 2
        conn = sqlite3.connect(db)
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_emb = conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
        # Todos los chunks deben tener embedding
        assert n_emb == n_chunks and n_emb >= 2, (n_emb, n_chunks)
        # Vec no puede estar vacio
        vec = conn.execute("SELECT vec FROM chunk_embeddings LIMIT 1").fetchone()[0]
        assert len(vec) > 0
        conn.close()


def test_reranker_reordena_semanticamente_con_mock():
    """El re-ranker debe traer primero el chunk mas parecido al query segun bag-of-chars.

    Con el MockEncoder, el chunk con mayor solapamiento de letras contra la query
    gana. Preparamos 3 docs donde el mas 'parecido lexicamente' al query no es
    necesariamente el mejor rankeado por BM25.
    """
    _reset_reranker()
    with tempfile.TemporaryDirectory() as d:
        # Todos matchean 'aviso' en FTS5.
        (Path(d) / "menos_similar.txt").write_text(
            "aviso xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", encoding="utf-8"
        )
        (Path(d) / "mas_similar.txt").write_text(
            "aviso al asegurado sobre alcoholemia y exclusion de cobertura clausula",
            encoding="utf-8",
        )
        (Path(d) / "medio_similar.txt").write_text(
            "aviso zzzzzzzzzz", encoding="utf-8"
        )
        db = str(Path(d) / "idx.db")
        stats = index_root(d, db, workers=1)
        assert stats["ok"] == 3
        search_mod.INDEX_DB = db

        # Query rica en letras que aparecen mucho en "mas_similar.txt"
        r = search_mod.buscar_en_documentos(
            "aviso al asegurado sobre alcoholemia exclusion clausula", top_k=3
        )
        primera_linea = r.split("\n")[0]
        # El re-ranker (bag-of-chars) debe poner mas_similar primero.
        assert "mas_similar" in primera_linea, f"esperaba mas_similar primero, got: {r}"


def test_search_sin_reranker_sigue_funcionando():
    _reset_reranker()
    os.environ["USE_RERANKER"] = "0"
    try:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "poliza.txt").write_text("alcoholemia exclusion", encoding="utf-8")
            db = str(Path(d) / "idx.db")
            stats = index_root(d, db, workers=1)
            # Sin reranker, no se pueblan embeddings.
            assert stats["ok"] == 1
            assert stats["embedded"] == 0, stats
            search_mod.INDEX_DB = db
            r = search_mod.buscar_en_documentos("alcoholemia")
            assert "poliza" in r
    finally:
        os.environ["USE_RERANKER"] = "1"


def main():
    tests = [
        test_extractor_indexa_txt_y_reencuentra,
        test_extractor_incremental_no_reprocesa,
        test_sanitize_query_saca_puntuacion,
        test_elegir_modelo_rutea_a_heavy_cuando_pide_redactar,
        test_agent_tool_schemas_matchean_tool_map,
        test_gdocs_fallback_local_genera_docx,
        test_text_cleaner_saca_markdown_y_divide,
        test_reranker_embeddings_se_pueblan_al_indexar,
        test_reranker_reordena_semanticamente_con_mock,
        test_search_sin_reranker_sigue_funcionando,
    ]
    fallos = 0
    for t in tests:
        try:
            t()
            print("OK", t.__name__)
        except AssertionError as e:
            fallos += 1
            print("FAIL", t.__name__, "->", e)
        except Exception as e:
            fallos += 1
            print("ERROR", t.__name__, "->", type(e).__name__, e)
    print(f"\n{len(tests) - fallos}/{len(tests)} tests OK")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
