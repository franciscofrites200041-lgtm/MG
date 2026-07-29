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


def test_agent_detecta_reasoning_models():
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    from agent import _is_reasoning_model
    assert _is_reasoning_model("gpt-5")
    assert _is_reasoning_model("gpt-5-mini")
    assert _is_reasoning_model("o1-preview")
    assert _is_reasoning_model("o3-mini")
    assert not _is_reasoning_model("gpt-4o")
    assert not _is_reasoning_model("gpt-4.1")


def test_agent_construye_content_multimodal_con_imagenes():
    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    from agent import _build_user_content

    # Sin imagenes: content es string simple
    r = _build_user_content("hola", None)
    assert r == "hola"
    r = _build_user_content("hola", [])
    assert r == "hola"

    # Con 1 imagen: lista con text + image_url data URL
    r = _build_user_content("que ves?", [("image/jpeg", b"\xff\xd8fake")])
    assert isinstance(r, list) and len(r) == 2
    assert r[0] == {"type": "text", "text": "que ves?"}
    assert r[1]["type"] == "image_url"
    assert r[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    # Con 2 imagenes: 3 partes (text + 2 img)
    r = _build_user_content("compara", [("image/png", b"a"), ("image/jpeg", b"b")])
    assert len(r) == 3
    assert r[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert r[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


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


def test_extractor_docx_captura_tablas_headers_y_footers():
    """DOCX con tabla + header + footer: todo el texto termina en FTS5."""
    _reset_reranker()
    try:
        import docx
    except ImportError:
        print("SKIP docx (python-docx no instalado)")
        return
    with tempfile.TemporaryDirectory() as d:
        doc = docx.Document()
        doc.sections[0].header.paragraphs[0].text = "HEADERUNICO Estudio MG"
        doc.sections[0].footer.paragraphs[0].text = "FOOTERUNICO pagina 1"
        doc.add_paragraph("Parrafo con contenido normal del escrito.")
        t = doc.add_table(rows=2, cols=2)
        t.cell(0, 0).text = "POLIZAABC"
        t.cell(0, 1).text = "MONTOCIENMIL"
        t.cell(1, 0).text = "POLIZAXYZ"
        t.cell(1, 1).text = "MONTODOSMIL"
        f = Path(d) / "escrito_con_tabla.docx"
        doc.save(str(f))

        db = str(Path(d) / "idx.db")
        stats = index_root(d, db, workers=1)
        assert stats["ok"] == 1, stats
        search_mod.INDEX_DB = db

        # Contenido de la tabla debe ser encontrable
        r_tabla = search_mod.buscar_en_documentos("POLIZAABC MONTOCIENMIL")
        assert "escrito_con_tabla" in r_tabla, f"tabla no indexada: {r_tabla}"
        # Header
        r_h = search_mod.buscar_en_documentos("HEADERUNICO")
        assert "escrito_con_tabla" in r_h, f"header no indexado: {r_h}"
        # Footer
        r_f = search_mod.buscar_en_documentos("FOOTERUNICO")
        assert "escrito_con_tabla" in r_f, f"footer no indexado: {r_f}"


def test_extractor_txt_maneja_encoding_latin1():
    """TXT guardado en latin-1 con acentos: se decodea sin romper."""
    _reset_reranker()
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "viejo.txt"
        f.write_bytes("Poliza con acentuacion tipica: aeiouñ".encode("latin-1"))
        db = str(Path(d) / "idx.db")
        stats = index_root(d, db, workers=1)
        assert stats["ok"] == 1, stats
        search_mod.INDEX_DB = db
        r = search_mod.buscar_en_documentos("acentuacion poliza")
        assert "viejo" in r, r


def test_extractor_pdf_sin_capa_de_texto_se_marca_sin_texto():
    """PDF sin capa de texto (equivalente a escaneado): status='sin_texto', no rompe."""
    _reset_reranker()
    try:
        import pymupdf
    except ImportError:
        print("SKIP pymupdf")
        return
    import sqlite3
    with tempfile.TemporaryDirectory() as d:
        pdf_path = Path(d) / "escaneado.pdf"
        doc = pymupdf.open()
        doc.new_page()  # pagina en blanco, sin texto
        doc.save(str(pdf_path))
        doc.close()

        db = str(Path(d) / "idx.db")
        stats = index_root(d, db, workers=1)
        assert stats["ok"] == 0 and stats["sin_texto"] == 1, stats
        assert stats["err"] == 0, stats

        conn = sqlite3.connect(db)
        row = conn.execute("SELECT status, n_chunks FROM files").fetchone()
        assert row[0] == "sin_texto" and row[1] == 0, row
        conn.close()


def test_handlers_extrae_texto_de_docx_adjunto():
    """El helper que usa on_document debe capturar tablas, headers y footers igual que el pipeline offline."""
    try:
        import docx
    except ImportError:
        print("SKIP docx")
        return
    from rag.extractor import extraer_texto_de_archivo
    with tempfile.TemporaryDirectory() as d:
        doc = docx.Document()
        doc.sections[0].header.paragraphs[0].text = "HEADER XYZ"
        doc.add_paragraph("Cuerpo principal del contrato de seguro.")
        t = doc.add_table(rows=1, cols=2)
        t.cell(0, 0).text = "POLIZA123"
        t.cell(0, 1).text = "PRIMA5000"
        f = Path(d) / "contrato.docx"
        doc.save(str(f))

        texto, n = extraer_texto_de_archivo(f, ".docx")
        assert n >= 1, n
        assert "HEADER XYZ" in texto, texto
        assert "Cuerpo principal" in texto, texto
        assert "POLIZA123" in texto and "PRIMA5000" in texto, texto


def test_handlers_extrae_texto_de_txt_adjunto():
    from rag.extractor import extraer_texto_de_archivo
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "nota.txt"
        f.write_text("El asegurado incumplio el aviso.", encoding="utf-8")
        texto, n = extraer_texto_de_archivo(f, ".txt")
        assert n == 1
        assert "aviso" in texto


def test_handlers_extension_no_soportada_devuelve_vacio():
    from rag.extractor import extraer_texto_de_archivo
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "algo.xyz"
        f.write_text("contenido", encoding="utf-8")
        texto, n = extraer_texto_de_archivo(f, ".xyz")
        assert texto == "" and n == 0


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
        test_agent_detecta_reasoning_models,
        test_agent_construye_content_multimodal_con_imagenes,
        test_agent_tool_schemas_matchean_tool_map,
        test_gdocs_fallback_local_genera_docx,
        test_text_cleaner_saca_markdown_y_divide,
        test_reranker_embeddings_se_pueblan_al_indexar,
        test_reranker_reordena_semanticamente_con_mock,
        test_extractor_docx_captura_tablas_headers_y_footers,
        test_extractor_txt_maneja_encoding_latin1,
        test_extractor_pdf_sin_capa_de_texto_se_marca_sin_texto,
        test_handlers_extrae_texto_de_docx_adjunto,
        test_handlers_extrae_texto_de_txt_adjunto,
        test_handlers_extension_no_soportada_devuelve_vacio,
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
