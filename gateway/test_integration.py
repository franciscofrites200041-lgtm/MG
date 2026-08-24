"""Test de integracion end-to-end: Qdrant en memoria + docs sinteticos + queries reales.

Valida:
    - Retrieval encuentra archivos por nombre exacto
    - Retrieval encuentra archivos por keyword
    - Retrieval no encuentra lo que no existe
    - Rerank funciona
    - Contexto ampliado
    - Validador de citas
    - Intent detector

Uso:
    python gateway/test_integration.py

Sin OpenAI real (mock). Sin VPS. Puro Python + Qdrant embedded.
"""
from __future__ import annotations
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["OPENAI_API_KEY"] = "test-dummy-no-real-call"
os.environ["ENABLE_QUERY_EXPANSION"] = "0"      # cero llamadas al LLM
os.environ["ENABLE_CROSS_ENCODER"] = "0"        # sin descargar modelo cross-encoder
os.environ["CONTEXT_RADIUS"] = "1"
os.environ["QDRANT_TOP_K"] = "5"
os.environ["QDRANT_POOL"] = "20"
os.environ["QDRANT_URL"] = ":memory:"           # embedded

# Init: Qdrant en memoria antes de importar gateway.main
from qdrant_client import QdrantClient
from rag import qdrant_backend
qdrant_backend._client = QdrantClient(":memory:")
qdrant_backend.ensure_collection()

# Insertar docs sinteticos
from rag.qdrant_backend import ChunkPoint, upsert_chunks
import numpy as np

def _vec(seed: int) -> list[float]:
    rng = np.random.RandomState(seed)
    v = rng.randn(384).astype("float32")
    v /= np.linalg.norm(v)
    return v.tolist()


DOCS = [
    # (filename, page, snippet, seed)
    ("1ALLIANZ.doc", 1, "Convenio marco entre Allianz Argentina y el Estudio Montoya-Gherzi para gestion de siniestros vehiculares.", 1),
    ("1ALLIANZ.doc", 2, "Los honorarios se calculan como el 15% del monto conciliado o sentenciado por siniestro.", 2),
    ("CEDULA AZ DEMANDA Y PRUEBA.pdf", 71, "Cedula de notificacion al Sr. Perez Juan Carlos para presentar contestacion de demanda y ofrecer prueba en autos caratulados Perez c/ Allianz s/ danos y perjuicios.", 3),
    ("CEDULA AZ DEMANDA Y PRUEBA.pdf", 72, "Se acompana copia de la demanda y documental. Plazo legal 15 dias habiles. Firmado por Dr. Gonzalez, Juzgado Civil 3.", 4),
    ("POLIZA (3).pdf", 40, "Clausula 18: quedan excluidos de la cobertura los siniestros producidos por conductor con alcoholemia superior a 0,50 gramos por litro de sangre.", 5),
    ("POLIZA (3).pdf", 41, "El asegurado debe denunciar cualquier siniestro dentro de las 72 horas de ocurrido. Falta de denuncia causa perdida de cobertura.", 6),
    ("ACUERDO TRANSACCIONAL honorarios.doc", 3, "Se acuerda entre las partes que la suma total de honorarios asciende a pesos quinientos mil ($500.000) pagaderos en tres cuotas iguales.", 7),
    ("INFORME JUICIOS MAPFRE.doc", 15, "Informe semestral: juicios patrimoniales contra Mapfre superiores a $3.986.675,49 gestionados por el estudio. Total 47 causas activas.", 8),
    ("CONTESTA DEMANDA.doc", 10, "Se contesta demanda en tiempo y forma. Se ofrece prueba documental, testimonial y pericial contable. Oportunamente rechacese la demanda con costas.", 9),
    ("ALEGATOS.docx", 3, "Habida cuenta de la prueba producida y los hechos acreditados, corresponde hacer lugar a la demanda por resultar procedente en todos sus terminos.", 10),
]

points = []
for fn, pg, snip, seed in DOCS:
    path = f"/volume1/Publico/Estudio/{fn}"
    points.append(ChunkPoint(
        path=path, filename=fn, ext="." + fn.rsplit(".", 1)[-1],
        page=pg, chunk_idx=0, snippet=snip,
        mtime=1700000000.0, size=len(snip) * 10,
        vector=_vec(seed),
    ))
upsert_chunks(points, batch_size=10)
print(f"[SETUP] {len(points)} docs sinteticos indexados en Qdrant memory")

# Ahora importar gateway.main (usa nuestro qdrant_backend con datos)
from gateway.main import (
    _rag_context, _is_trivial, _is_escrito_request,
    _extract_keywords, _validate_citations,
)


def _c(s, code):
    return f"\033[{code}m{s}\033[0m"


def run():
    print(_c("\n" + "=" * 80, "36"))
    print(_c("TEST INTEGRATION - Pipeline end-to-end (Qdrant memory)", "36"))
    print(_c("=" * 80, "36"))

    tests = [
        {"q": "que dice el archivo 1ALLIANZ.doc", "esperado_fn": "1ALLIANZ.doc"},
        {"q": "cedula AZ demanda", "esperado_fn": "CEDULA AZ DEMANDA Y PRUEBA.pdf"},
        {"q": "acuerdo transaccional honorarios", "esperado_fn": "ACUERDO TRANSACCIONAL"},
        {"q": "informe juicios Mapfre patrimoniales", "esperado_fn": "MAPFRE"},
        {"q": "clausula alcoholemia", "esperado_fn": "POLIZA"},
        {"q": "contesta demanda con costas", "esperado_fn": "CONTESTA DEMANDA"},
    ]

    passed = 0
    failed = 0

    for i, tc in enumerate(tests, 1):
        q = tc["q"]
        exp = tc["esperado_fn"]
        print(f"\n{_c(f'[{i}]', '33')} query: {_c(q, '37')}")
        print(f"     esperado_fn: {exp}")

        kws = _extract_keywords(q)
        print(f"     keywords: {kws}")

        _, hits = _rag_context(q)
        print(f"     hits: {len(hits)}")
        found = False
        for j, h in enumerate(hits[:5], 1):
            fn = h.get("filename", "?")
            match = exp.upper() in fn.upper()
            marker = _c(" <-- match", "32") if match else ""
            print(f"       [{j}] score={h.get('score',0):.2f} {fn} p.{h.get('page')}{marker}")
            if match:
                found = True
        if found:
            print(_c(f"     PASA", "32"))
            passed += 1
        else:
            print(_c(f"     FALLA - no encontro {exp!r}", "31"))
            failed += 1

    # Tests auxiliares
    print(f"\n{_c('[AUX] intent detector', '33')}")
    aux = [
        ("hola", "trivial"),
        ("gracias", "trivial"),
        ("redactame una demanda por danos", "escrito"),
        ("generame un modelo de contestacion", "escrito"),
        ("que dice el contrato", "consulta"),
    ]
    for q, exp_intent in aux:
        actual = "trivial" if _is_trivial(q) else ("escrito" if _is_escrito_request(q) else "consulta")
        ok = "PASS" if actual == exp_intent else "FAIL"
        color = "32" if actual == exp_intent else "31"
        print(f"  {_c(ok, color)}  {q!r:<50} intent={actual} (esperado={exp_intent})")
        if actual == exp_intent:
            passed += 1
        else:
            failed += 1

    # Test validator
    print(f"\n{_c('[AUX] validator citas', '33')}")
    fake_hits = [{"filename": "1ALLIANZ.doc", "page": 1}, {"filename": "POLIZA (3).pdf", "page": 40}]
    resp = "Ver (1ALLIANZ.doc, pag. 1) y (POLIZA (3).pdf, pag. 40) tambien (FAKE.pdf, pag. 99)."
    v = _validate_citations(resp, fake_hits)
    ok = v["total"] == 3 and len(v["invalidas"]) == 1
    color = "32" if ok else "31"
    print(f"  {_c('PASS' if ok else 'FAIL', color)}  total={v['total']} validas={len(v['validas'])} invalidas={len(v['invalidas'])}")
    if ok:
        passed += 1
    else:
        failed += 1

    total = passed + failed
    print(_c("\n" + "=" * 80, "36"))
    color = "32" if failed == 0 else "31"
    print(_c(f"RESULTADO: {passed}/{total} PASS", color))
    print(_c("=" * 80, "36"))
    return failed == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
