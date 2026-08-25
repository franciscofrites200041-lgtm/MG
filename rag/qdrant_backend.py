"""Backend Qdrant para el RAG del estudio.

Wrapper thin sobre qdrant-client. Expone:
- get_client(): singleton QdrantClient
- ensure_collection(): crea la coleccion si no existe (idempotente)
- upsert_chunks(): batch upsert de chunks con embeddings + payload
- search(): busqueda vectorial con filtros opcionales
- delete_by_path(): borra todos los chunks de un archivo (para re-indexar)
- exists_by_path_mtime(): dedup por path + mtime (para bg_worker)

El id de cada punto es hash determinstico de (path, page, chunk_idx) -> upsert idempotente.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger("rag.qdrant")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
COLLECTION = os.getenv("QDRANT_COLLECTION", "mg_docs")
VECTOR_SIZE = 384  # MiniLM-L12-v2
VIRTUAL_ROOT = os.getenv("VIRTUAL_ROOT", "").rstrip("/")

_client = None
_carpetas_cache: tuple[float, list[str]] | None = None  # (expiry_epoch, carpetas)
_CARPETAS_TTL = 300.0


def get_client():
    """Singleton QdrantClient."""
    global _client
    if _client is not None:
        return _client
    from qdrant_client import QdrantClient
    _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    return _client


def ensure_collection() -> None:
    """Crea la coleccion si no existe + asegura indices (incluido TEXT para filename).
    Idempotente: se puede correr multiples veces sin efecto negativo."""
    from qdrant_client.models import (
        Distance, VectorParams, PayloadSchemaType,
        TextIndexParams, TokenizerType,
    )

    c = get_client()
    existing = {col.name for col in c.get_collections().collections}
    if COLLECTION not in existing:
        logger.info("Creando coleccion %s (dim=%d, cosine)", COLLECTION, VECTOR_SIZE)
        c.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    # Indices para filtros exactos + rangos.
    for field, schema in [
        ("carpeta_top", PayloadSchemaType.KEYWORD),
        ("mtime", PayloadSchemaType.FLOAT),
        ("page", PayloadSchemaType.INTEGER),  # para Range (contexto paginas +/- N)
    ]:
        try:
            c.create_payload_index(collection_name=COLLECTION, field_name=field, field_schema=schema)
        except Exception as e:
            logger.warning("Index %s ya existia: %s", field, str(e)[:80])

    # path como KEYWORD (para match exact en scroll/leer_pagina) Y TEXT (para busqueda substring).
    # Qdrant permite ambos indices sobre el mismo campo.
    try:
        c.create_payload_index(collection_name=COLLECTION, field_name="path",
                               field_schema=PayloadSchemaType.KEYWORD)
    except Exception as e:
        logger.warning("Index path(keyword) ya existia: %s", str(e)[:80])
    try:
        c.create_payload_index(
            collection_name=COLLECTION, field_name="path",
            field_schema=TextIndexParams(
                type="text", tokenizer=TokenizerType.WORD,
                min_token_len=2, max_token_len=30, lowercase=True,
            ),
        )
    except Exception as e:
        logger.warning("Index path(text) ya existia: %s", str(e)[:80])

    # filename como TEXT (full-text con tokenizacion) para hybrid search.
    try:
        c.create_payload_index(
            collection_name=COLLECTION,
            field_name="filename",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2, max_token_len=30, lowercase=True,
            ),
        )
    except Exception as e:
        logger.warning("Index filename(text) ya existia: %s", str(e)[:80])


_ID_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")  # namespace fijo para determinismo


def _point_id(path: str, page: int, chunk_idx: int) -> str:
    """ID determinstico UUID5. Mismo (path, page, chunk_idx) -> mismo id -> upsert reemplaza."""
    return str(uuid.uuid5(_ID_NS, f"{path}|{page}|{chunk_idx}"))


def _carpeta_top(path: str) -> str:
    """Primer segmento del path por debajo de VIRTUAL_ROOT.

    Ej con VIRTUAL_ROOT='/volume1/Publico/Estudio':
      '/volume1/Publico/Estudio/ARABELA/juicio1.pdf' -> 'ARABELA'
    Sin VIRTUAL_ROOT (dev local): primer segmento absoluto.
    """
    p = path.replace("\\", "/")
    if VIRTUAL_ROOT and p.startswith(VIRTUAL_ROOT + "/"):
        p = p[len(VIRTUAL_ROOT) + 1:]
    p = p.lstrip("/")
    return p.split("/", 1)[0] if "/" in p else p


@dataclass
class ChunkPoint:
    path: str
    filename: str
    ext: str
    page: int
    chunk_idx: int
    snippet: str
    mtime: float
    size: int
    vector: list[float]


def upsert_chunks(points: Iterable[ChunkPoint], batch_size: int = 256) -> int:
    """Upsert batch. Devuelve cantidad total upserted."""
    from qdrant_client.models import PointStruct

    c = get_client()
    total = 0
    batch: list[PointStruct] = []
    for cp in points:
        batch.append(PointStruct(
            id=_point_id(cp.path, cp.page, cp.chunk_idx),
            vector=cp.vector,
            payload={
                "path": cp.path,
                "carpeta_top": _carpeta_top(cp.path),
                "filename": cp.filename,
                "ext": cp.ext,
                "page": cp.page,
                "chunk_idx": cp.chunk_idx,
                "snippet": cp.snippet,
                "mtime": cp.mtime,
                "size": cp.size,
            },
        ))
        if len(batch) >= batch_size:
            c.upsert(collection_name=COLLECTION, points=batch, wait=False)
            total += len(batch)
            batch = []
    if batch:
        c.upsert(collection_name=COLLECTION, points=batch, wait=True)
        total += len(batch)
    return total


def search(query_vec: list[float], limit: int = 20, carpeta: str | None = None) -> list[dict]:
    """Retorna top-k por cosine. Filtro opcional por carpeta_top."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    c = get_client()
    q_filter = None
    if carpeta:
        q_filter = Filter(must=[FieldCondition(key="carpeta_top", match=MatchValue(value=carpeta))])
    result = c.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        limit=limit,
        query_filter=q_filter,
        with_payload=True,
    )
    return [{"score": h.score, **(h.payload or {})} for h in result.points]


def _build_scope_filter(carpeta: str | None = None, path_terms: list[str] | None = None):
    """Filtro Qdrant que restringe el pool: match por carpeta_top O por substring en path.
    Ambos apuntan al mismo scope 'carpeta X' pero cubren datos viejos (sin carpeta_top
    bien seteado) y nuevos.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText

    if not carpeta and not path_terms:
        return None
    should = []
    if carpeta:
        should.append(FieldCondition(key="carpeta_top", match=MatchValue(value=carpeta)))
        # Substring en path (case insensitive gracias al TEXT index lowercase)
        should.append(FieldCondition(key="path", match=MatchText(text=carpeta)))
    for t in (path_terms or []):
        if t and len(t) >= 2:
            should.append(FieldCondition(key="path", match=MatchText(text=t)))
    return Filter(should=should) if should else None


def search_hybrid(
    query_vec: list[float],
    limit: int = 20,
    carpeta: str | None = None,
    filename_terms: list[str] | None = None,
    expedientes: list[str] | None = None,
) -> list[dict]:
    """Hybrid retrieval en 3 canales, merged y dedupped por (path, page, chunk_idx).

    Canales:
    1. Semantic: cosine sobre embedding, restringido al scope (carpeta/path terms).
    2. Filename MatchText: hits cuyo filename contiene alguno de los filename_terms
       (nombres propios, expedientes). Boost fuerte porque el usuario nombro el archivo.
    3. Path MatchText por expediente: hits cuyo path contiene el numero de expediente.

    El scoring final es max de canales (los hits por filename ganan casi siempre)
    para preservar la interpretabilidad ('salio porque el filename matcheo').
    """
    from qdrant_client.models import Filter, FieldCondition, MatchText

    c = get_client()
    scope = _build_scope_filter(carpeta=carpeta)
    per_channel = max(limit, 20)

    merged: dict[str, dict] = {}  # id -> hit dict

    def _key(h) -> str:
        return f"{h.get('path')}|{h.get('page')}|{h.get('chunk_idx')}"

    def _absorb(hits: list[dict], channel: str, weight: float) -> None:
        for h in hits:
            h = dict(h)
            h["_channel"] = channel
            h["_weight"] = weight
            k = _key(h)
            if k not in merged or (h.get("score", 0) * weight) > (merged[k].get("_final", 0)):
                h["_final"] = (h.get("score", 0) or 0) * weight
                merged[k] = h

    # 1. Semantic con scope
    try:
        sem = c.query_points(
            collection_name=COLLECTION, query=query_vec, limit=per_channel,
            query_filter=scope, with_payload=True,
        )
        _absorb(
            [{"score": h.score, **(h.payload or {})} for h in sem.points],
            channel="semantic", weight=1.0,
        )
    except Exception as e:
        logger.warning("hybrid semantic fallo: %s", e)

    # 2. Filename MatchText (uno por term). Score sintetico 1.0 con boost.
    for term in (filename_terms or []):
        if not term or len(term) < 2:
            continue
        try:
            filt_must = [FieldCondition(key="filename", match=MatchText(text=term))]
            # Si hay scope, sumarlo como must adicional
            if carpeta:
                sc = _build_scope_filter(carpeta=carpeta)
                if sc:
                    # merge: must del scope se cumple si algun should matchea; representamos con OR via should
                    fn_filter = Filter(must=filt_must, should=sc.should)
                else:
                    fn_filter = Filter(must=filt_must)
            else:
                fn_filter = Filter(must=filt_must)
            scrolled, _ = c.scroll(
                collection_name=COLLECTION, scroll_filter=fn_filter,
                limit=per_channel, with_payload=True, with_vectors=False,
            )
            _absorb(
                [{"score": 1.0, **(p.payload or {})} for p in scrolled],
                channel=f"filename:{term}", weight=1.5,
            )
        except Exception as e:
            logger.warning("hybrid filename MatchText '%s' fallo: %s", term, e)

    # 3. Path MatchText por expediente (los numeros de expte suelen estar en el path)
    for exp in (expedientes or []):
        if not exp or len(exp) < 3:
            continue
        try:
            filt = Filter(must=[FieldCondition(key="path", match=MatchText(text=exp))])
            scrolled, _ = c.scroll(
                collection_name=COLLECTION, scroll_filter=filt,
                limit=per_channel, with_payload=True, with_vectors=False,
            )
            _absorb(
                [{"score": 1.0, **(p.payload or {})} for p in scrolled],
                channel=f"expte:{exp}", weight=1.8,
            )
        except Exception as e:
            logger.warning("hybrid path MatchText '%s' fallo: %s", exp, e)

    ordenados = sorted(merged.values(), key=lambda h: h.get("_final", 0), reverse=True)
    return ordenados[:limit]


def list_carpetas_top(limit: int = 100) -> list[str]:
    """Devuelve las carpetas top-level indexadas. Cache 5 min para no pegarle en cada query."""
    global _carpetas_cache
    now = time.time()
    if _carpetas_cache and _carpetas_cache[0] > now:
        return _carpetas_cache[1]

    c = get_client()
    seen: set[str] = set()
    try:
        # Scroll paginado; con 500k puntos no queremos leer todo. Muestreamos.
        # Cada scroll devuelve payloads; extraemos carpeta_top unicas hasta juntar N distintas
        # o hasta un tope de scans.
        offset = None
        scans = 0
        while len(seen) < limit and scans < 20:
            batch, offset = c.scroll(
                collection_name=COLLECTION, limit=500, offset=offset,
                with_payload=["carpeta_top"], with_vectors=False,
            )
            if not batch:
                break
            for p in batch:
                ct = (p.payload or {}).get("carpeta_top")
                if ct:
                    seen.add(ct)
            if offset is None:
                break
            scans += 1
    except Exception as e:
        logger.warning("list_carpetas_top scroll fallo: %s", e)

    carpetas = sorted(seen)
    _carpetas_cache = (now + _CARPETAS_TTL, carpetas)
    return carpetas


def search_filenames_similar(query_terms: list[str], limit: int = 5) -> list[dict]:
    """Fallback cuando semantic no encuentra nada: devuelve archivos cuyo filename
    contiene alguno de los terms. Uso: 'no consta X, quiza quisiste decir Y.pdf'.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchText

    c = get_client()
    seen_paths: dict[str, dict] = {}
    for term in query_terms:
        if not term or len(term) < 3:
            continue
        try:
            scrolled, _ = c.scroll(
                collection_name=COLLECTION,
                scroll_filter=Filter(must=[
                    FieldCondition(key="filename", match=MatchText(text=term)),
                ]),
                limit=limit * 3, with_payload=True, with_vectors=False,
            )
            for p in scrolled:
                pl = p.payload or {}
                path = pl.get("path")
                if path and path not in seen_paths:
                    seen_paths[path] = pl
                    if len(seen_paths) >= limit:
                        break
        except Exception as e:
            logger.debug("search_filenames_similar '%s' fallo: %s", term, e)
        if len(seen_paths) >= limit:
            break
    return list(seen_paths.values())[:limit]


def exists_by_path_mtime(path: str, mtime: float) -> bool:
    """Dedup: True si ya hay al menos 1 punto con este path y mtime igual (tolerancia 1ms)."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

    c = get_client()
    q_filter = Filter(must=[
        FieldCondition(key="path", match=MatchValue(value=path)),
        FieldCondition(key="mtime", range=Range(gte=mtime - 0.001, lte=mtime + 0.001)),
    ])
    # exact=True: sin esto, Qdrant devuelve una estimacion que puede reportar >0
    # cuando en realidad son 0 hits -> marca archivos como ya indexados y los saltea.
    result = c.count(collection_name=COLLECTION, count_filter=q_filter, exact=True)
    return result.count > 0


def delete_by_path(path: str) -> None:
    """Borra todos los chunks de un archivo (para re-indexar cuando mtime cambio)."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

    c = get_client()
    c.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(filter=Filter(
            must=[FieldCondition(key="path", match=MatchValue(value=path))],
        )),
        wait=True,
    )


def stats() -> dict:
    """Info basica de la coleccion para /stats y health checks."""
    c = get_client()
    try:
        info = c.get_collection(COLLECTION)
        return {
            "collection": COLLECTION,
            "points": info.points_count,
            "vectors_size": info.config.params.vectors.size,
            "distance": str(info.config.params.vectors.distance),
            "status": info.status,
        }
    except Exception as e:
        return {"collection": COLLECTION, "error": str(e)}


def demo() -> None:
    """Self-check contra un Qdrant efimero en memoria. No requiere servicio corriendo."""
    from qdrant_client import QdrantClient
    global _client
    _client = QdrantClient(":memory:")
    ensure_collection()

    def vec(seed: int) -> list[float]:
        import random
        r = random.Random(seed)
        v = [r.gauss(0, 1) for _ in range(VECTOR_SIZE)]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v]

    v_alcohol = vec(1)
    v_recibo = vec(2)

    points = [
        ChunkPoint(
            path="Casos/2024/Perez/demanda.pdf",
            filename="demanda.pdf", ext="pdf",
            page=1, chunk_idx=0,
            snippet="alcoholemia clausula exclusion",
            mtime=1700000000.0, size=1234,
            vector=v_alcohol,
        ),
        ChunkPoint(
            path="Casos/2024/Perez/demanda.pdf",
            filename="demanda.pdf", ext="pdf",
            page=1, chunk_idx=0,  # mismo id -> reemplaza
            snippet="alcoholemia clausula exclusion v2",
            mtime=1700000000.0, size=1234,
            vector=v_alcohol,
        ),
        ChunkPoint(
            path="Casos/2024/Perez/anexo.docx",
            filename="anexo.docx", ext="docx",
            page=2, chunk_idx=1,
            snippet="recibo de pago",
            mtime=1700000001.0, size=5678,
            vector=v_recibo,
        ),
    ]
    n = upsert_chunks(points, batch_size=2)
    assert n == 3, n

    hits = search(v_alcohol, limit=5)
    assert hits[0]["snippet"] == "alcoholemia clausula exclusion v2", hits[0]
    assert hits[0]["carpeta_top"] == "Casos", hits[0]

    assert exists_by_path_mtime("Casos/2024/Perez/demanda.pdf", 1700000000.0)
    assert not exists_by_path_mtime("Casos/2024/Perez/demanda.pdf", 1699999999.0)

    # Hybrid: filename term "anexo" debe traer el anexo aunque el vector sea de alcohol
    hits_h = search_hybrid(v_alcohol, limit=5, filename_terms=["anexo"])
    paths = [h["path"] for h in hits_h]
    assert "Casos/2024/Perez/anexo.docx" in paths, f"filename boost fallo: {paths}"

    # list_carpetas_top
    global _carpetas_cache
    _carpetas_cache = None  # invalidar cache
    tops = list_carpetas_top()
    assert "Casos" in tops, f"tops={tops}"

    # search_filenames_similar
    sims = search_filenames_similar(["anexo"], limit=3)
    assert any("anexo" in (s.get("filename") or "").lower() for s in sims), sims

    delete_by_path("Casos/2024/Perez/demanda.pdf")
    hits2 = search(v_alcohol, limit=5)
    paths = {h["path"] for h in hits2}
    assert "Casos/2024/Perez/demanda.pdf" not in paths, hits2

    # _carpeta_top con VIRTUAL_ROOT
    global VIRTUAL_ROOT
    _orig = VIRTUAL_ROOT
    try:
        VIRTUAL_ROOT = "/volume1/Publico/Estudio"
        assert _carpeta_top("/volume1/Publico/Estudio/ARABELA/x.pdf") == "ARABELA"
        assert _carpeta_top("/volume1/Publico/Estudio/DOCUMENTOS CARO/y/z.pdf") == "DOCUMENTOS CARO"
    finally:
        VIRTUAL_ROOT = _orig

    print("qdrant_backend.demo OK")


if __name__ == "__main__":
    demo()
