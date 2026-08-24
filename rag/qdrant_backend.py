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
import uuid
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger("rag.qdrant")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
COLLECTION = os.getenv("QDRANT_COLLECTION", "mg_docs")
VECTOR_SIZE = 384  # MiniLM-L12-v2

_client = None


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
        ("path", PayloadSchemaType.KEYWORD),
        ("carpeta_top", PayloadSchemaType.KEYWORD),
        ("mtime", PayloadSchemaType.FLOAT),
        ("page", PayloadSchemaType.INTEGER),  # para Range (contexto paginas +/- N)
    ]:
        try:
            c.create_payload_index(collection_name=COLLECTION, field_name=field, field_schema=schema)
        except Exception as e:
            logger.warning("Index %s ya existia: %s", field, str(e)[:80])

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
    """Primer segmento del path para agrupar. Ej: 'Casos/2024/Perez_Juan/x.pdf' -> 'Casos'."""
    p = path.lstrip("/").replace("\\", "/")
    return p.split("/", 1)[0] if "/" in p else ""


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


def exists_by_path_mtime(path: str, mtime: float) -> bool:
    """Dedup: True si ya hay al menos 1 punto con este path y mtime igual (tolerancia 1ms)."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

    c = get_client()
    q_filter = Filter(must=[
        FieldCondition(key="path", match=MatchValue(value=path)),
        FieldCondition(key="mtime", range=Range(gte=mtime - 0.001, lte=mtime + 0.001)),
    ])
    result = c.count(collection_name=COLLECTION, count_filter=q_filter, exact=False)
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

    delete_by_path("Casos/2024/Perez/demanda.pdf")
    hits2 = search(v_alcohol, limit=5)
    paths = {h["path"] for h in hits2}
    assert "Casos/2024/Perez/demanda.pdf" not in paths, hits2

    print("qdrant_backend.demo OK")


if __name__ == "__main__":
    demo()
