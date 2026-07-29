"""Re-ranker semantico basado en embeddings.

Estrategia:
- Embeddings pre-computados por chunk se guardan en chunk_embeddings (BLOB float32).
- El modelo es multilingual (MiniLM L12 v2, 384-dim, ~470MB). Corre en GPU si hay,
  sino en CPU (aceptable para query 1x1 en runtime del NAS).
- Modo MOCK (env RERANKER_MOCK=1) usa un encoder bag-of-chars para tests
  determinsticos sin bajar el modelo real.
- Si USE_RERANKER=0 o sentence-transformers no esta instalado, el sistema cae
  automaticamente a solo FTS5.
"""
from __future__ import annotations

import logging
import os
import string
from typing import Iterable

import numpy as np

logger = logging.getLogger("rag.reranker")

MODEL_NAME = os.getenv(
    "RERANKER_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
_encoder = None
_encoder_dim: int | None = None


def is_enabled() -> bool:
    return os.getenv("USE_RERANKER", "1") == "1"


def is_mock() -> bool:
    return os.getenv("RERANKER_MOCK") == "1"


class _MockEncoder:
    """Encoder deterministico para tests. Vector = frecuencia de chars a-z0-9 L2-normalizado."""

    _ALPHA = string.ascii_lowercase + string.digits
    _IDX = {c: i for i, c in enumerate(_ALPHA)}
    dim = len(_ALPHA)

    def encode(self, texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for c in (t or "").lower():
                j = self._IDX.get(c)
                if j is not None:
                    out[i, j] += 1
            n = float(np.linalg.norm(out[i])) or 1.0
            out[i] /= n
        return out


def get_encoder():
    """Devuelve el encoder (lazy). Levanta si USE_RERANKER=0 o si no hay backend."""
    global _encoder, _encoder_dim
    if _encoder is not None:
        return _encoder
    if is_mock():
        _encoder = _MockEncoder()
        _encoder_dim = _encoder.dim
        logger.info("Reranker en modo MOCK (dim=%d)", _encoder_dim)
        return _encoder
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers no instalado. `pip install sentence-transformers`"
        ) from e
    device = "cpu"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass
    logger.info("Cargando reranker %s en %s (primera vez baja ~470MB)", MODEL_NAME, device)
    _encoder = SentenceTransformer(MODEL_NAME, device=device)
    _encoder_dim = int(_encoder.get_sentence_embedding_dimension())
    return _encoder


def encoder_dim() -> int:
    get_encoder()
    assert _encoder_dim is not None
    return _encoder_dim


def embed_texts(texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
    enc = get_encoder()
    lista = list(texts)
    if not lista:
        return np.zeros((0, encoder_dim()), dtype=np.float32)
    vecs = enc.encode(
        lista,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def vec_to_blob(v: np.ndarray) -> bytes:
    return v.astype(np.float32, copy=False).tobytes()


def blob_to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def rerank_by_cosine(
    query_vec: np.ndarray, candidatos: list[tuple[int, np.ndarray]]
) -> list[int]:
    """Devuelve rowids ordenados por similitud coseno descendente.

    Asume vectores ya L2-normalizados (embed_texts hace normalize_embeddings=True),
    por lo que el producto punto == cosine similarity.
    """
    if not candidatos:
        return []
    mat = np.stack([v for _, v in candidatos])
    sims = mat @ query_vec
    order = np.argsort(-sims)
    return [candidatos[int(i)][0] for i in order]


def demo() -> None:
    """Self-check en modo mock."""
    os.environ["RERANKER_MOCK"] = "1"
    global _encoder
    _encoder = None
    q = embed_query("alcoholemia exclusion")
    docs = embed_texts(
        [
            "alcoholemia exclusion de cobertura",  # muy similar
            "receta de milanesas napolitanas",  # nada que ver
            "exclusion de responsabilidad por alcohol",  # parcialmente similar
        ]
    )
    sims = docs @ q
    orden = np.argsort(-sims)
    assert orden[0] == 0, f"esperaba idx 0 primero, got {orden}"
    assert orden[-1] == 1, f"esperaba idx 1 ultimo (milanesas), got {orden}"
    print("reranker.demo OK, orden:", orden.tolist(), "sims:", sims.round(3).tolist())


if __name__ == "__main__":
    demo()
