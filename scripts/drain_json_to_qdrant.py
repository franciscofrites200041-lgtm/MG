"""Drain: mueve JSONs del drop-folder de bg_worker.py -> Qdrant.

Lee cada .json (que ya tiene embeddings base64), arma ChunkPoints y upsertea.
Cuando termina cada archivo, lo mueve a done/ (o lo borra si --delete).

Uso:
    python scripts/drain_json_to_qdrant.py \
        --drop-dir C:\\Users\\franc\\.claude\\jobs\\e21051fc\\pilot_drop \
        --qdrant-url http://localhost:6333 \
        --collection mg_docs \
        --batch 512 \
        --delete

Corre en un loop hasta --once-terminado (si querés dejarlo tirado mientras bg_worker
genera archivos, sin --once).
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import qdrant_backend  # noqa: E402
from rag.qdrant_backend import ChunkPoint  # noqa: E402

logger = logging.getLogger("drain_qdrant")


def _payload_to_points(payload: dict, virtual_root: str, source_root: str) -> list[ChunkPoint]:
    """El JSON viene con 'path' ya mapeado a virtual (bg_worker lo hace).
    Chunks + embeddings alineados por indice."""
    chunks = payload.get("chunks", [])
    embeds = payload.get("embeddings", [])
    if not chunks or not embeds or payload.get("status") != "ok":
        return []
    filename = payload.get("filename", "")
    ext = payload.get("ext", "")
    mtime = float(payload.get("mtime", 0))
    size = int(payload.get("size", 0))
    path = payload["path"]

    seen_idx: dict[int, int] = {}
    out = []
    for c, e in zip(chunks, embeds):
        page = int(c["page"])
        text = c["text"]
        idx = seen_idx.get(page, 0)
        seen_idx[page] = idx + 1
        vec = _b64_to_vec(e["vec_b64"], e["dim"])
        out.append(ChunkPoint(
            path=path, filename=filename, ext=ext, page=page, chunk_idx=idx,
            snippet=text[:800], mtime=mtime, size=size, vector=vec,
        ))
    return out


def _b64_to_vec(b64: str, dim: int) -> list[float]:
    import array
    raw = base64.b64decode(b64)
    a = array.array("f")
    a.frombytes(raw)
    assert len(a) == dim, (len(a), dim)
    return list(a)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--drop-dir", required=True)
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "mg_docs"))
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--delete", action="store_true", help="Borrar JSON tras upsert (default: mover a done/)")
    p.add_argument("--once", action="store_true", help="Sale cuando no quedan JSONs (sino loopea cada 10s)")
    args = p.parse_args()

    os.environ["QDRANT_URL"] = args.qdrant_url
    os.environ["QDRANT_COLLECTION"] = args.collection
    # ponytail: rebind vars module-level (se leen al import, no en get_client)
    qdrant_backend.QDRANT_URL = args.qdrant_url
    qdrant_backend.COLLECTION = args.collection
    qdrant_backend._client = None  # forzar recreacion
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # ponytail: silenciar httpx/httpcore INFO (llena la pantalla con cada request)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    qdrant_backend.ensure_collection()
    drop = Path(args.drop_dir)
    done = drop / "done"
    if not args.delete:
        done.mkdir(exist_ok=True)

    t0 = time.time()
    total_files = 0
    total_points = 0
    import gc

    def _iter_jsons():
        """Iter lazy con scandir. Evita acumular 389k paths en memoria."""
        for e in os.scandir(drop):
            if e.is_file() and e.name.endswith(".json"):
                yield Path(e.path)

    while True:
        it = _iter_jsons()
        batch: list[ChunkPoint] = []
        batch_files: list[Path] = []
        any_seen = False
        for jf in it:
            any_seen = True
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                pts = _payload_to_points(data, "", "")
                batch.extend(pts)
                batch_files.append(jf)
                if len(batch) >= args.batch:
                    qdrant_backend.upsert_chunks(batch)
                    total_points += len(batch)
                    _dispose(batch_files, done, args.delete)
                    total_files += len(batch_files)
                    batch, batch_files = [], []
                    if total_files % 2000 == 0:
                        gc.collect()
                        dt = time.time() - t0
                        rate_f = total_files / dt if dt > 0 else 0
                        rate_p = total_points / dt if dt > 0 else 0
                        print(f"[{time.strftime('%H:%M:%S')}] {total_files:,} files, "
                              f"{total_points:,} points ({rate_f:.0f} f/s, {rate_p:.0f} p/s)",
                              flush=True)
            except Exception as e:
                logger.warning("Skip %s: %s", jf.name, type(e).__name__)
                if not args.delete:
                    try:
                        (done / (jf.name + ".bad")).write_bytes(jf.read_bytes())
                    except Exception:
                        pass
                jf.unlink(missing_ok=True)

        if batch:
            qdrant_backend.upsert_chunks(batch)
            total_points += len(batch)
            _dispose(batch_files, done, args.delete)
            total_files += len(batch_files)

        if not any_seen:
            if args.once:
                break
            print(f"[{time.strftime('%H:%M:%S')}] esperando JSONs...", flush=True)
            time.sleep(10)

        dt = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] ciclo fin: {total_files:,} files, "
              f"{total_points:,} points ({dt/60:.1f} min)", flush=True)
        if args.once:
            break


def _dispose(files: list[Path], done: Path, delete: bool) -> None:
    for f in files:
        if delete:
            f.unlink(missing_ok=True)
        else:
            f.rename(done / f.name)


def demo() -> None:
    """Self-check: escribe un JSON de bg_worker fake, drena a Qdrant :memory:, verifica."""
    import tempfile
    from qdrant_client import QdrantClient
    qdrant_backend._client = QdrantClient(":memory:")
    qdrant_backend.ensure_collection()

    with tempfile.TemporaryDirectory() as tmp:
        drop = Path(tmp)
        # Fake payload como el que genera bg_worker.py
        import numpy as np
        vec = np.random.randn(384).astype("float32")
        vec /= np.linalg.norm(vec)
        payload = {
            "path": "/volume1/Publico/Estudio/foo/x.pdf",
            "filename": "x.pdf", "ext": "pdf",
            "mtime": 1700000000.0, "size": 12345, "status": "ok", "error": None,
            "chunks": [{"page": 1, "text": "alcoholemia clausula"}, {"page": 2, "text": "recibo"}],
            "embeddings": [
                {"page": 1, "model": "test", "dim": 384, "vec_b64": base64.b64encode(vec.tobytes()).decode()},
                {"page": 2, "model": "test", "dim": 384, "vec_b64": base64.b64encode(vec.tobytes()).decode()},
            ],
        }
        (drop / "abc.json").write_text(json.dumps(payload), encoding="utf-8")

        pts = _payload_to_points(payload, "", "")
        assert len(pts) == 2, len(pts)
        assert pts[0].path == "/volume1/Publico/Estudio/foo/x.pdf"
        assert pts[0].page == 1 and pts[0].chunk_idx == 0
        assert len(pts[0].vector) == 384
        qdrant_backend.upsert_chunks(pts)

        hits = qdrant_backend.search(vec.tolist(), limit=5)
        assert any(h["path"] == "/volume1/Publico/Estudio/foo/x.pdf" for h in hits), hits
        print("drain_json_to_qdrant.demo OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()
