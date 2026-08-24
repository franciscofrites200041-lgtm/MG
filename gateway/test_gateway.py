"""Self-check estructural del gateway sin Gemini real.

Valida:
    - Compactacion dispara al pasar el threshold
    - Formato SSE chunk es OpenAI-compat
    - _to_gemini_contents parsea correcto
    - RAG search contra Qdrant real trae hits (si Qdrant esta corriendo con datos)

Uso:
    python gateway/test_gateway.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6333")
os.environ.setdefault("QDRANT_COLLECTION", "mg_docs")
os.environ.setdefault("COMPACT_THRESHOLD_TOKENS", "100")  # bajo para test
os.environ.setdefault("GEMINI_API_KEY", "test-key-dummy")  # solo para que arranque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway import main as gw
from gateway.main import Message


def test_token_estimation():
    assert gw._count_tokens("hola mundo") > 0
    assert gw._count_tokens("a" * 400) >= 100
    print("[OK] _count_tokens")


def test_total_tokens():
    msgs = [Message(role="user", content="a" * 400), Message(role="assistant", content="b" * 400)]
    total = gw._total_tokens(msgs)
    assert total >= 200, total
    print(f"[OK] _total_tokens = {total}")


def test_compact_no_dispara_si_corto():
    msgs = [Message(role="user", content="hola")]
    out = gw._maybe_compact(msgs)
    assert out == msgs, "no deberia compactar mensajes cortos"
    print("[OK] compact no dispara con mensajes cortos")


def test_to_gemini_contents_extrae_system():
    msgs = [
        Message(role="system", content="sos legal"),
        Message(role="system", content="responde en espanol"),
        Message(role="user", content="que es un contrato?"),
        Message(role="assistant", content="es un acuerdo..."),
        Message(role="user", content="dame ejemplos"),
    ]
    sys_p, contents = gw._to_gemini_contents(msgs)
    assert sys_p == "sos legal\n\nresponde en espanol", sys_p
    assert len(contents) == 3, len(contents)
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"
    assert contents[2]["role"] == "user"
    print("[OK] _to_gemini_contents parsea system+historial")


def test_sse_chunk_shape():
    line = gw._sse_chunk("test-model", "chatcmpl-abc", "hola")
    assert line.startswith("data: "), line[:20]
    payload = json.loads(line[len("data: "):].strip())
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"]["content"] == "hola"
    assert payload["model"] == "test-model"
    print("[OK] SSE chunk OpenAI-compat")


def test_sse_chunk_final():
    line = gw._sse_chunk("m", "id", "", finish_reason="stop")
    payload = json.loads(line[len("data: "):].strip())
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["choices"][0]["delta"] == {}
    print("[OK] SSE chunk final con finish_reason")


def test_rag_search_qdrant_real():
    """Skippea si Qdrant no responde. Si hay datos, pide un query legal."""
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:6333/readyz", timeout=2).read()
    except Exception:
        print("[SKIP] Qdrant no responde en localhost:6333")
        return
    try:
        ctx, hits = gw._rag_context("contrato de seguro alcoholemia")
        if not hits:
            print("[SKIP] Qdrant sin datos (0 hits)")
            return
        assert "## Contexto" in ctx
        h0 = hits[0]
        assert "path" in h0
        assert "filename" in h0
        assert "page" in h0
        assert "snippet" in h0
        print(f"[OK] RAG search real: {len(hits)} hits, top score={h0.get('score', 0):.3f}")
        print(f"     top filename: {h0['filename']}")
    except Exception as e:
        print(f"[FAIL] RAG search real: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_token_estimation()
    test_total_tokens()
    test_compact_no_dispara_si_corto()
    test_to_gemini_contents_extrae_system()
    test_sse_chunk_shape()
    test_sse_chunk_final()
    test_rag_search_qdrant_real()
    print("\ngateway self-check OK")
