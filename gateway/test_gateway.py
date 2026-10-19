"""Self-check estructural del gateway sin OpenAI real."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6333")
os.environ.setdefault("QDRANT_COLLECTION", "mg_docs")
os.environ.setdefault("COMPACT_THRESHOLD_TOKENS", "100")
os.environ.setdefault("OPENAI_API_KEY", "test-key-dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gateway import main as gw
from gateway.main import Message


def test_multimodal_text_normalization_and_images():
    parts = [
        {"type": "text", "text": "Revisa este archivo"},
        {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
        {"type": "file", "file": {"filename": "secret.xlsx", "file_data": "BASE64"}},
        {"type": "text", "text": "y responde"},
    ]
    assert gw._content_text(parts) == "Revisa este archivo\ny responde"
    assert gw._forward_content(parts) == [parts[0], parts[1], parts[3]]
    assert gw._count_tokens(parts) == gw._count_tokens("Revisa este archivo\ny responde")
    print("[OK] multimodal content only forwards text and image_url")


def test_bad_and_attachment_only_content():
    assert gw._content_text([{"type": "file", "file": {"file_data": "BASE64"}}]) == ""
    assert gw._content_text([{"type": "unknown", "value": "ignored"}]) == ""
    assert gw._content_text([{"type": "text", "text": ""}]) == ""
    assert gw._forward_content([{"type": "file", "file": {"file_data": "BASE64"}}]) == ""
    assert gw._forward_content([{"type": "image_url", "image_url": {"url": "data:application/pdf;base64,c2VjcmV0"}}]) == ""
    assert gw._forward_content([{"type": "image_url", "image_url": {"url": "https://"}}]) == ""
    assert gw._forward_content([{"type": "image_url", "image_url": {"url": "https://["}}]) == ""
    assert gw._forward_content([{"type": "image_url", "image_url": {"url": "data:image/png;base64,A"}}]) == ""
    valid_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9JQ0YAAAAASUVORK5CYII="
    assert gw._forward_content([{"type": "image_url", "image_url": {"url": valid_png}}]) == [{"type": "image_url", "image_url": {"url": valid_png}}]
    assert gw._forward_content([{"type": "image_url", "image_url": {"url": "https://example.test/image.jpg", "detail": {"file_data": "BASE64"}}}]) == [{"type": "image_url", "image_url": {"url": "https://example.test/image.jpg"}}]
    print("[OK] malformed/unknown/attachment-only content is controlled")


def test_legacy_string_content():
    assert gw._content_text("consulta") == "consulta"
    assert gw._forward_content("consulta") == "consulta"
    print("[OK] legacy string content")


def test_message_rejects_non_string_or_list_content():
    try:
        Message(role="user", content={"type": "text", "text": "consulta"})
    except Exception:
        print("[OK] message rejects object content")
    else:
        raise AssertionError("message content must be a string or a list of parts")


def test_http_multimodal_content_is_normalized_and_streamed():
    captured = []
    rag_queries = []

    class FakeCompletions:
        def create(self, **kwargs):
            captured.append(kwargs)
            if kwargs.get("stream"):
                return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="respuesta"))])])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="respuesta"))])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    original_client, original_rag = gw._openai_client, gw._rag_context
    gw._openai_client = lambda: fake_client
    gw._rag_context = lambda query: (rag_queries.append(query) or "", [])
    payload = {
        "model": "mg-bot",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Revisa el Excel"},
            {"type": "image_url", "image_url": {"url": "https://example.test/image.jpg"}},
            {"type": "file", "file": {"file_data": "not-forwarded"}},
        ]}],
    }
    try:
        with TestClient(gw.app) as client:
            response = client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == "respuesta"
            response = client.post("/v1/chat/completions", json={**payload, "stream": True})
            assert response.status_code == 200, response.text
            assert "data: [DONE]" in response.text
            response = client.post("/v1/chat/completions", json={
                "model": "mg-bot",
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.test/image.jpg"}},
                ]}],
            })
            assert response.status_code == 200, response.text
            assert captured[-1]["model"] == gw.OPENAI_MODEL
    finally:
        gw._openai_client, gw._rag_context = original_client, original_rag

    forwarded = captured[0]["messages"][-1]["content"]
    assert forwarded == [payload["messages"][0]["content"][0], payload["messages"][0]["content"][1]]
    assert captured[1]["messages"][-1]["content"] == forwarded
    assert rag_queries == ["Revisa el Excel", "Revisa el Excel"]
    assert captured[-1]["messages"][0]["content"] == gw.SYSTEM_PROMPT_LEGAL
    print("[OK] HTTP multimodal non-stream and stream requests")


def test_token_estimation():
    assert gw._count_tokens("hola mundo") > 0
    assert gw._count_tokens("a" * 400) >= 100
    print("[OK] _count_tokens")


def test_total_tokens():
    msgs = [Message(role="user", content="a" * 400), Message(role="assistant", content="b" * 400)]
    assert gw._total_tokens(msgs) >= 200
    print("[OK] _total_tokens")


def test_compact_no_dispara_si_corto():
    msgs = [Message(role="user", content="hola")]
    assert gw._maybe_compact(msgs) == msgs
    print("[OK] compact no dispara con mensajes cortos")


def test_sse_chunk_shape():
    line = gw._sse_chunk("test-model", "chatcmpl-abc", "hola")
    payload = json.loads(line[len("data: "):].strip())
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"]["content"] == "hola"
    print("[OK] SSE chunk OpenAI-compat")


def test_sse_chunk_final():
    line = gw._sse_chunk("m", "id", "", finish_reason="stop")
    payload = json.loads(line[len("data: "):].strip())
    assert payload["choices"][0]["finish_reason"] == "stop"
    print("[OK] SSE final")


if __name__ == "__main__":
    test_multimodal_text_normalization_and_images()
    test_bad_and_attachment_only_content()
    test_legacy_string_content()
    test_message_rejects_non_string_or_list_content()
    test_http_multimodal_content_is_normalized_and_streamed()
    test_token_estimation()
    test_total_tokens()
    test_compact_no_dispara_si_corto()
    test_sse_chunk_shape()
    test_sse_chunk_final()
