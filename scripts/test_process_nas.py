"""Self-check estructural de process_nas_ondemand.py sin rclone ni Qdrant reales."""
from __future__ import annotations
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["OPENAI_API_KEY"] = "test"

from scripts.process_nas_ondemand import (
    _is_supported, _mtime_epoch, _virtual_path, SUPPORTED_EXT, EXCLUDE_PREFIXES,
)


def test_supported_ext():
    assert _is_supported("foo/bar.pdf")
    assert _is_supported("Casos/Perez.docx")
    assert _is_supported("acta.doc")
    assert _is_supported("nota.txt")
    assert not _is_supported("foto.jpg")
    assert not _is_supported("hoja.xlsx")
    assert not _is_supported("Publico.zip")
    assert not _is_supported("Publico.zip.tmp")
    assert not _is_supported("dir/#recycle/foo.pdf")
    assert not _is_supported("@eaDir/thumb.jpg")
    print("[OK] _is_supported filtra extensiones y prefijos")


def test_virtual_path():
    os.environ["VIRTUAL_ROOT"] = "/volume1/Publico/Estudio"
    # Reimport para que tome la env
    from importlib import reload
    from scripts import process_nas_ondemand
    reload(process_nas_ondemand)
    assert process_nas_ondemand._virtual_path("Casos/Perez/x.pdf") == "/volume1/Publico/Estudio/Casos/Perez/x.pdf"
    assert process_nas_ondemand._virtual_path("x.pdf") == "/volume1/Publico/Estudio/x.pdf"
    # Windows separator
    assert process_nas_ondemand._virtual_path("Casos\\Perez\\x.pdf") == "/volume1/Publico/Estudio/Casos/Perez/x.pdf"
    print("[OK] _virtual_path preserva estructura NAS")


def test_mtime_parse():
    # Formato rclone lsjson: "2025-01-15T10:30:00.123456789-03:00"
    ts1 = _mtime_epoch("2025-01-15T10:30:00.000000000-03:00")
    assert ts1 > 1700000000, ts1
    ts2 = _mtime_epoch("2024-06-20T14:00:00.500000000+00:00")
    assert ts2 > 1700000000, ts2
    ts3 = _mtime_epoch("")
    assert ts3 == 0.0
    # Mismo momento en dos zonas -> mismo epoch
    a = _mtime_epoch("2025-01-15T13:30:00.000000000+00:00")
    b = _mtime_epoch("2025-01-15T10:30:00.000000000-03:00")
    assert abs(a - b) < 1, (a, b)  # tolera 1s
    print("[OK] _mtime_epoch parsea timestamps rclone con TZ")


def test_needs_process_no_qdrant():
    """Con Qdrant offline el fallback devuelve True (procesar ante duda)."""
    os.environ["QDRANT_URL"] = "http://127.0.0.1:19999"  # puerto inexistente
    from scripts.process_nas_ondemand import _needs_process
    from rag import qdrant_backend
    qdrant_backend._client = None
    assert _needs_process("/vol/x.pdf", 1700000000.0) is True
    print("[OK] _needs_process safe con Qdrant offline")


def test_exclude_prefixes():
    for p in EXCLUDE_PREFIXES:
        assert not _is_supported(f"{p}foo.pdf"), p
    print("[OK] EXCLUDE_PREFIXES bloquea basura del NAS")


if __name__ == "__main__":
    test_supported_ext()
    test_virtual_path()
    test_mtime_parse()
    test_exclude_prefixes()
    test_needs_process_no_qdrant()
    print("\nprocess_nas_ondemand self-check OK")
