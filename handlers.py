"""Router aiogram: texto, voz y archivos adjuntos de Telegram hacia el agente."""
from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import tempfile
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction
from aiogram.types import FSInputFile, Message

from agent import chat_with_agent
from utils.text_cleaner import dividir_mensaje, limpiar_texto
from utils.voice import transcribir_audio

logger = logging.getLogger("handlers")
router = Router()

# El tool generar_escrito devuelve un string tipo "DOCUMENTO_GENERADO: /path/x.docx"
_DOC_MARK = re.compile(r"DOCUMENTO_GENERADO:\s*([^\n]+\.docx)")

# --- Adjuntos ---
# Telegram bot API descarga hasta 20MB. Mas alla de eso, el usuario debe fragmentar.
_MAX_DOC_SIZE = int(os.getenv("DOC_MAX_SIZE_BYTES", str(20 * 1024 * 1024)))
# Limite de chars del texto extraido que se inyecta al agente. gpt-5 tiene mucho
# contexto pero cada token cuesta; 100k chars ~= 25k tokens ~= mayoria de escritos.
_DOC_MAX_CHARS = int(os.getenv("DOC_MAX_CHARS", "100000"))
_FMT_TEXTO = {".pdf", ".docx", ".doc", ".txt"}
_FMT_IMAGEN = {".jpg", ".jpeg", ".png", ".webp"}
_FMT_SOPORTADOS = _FMT_TEXTO | _FMT_IMAGEN
# Telegram fotos pueden pesar hasta 10MB; documentos hasta 20MB (bot API).
_MAX_IMG_SIZE = int(os.getenv("IMG_MAX_SIZE_BYTES", str(10 * 1024 * 1024)))
_MIME_POR_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


from rag.extractor import extraer_texto_de_archivo


async def _enviar_respuesta(m: Message, bot: Bot, reply: str) -> None:
    """Manda la respuesta al usuario, cortando por Telegram y enviando docs generados."""
    docs = _DOC_MARK.findall(reply)
    # Sacamos las marcas del texto que ve el humano.
    texto = _DOC_MARK.sub("", reply).strip()
    texto = limpiar_texto(texto)

    for parte in dividir_mensaje(texto):
        if parte:
            await m.answer(parte)

    for path in docs:
        p = Path(path.strip())
        if p.exists():
            try:
                await bot.send_document(m.chat.id, FSInputFile(str(p)))
            except Exception as e:
                logger.exception("Fallo enviando doc %s: %s", p, e)
                await m.answer(f"No pude enviar el documento generado: {e}")
        else:
            logger.warning("Doc marcado pero no existe: %s", p)


@router.message(F.text)
async def on_text(m: Message, bot: Bot) -> None:
    await bot.send_chat_action(m.chat.id, ChatAction.TYPING)
    reply = await chat_with_agent(str(m.chat.id), m.text or "")
    await _enviar_respuesta(m, bot, reply)


@router.message(F.voice | F.audio)
async def on_voice(m: Message, bot: Bot) -> None:
    await bot.send_chat_action(m.chat.id, ChatAction.RECORD_VOICE)

    file_id = m.voice.file_id if m.voice else m.audio.file_id
    tg_file = await bot.get_file(file_id)

    tmp_dir = os.getenv("VOICE_TMP", tempfile.gettempdir())
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    local = Path(tmp_dir) / f"{file_id}.ogg"
    await bot.download_file(tg_file.file_path, destination=str(local))

    texto = await transcribir_audio(str(local))
    if not texto:
        await m.answer("No pude transcribir el audio, mandame el mensaje por texto.")
        return

    await m.answer(f"Escuche: {texto}")
    await bot.send_chat_action(m.chat.id, ChatAction.TYPING)
    reply = await chat_with_agent(str(m.chat.id), texto)
    await _enviar_respuesta(m, bot, reply)


async def _descargar_bytes(bot: Bot, file_id: str) -> bytes:
    tg_file = await bot.get_file(file_id)
    buf = io.BytesIO()
    await bot.download_file(tg_file.file_path, destination=buf)
    return buf.getvalue()


async def _procesar_imagen(m: Message, bot: Bot, data: bytes, mime: str, source_name: str) -> None:
    caption = (m.caption or "").strip() or (
        "Describi juridicamente esta imagen. Si es documento, transcribi todo el texto legible. "
        "Si algo esta ilegible, decilo. No inventes."
    )
    prompt = f"[IMAGEN ADJUNTA: {source_name}]\n{caption}"
    logger.info("Imagen chat=%s source=%s size=%s mime=%s",
                m.chat.id, source_name, len(data), mime)
    reply = await chat_with_agent(str(m.chat.id), prompt, images=[(mime, data)])
    await _enviar_respuesta(m, bot, reply)


@router.message(F.photo)
async def on_photo(m: Message, bot: Bot) -> None:
    await bot.send_chat_action(m.chat.id, ChatAction.TYPING)
    # Telegram manda varias resoluciones; la ultima es la mejor.
    ph = m.photo[-1]
    if ph.file_size and ph.file_size > _MAX_IMG_SIZE:
        await m.answer(
            f"La foto pesa {ph.file_size / (1024*1024):.1f}MB, maximo "
            f"{_MAX_IMG_SIZE // (1024*1024)}MB. Reduci calidad y mandala de nuevo."
        )
        return
    try:
        data = await _descargar_bytes(bot, ph.file_id)
    except Exception as e:
        logger.exception("Fallo descargando foto: %s", e)
        await m.answer(f"No pude descargar la imagen: {e}")
        return
    await _procesar_imagen(m, bot, data, "image/jpeg", f"foto_{ph.file_unique_id}.jpg")


@router.message(F.document)
async def on_document(m: Message, bot: Bot) -> None:
    doc = m.document
    if doc is None:
        return
    fn = doc.file_name or f"{doc.file_id}"
    ext = Path(fn).suffix.lower()

    if ext not in _FMT_SOPORTADOS:
        await m.answer(
            f"Formato '{ext}' no soportado. Puedo leer PDF, DOCX, DOC, TXT y "
            "imagenes JPG/PNG/WEBP."
        )
        return

    # --- Imagen adjuntada como archivo ---
    if ext in _FMT_IMAGEN:
        if doc.file_size and doc.file_size > _MAX_IMG_SIZE:
            mb = doc.file_size / (1024 * 1024)
            await m.answer(
                f"La imagen pesa {mb:.1f}MB, maximo {_MAX_IMG_SIZE // (1024*1024)}MB."
            )
            return
        await bot.send_chat_action(m.chat.id, ChatAction.TYPING)
        try:
            data = await _descargar_bytes(bot, doc.file_id)
        except Exception as e:
            logger.exception("Fallo descargando imagen %s: %s", fn, e)
            await m.answer(f"No pude descargar la imagen: {e}")
            return
        mime = _MIME_POR_EXT.get(ext, "image/jpeg")
        await _procesar_imagen(m, bot, data, mime, fn)
        return

    # --- Documento de texto (PDF/DOCX/DOC/TXT) ---
    if doc.file_size and doc.file_size > _MAX_DOC_SIZE:
        mb = doc.file_size / (1024 * 1024)
        await m.answer(
            f"El archivo pesa {mb:.1f}MB, maximo {_MAX_DOC_SIZE // (1024 * 1024)}MB. "
            "Fragmentalo o mandame solo la seccion relevante."
        )
        return

    await bot.send_chat_action(m.chat.id, ChatAction.TYPING)

    tg_file = await bot.get_file(doc.file_id)
    tmp_dir = os.getenv("VOICE_TMP", tempfile.gettempdir())
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    local = Path(tmp_dir) / f"{doc.file_id}{ext}"
    try:
        await bot.download_file(tg_file.file_path, destination=str(local))
    except Exception as e:
        logger.exception("Fallo descargando doc %s: %s", fn, e)
        await m.answer(f"No pude descargar el archivo: {e}")
        return

    try:
        texto, n_paginas = await asyncio.to_thread(extraer_texto_de_archivo, local, ext)
    except Exception as e:
        logger.exception("Fallo extrayendo %s: %s", fn, e)
        await m.answer(f"No pude leer el archivo '{fn}': {e}")
        return
    finally:
        try:
            local.unlink(missing_ok=True)
        except Exception:
            pass

    if not texto:
        await m.answer(
            f"'{fn}' no tiene texto extraible. Probablemente sea un escaneo sin OCR. "
            "Podes mandarlo como IMAGEN (foto o adjunto .jpg/.png) para que lo lea con vision, "
            "o exportarlo con capa de texto."
        )
        return

    total_chars = len(texto)
    truncado = ""
    if total_chars > _DOC_MAX_CHARS:
        texto = texto[:_DOC_MAX_CHARS]
        truncado = (
            f"\n\n[NOTA: archivo truncado. Se enviaron {_DOC_MAX_CHARS // 1000}k de "
            f"{total_chars // 1000}k caracteres totales. Si necesitas otra seccion, "
            f"pedile al usuario que la mande especificamente.]"
        )

    caption = (m.caption or "").strip() or "Analiza este archivo."

    prompt = (
        f"[ARCHIVO ADJUNTO: {fn} ({n_paginas} paginas)]\n"
        f"{caption}\n\n"
        f"CONTENIDO DEL ARCHIVO:\n{texto}{truncado}"
    )

    logger.info("Doc adjunto chat=%s file=%s size=%s chars=%s truncado=%s",
                m.chat.id, fn, doc.file_size, total_chars, bool(truncado))

    reply = await chat_with_agent(str(m.chat.id), prompt)
    await _enviar_respuesta(m, bot, reply)


@router.message()
async def on_other(m: Message) -> None:
    await m.answer(
        "Puedo procesar: texto, audio, imagenes (JPG/PNG/WEBP) y archivos "
        "PDF/DOCX/DOC/TXT. Mandame tu consulta o adjuntame lo que necesites."
    )
