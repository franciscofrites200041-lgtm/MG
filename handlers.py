"""Router aiogram: texto, voz y archivos adjuntos de Telegram hacia el agente."""
from __future__ import annotations

import asyncio
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
_FMT_SOPORTADOS = {".pdf", ".docx", ".doc", ".txt"}


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


@router.message(F.document)
async def on_document(m: Message, bot: Bot) -> None:
    doc = m.document
    if doc is None:
        return
    fn = doc.file_name or f"{doc.file_id}"
    ext = Path(fn).suffix.lower()

    if ext not in _FMT_SOPORTADOS:
        await m.answer(
            f"Formato '{ext}' no soportado. Puedo leer PDF, DOCX, DOC y TXT."
        )
        return
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
        # to_thread: la extraccion es sync y puede tardar en PDFs grandes.
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
            f"'{fn}' no tiene texto extraible. Probablemente sea un escaneo sin OCR; "
            "necesito el archivo con capa de texto o que copies el contenido a mano."
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
        "Puedo procesar: texto, audio y archivos PDF/DOCX/DOC/TXT. "
        "Mandame tu consulta o adjuntame el archivo."
    )
