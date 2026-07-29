"""Router aiogram: texto y voz de Telegram hacia el agente."""
from __future__ import annotations

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


@router.message()
async def on_other(m: Message) -> None:
    await m.answer("Por ahora solo proceso texto y audio. Mandame tu consulta escrita o de voz.")
