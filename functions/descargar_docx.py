"""
title: Descargar como DOCX
author: Estudio Montoya-Gherzi
author_url: https://github.com/franciscofrites200041-lgtm/MG
version: 2.0.0
required_open_webui_version: 0.3.0
requirements: httpx
description: Envia el ultimo mensaje del asistente al gateway, que genera un .docx con formato legal argentino y lo sirve como link publico.
"""

from __future__ import annotations
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel, Field


class Action:
    class Valves(BaseModel):
        gateway_url: str = Field(
            default="http://gateway:8000",
            description="URL interna del gateway (dentro de la docker network)",
        )
        public_base_url: str = Field(
            default="https://bot-estudio.montoya-gherzi.com.ar",
            description="URL publica del sitio (donde el nginx sirve /download/*)",
        )
        default_filename: str = Field(
            default="escrito_montoya_gherzi.docx",
            description="Nombre por defecto del archivo",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.icon = "📄"

    async def action(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __event_call__: Optional[Callable[[dict], Awaitable[Any]]] = None,
    ) -> Optional[dict]:
        import httpx

        msgs = body.get("messages", [])
        last_assistant = ""
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                last_assistant = (m.get("content") or "").strip()
                break

        if not last_assistant:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "notification",
                    "data": {"type": "warning", "content": "No hay mensaje del asistente para exportar."},
                })
            return None

        filename = self.valves.default_filename

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    f"{self.valves.gateway_url}/download/docx",
                    json={"text": last_assistant, "filename": filename},
                )
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "notification",
                    "data": {"type": "error", "content": f"Error generando DOCX: {type(e).__name__}"},
                })
            return None

        file_id = data.get("id")
        download_url = f"{self.valves.public_base_url}/download/docx/{file_id}?filename={filename}"

        if __event_emitter__:
            md = f"\n\n📄 [**Descargar {filename}**]({download_url})\n\n"
            await __event_emitter__({"type": "message", "data": {"content": md}})
            await __event_emitter__({
                "type": "notification",
                "data": {"type": "success", "content": "DOCX generado. Click en el link para descargar."},
            })
        return None
