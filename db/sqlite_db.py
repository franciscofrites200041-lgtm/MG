"""Memoria de chat persistente por session_id (chat_id de Telegram)."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import aiosqlite

logger = logging.getLogger("sqlite_db")

DB_PATH = os.getenv("MEMORY_DB_PATH", os.path.join(os.path.dirname(__file__), "memory.db"))


async def init_db() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_memory (
                session_id TEXT PRIMARY KEY,
                history TEXT
            )
            """
        )
        await db.commit()
    logger.info("Memoria SQLite lista: %s", DB_PATH)


async def get_history(session_id: str) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT history FROM chat_memory WHERE session_id = ?", (session_id,)
        )
        row = await cur.fetchone()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return []
        return []


async def save_history(session_id: str, history: list) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO chat_memory (session_id, history)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET history=excluded.history
            """,
            (session_id, json.dumps(history)),
        )
        await db.commit()
