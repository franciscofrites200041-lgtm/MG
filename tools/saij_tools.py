"""Scraping del Poder Judicial de Mendoza / SAIJ para jurisprudencia externa.

NOTA: los selectors reales viven en los subworkflows n8n originales:
- Buscar: workflowId isM1jzN03Sks9q9r (nombre "PJM")
- Leer:   workflowId 9qotp0bG30y4b869 (nombre "Leer sentencias SAIJ")

Este archivo implementa la interfaz que el agente ya conoce, con selectors
conservadores contra saij.gob.ar. Cuando exportes los subworkflows n8n vamos a
reemplazar la URL y los selectors por los exactos.

Nada mas del bot depende de esta implementacion: si falla, el agente responde
"la herramienta oficial no arrojo resultados" y sigue trabajando con RAG interno.
"""
from __future__ import annotations

import logging
import os
import re
import urllib.parse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("saij")

SEARCH_URL = os.getenv("SAIJ_SEARCH_URL", "https://www.saij.gob.ar/busqueda")
HTTP_TIMEOUT = float(os.getenv("SAIJ_TIMEOUT", "12"))


def _get(url: str) -> str | None:
    try:
        with httpx.Client(
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": "MontoyaGherziBot/1.0"},
            follow_redirects=True,
        ) as c:
            r = c.get(url)
            r.raise_for_status()
            return r.text
    except Exception as e:
        logger.warning("HTTP fallo %s: %s", url, e)
        return None


def buscar_jurisprudencia_mendoza(query: str) -> str:
    """Busca fallos en el portal oficial (SAIJ / Poder Judicial de Mendoza).

    IMPORTANTE: el buscador es antiguo. Pasale SOLO palabras clave sueltas,
    sin conectores. Ej: "9017", "Gomez Perez", "alcoholemia exclusion".

    Args:
        query: Palabras clave sueltas separadas por espacio. Max 5 tokens.

    Returns:
        Hasta 5 resultados en formato "Caratula: X | Enlace: URL", uno por linea.
        O mensaje explicito si no hay resultados.
    """
    q = (query or "").strip()
    if not q:
        return "La herramienta oficial no arrojo resultados para esos terminos exactos."

    url = f"{SEARCH_URL}?q={urllib.parse.quote(q)}"
    html = _get(url)
    if html is None:
        return "Error tecnico al consultar el buscador oficial."

    soup = BeautifulSoup(html, "html.parser")
    resultados = []

    # ponytail: patron conservador. Intentamos varias clases habituales de
    # sistemas judiciales viejos. Si ninguno matchea, devolvemos honesto.
    for sel in ("div.resultado", "li.resultado", "div.item-result", "article"):
        for item in soup.select(sel):
            a = item.find("a")
            if not a:
                continue
            titulo = (item.find(["h2", "h3"]) or a).get_text(strip=True) or "Sin titulo"
            href = a.get("href", "")
            if not href:
                continue
            if href.startswith("/"):
                base = re.match(r"(https?://[^/]+)", SEARCH_URL)
                if base:
                    href = base.group(1) + href
            resultados.append(f"Caratula: {titulo} | Enlace: {href}")
            if len(resultados) >= 5:
                break
        if resultados:
            break

    if not resultados:
        return "La herramienta oficial no arrojo resultados para esos terminos exactos."
    return "\n".join(resultados)


def leer_fallo_mendoza(url: str) -> str:
    """Descarga el texto completo de un fallo especifico dado su URL.

    Args:
        url: URL EXACTA obtenida por buscar_jurisprudencia_mendoza. Nunca inventar.

    Returns:
        Texto limpio del fallo (max ~25k chars) o mensaje de error.
    """
    if not url or not url.startswith("http"):
        return "URL invalida. Antes debes llamar a buscar_jurisprudencia_mendoza."

    html = _get(url)
    if html is None:
        return "Error tecnico al leer el fallo."

    soup = BeautifulSoup(html, "html.parser")

    contenido = None
    for sel in ("div.texto-completo", "div.document-content", "div#texto", "article", "main"):
        contenido = soup.select_one(sel)
        if contenido:
            break

    if contenido:
        for br in contenido.find_all("br"):
            br.replace_with("\n")
        texto = contenido.get_text(separator="\n", strip=True)
    else:
        # Fallback tosco: todo el body.
        texto = (soup.body.get_text(separator=" ", strip=True) if soup.body else "")

    if len(texto) > 25000:
        texto = texto[:25000] + "\n...[Fallo cortado por longitud]"
    return texto or "El documento fue descargado pero no se pudo extraer texto util."
