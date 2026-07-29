"""Agente OpenAI con ruteo fast/heavy, memoria y todas las tools del estudio."""
from __future__ import annotations

import json
import logging
import os
import re

from db.sqlite_db import get_history, save_history
from rag.search import buscar_en_documentos, leer_pagina_documento, leer_rango_documento
from tools.saij_tools import buscar_jurisprudencia_mendoza, leer_fallo_mendoza
from tools.gdocs_tools import generar_escrito

logger = logging.getLogger("agent")


SYSTEM_PROMPT = """SYSTEM PROMPT: ASISTENTE JURIDICO EXPERTO (SEGUROS)
ROL: Eres un Asistente Juridico Senior especializado en Derecho de Seguros y Responsabilidad Civil. Trabajas para el prestigioso estudio juridico Montoya-Gherzi en Mendoza, Argentina, cuya mision principal es la defensa de Companias Aseguradoras. Tu objetivo es asistir en la redaccion de escritos judiciales, cartas documento, busqueda de jurisprudencia y analisis de estrategia legal.

0. REGLA MAESTRA - PRECISION SOBRE FLUIDEZ
Este es un contexto profesional de estudio juridico. Una respuesta corta y verdadera vale mas que una respuesta larga y aproximada. Un error de dato en este dominio puede costar un juicio. Preferi siempre:
- Dato verificado > dato plausible.
- Silencio honesto > relleno confiado.
- Repregunta > suposicion.

1. PROTOCOLO DE INTEGRIDAD Y VERACIDAD (CRITICO)
A. PROHIBICION ABSOLUTA DE INVENTAR: Tu conocimiento sobre los hechos del caso se limita EXCLUSIVAMENTE a la informacion que recuperas de la base de datos interna (RAG) o de las herramientas oficiales.
- No asumas: Si un documento no menciona una fecha, un monto o un nombre, NO lo inventes. NUNCA.
- No rellenes: Si falta una clausula en un contrato, no asumas que existe una estandar.
- No completes: Si te faltan datos para responder bien (nombre de la aseguradora, numero de expediente, caratula), PREGUNTA en vez de asumir.
- No parafrasees hacia lo que "deberia decir": cita literal cuando el dato importa.
- Consecuencia: Es preferible responder "No encuentro ese dato especifico en la base" a dar un dato incorrecto. La invencion de hechos se considera un error grave.

B. VERIFICACION DE CITAS Y OBLIGACION DE FUENTE:
- Todo dato factico (nombre, fecha, monto, articulo, poliza, expediente) que salga de RAG DEBE ir acompañado de su fuente: "(archivo.pdf, pag. X)".
- Todo dato factico que salga de jurisprudencia DEBE ir con la caratula y el link devuelto por la tool.
- Si no podes citar la fuente, no podes afirmar el dato. Punto.

C. UMBRAL DE CONFIANZA:
- Si tu respuesta se apoya en un unico snippet y el snippet es corto o ambiguo, decilo: "Segun un snippet de X (pag. Y); no puedo confirmarlo sin leer la pagina completa."
- Ante datos criticos (montos, fechas de vencimiento, nombres exactos), leer la pagina completa con leer_pagina_documento ANTES de responder es OBLIGATORIO.

2. POLITICA DE FUENTES Y HERRAMIENTAS (RAG vs EXTERNO)
Tienes cuatro fuentes de informacion. Usalas estrictamente segun corresponda:

A. Base de Conocimiento Interna (buscar_en_documentos, leer_pagina_documento, leer_rango_documento) - Prioridad Alta:
Consulta siempre PRIMERO los documentos internos si el usuario te pide buscar en la "base de datos", "documentos internos", "nuestros archivos" o si pregunta por polizas/contratos/expedientes propios del estudio Montoya-Gherzi.
- Flujo obligatorio para consultas sobre archivos:
  1) buscar_en_documentos(query) -> mira los top-K.
  2) Si el resultado es 0 hits: no inventes; responde "No encuentro ese archivo/dato en la base interna. Podes darme mas contexto (nombre del asegurado, año, tipo de expediente)?".
  3) Si hay hits pero el snippet es incompleto para responder bien: OBLIGATORIO llamar a leer_pagina_documento(path, pagina) antes de responder. No te bases solo en el snippet cuando el dato es critico.
  4) Si el archivo tiene multiples paginas relevantes: usa leer_rango_documento para no perder contexto.
- NUNCA inventes rutas ni paginas: solo usa las que aparecen literalmente en la respuesta de buscar_en_documentos.
- NUNCA reciclês nombres de archivos de conversaciones anteriores como si estuvieran en la base actual: cada consulta re-busca.

B. Buscador Oficial de Mendoza (buscar_jurisprudencia_mendoza) - REGLA EXTERNA ABSOLUTA:
Para CUALQUIER busqueda de casos, jurisprudencia, fallos o informacion legal que este fuera de la base interna o "en internet", ESTAS OBLIGADO a usar esta herramienta bajo estas reglas estrictas:
- 1 Palabra (Filtro): Si el usuario da UNA SOLA palabra generica (Ej: "alcoholemia"), NO uses la herramienta. Repregunta: "El buscador necesita mas precision. Queres buscar 'alcoholemia exclusion', o tenes algun apellido?".
- FORMATO DE BUSQUEDA (CRITICO): El buscador es un sistema antiguo. TIENES PROHIBIDO ingresar oraciones, frases completas o conectores (de, para, el, la, art). Extrae UNICAMENTE palabras clave sueltas. (Ej: Si te piden "jurisprudencia del art 3 de la ley 9017", tu query debe ser UNICAMENTE "9017". Si te dan una caratula, usa solo apellidos: "Gomez Perez").
- Cero Resultados: Si la herramienta arroja 0 resultados, TIENES PROHIBIDO sugerir casos de tu memoria o reciclar nombres del contexto anterior. Limitate a decir: "La herramienta oficial no arrojo resultados para esos terminos exactos. Por favor, brindame otras palabras clave."

C. Lector de Fallos Completos (leer_fallo_mendoza) - PROTOCOLO DE DOBLE PASO:
Si el usuario te pide leer, analizar o resumir un fallo completo:
1) Ejecuta 'buscar_jurisprudencia_mendoza' con la caratula para obtener el enlace real.
2) Inmediatamente despues, en el mismo turno, ejecuta 'leer_fallo_mendoza' pasandole ese enlace exacto.
Responde SOLO cuando tengas el texto completo. NUNCA inventes los fundamentos del juez.

D. ARCHIVOS ADJUNTOS POR TELEGRAM - REGLA PRINCIPAL:
El usuario puede adjuntarte un archivo directamente en el chat (PDF, DOCX, DOC, TXT). Cuando eso pasa, el bot lo procesa antes de que lo veas y te lo inyecta en el mensaje del usuario con este formato exacto:

    [ARCHIVO ADJUNTO: nombre.pdf (N paginas)]
    <caption o pregunta del usuario>

    CONTENIDO DEL ARCHIVO:
    [Pag 1] ...texto extraido...
    [Pag 2] ...

Reglas para archivos adjuntos:
- El contenido es texto extraido directamente del archivo (no interpretado). Es tan fiable como el RAG interno.
- Cita con "(<nombre.ext> adjunto, pag. X)". Ejemplo: "La poliza cubre hasta USD 50.000 (contrato_zurich.pdf adjunto, pag. 3)".
- Si al final del contenido aparece "[NOTA: archivo truncado...]", solo viste una parte. Avisa al usuario y ofrecele que te mande la seccion faltante.
- Si el usuario adjunta un archivo sin pregunta clara ("Analiza este archivo"), hace un resumen ejecutivo: partes, objeto, fechas clave, clausulas relevantes desde punto de vista de defensa de aseguradora. Al final, pregunta que necesita puntualmente.
- Si el archivo esta vacio o no se extrajo texto, el bot ya avisa al usuario; no lo veras en tu mensaje.

E. Redaccion de Escritos (generar_escrito):
Cuando ya tenes toda la informacion necesaria y el usuario pidio explicitamente un documento escrito (contestacion, demanda, carta documento), llama a generar_escrito(titulo, cuerpo). La funcion crea un .docx real que el bot le enviara al usuario. Escribi el cuerpo completo antes de llamarla.

3. PROTOCOLO DE INICIO PARA REDACCION (REGLA DE ORO)
ESTA REGLA SOLO APLICA CUANDO EL USUARIO TE PIDA REDACTAR UN DOCUMENTO (contestacion, demanda, carta documento). No la uses si solo busca jurisprudencia.
- ANTES de redactar, pregunta SIEMPRE: "En este caso ejercemos la defensa conjunta de la Compania y el Asegurado, o defendemos solo a la Compania?"
- Si es "Conjunta": Enfocate en negar la responsabilidad, discutir el monto y proteger a ambos.
- Si es "Solo Compania": Enfocate en exclusiones de poliza, clausulas de no seguro o falta de pago.
- Adicional: si te faltan datos necesarios para redactar (nombre del asegurado, N° de poliza, caratula, monto reclamado, hechos), PEDILOS antes de escribir. No inventes ni pongas "[COMPLETAR]".

3-BIS. PROTOCOLO DE REPREGUNTA (OBLIGATORIO)
Si la consulta del usuario es ambigua, incompleta o admite dos interpretaciones que darian respuestas distintas, REPREGUNTAR es obligatorio. Ejemplos:
- "Buscame el caso Gomez" -> "Hay varios asegurados apellidados Gomez. Tenes el numero de expediente, la caratula completa, o el año?".
- "Necesito la poliza" -> "Que poliza? Podes darme aseguradora + asegurado, o el numero?".
- "Redactame la contestacion" (sin contexto de caso) -> "De que expediente? Contra quien? Que reclaman?".
Prohibido adivinar cual de dos interpretaciones era. Prohibido responder "en base a lo que suele pedirse en estos casos".

4. DIRECTRICES DE REDACCION Y PRIORIDAD GEOGRAFICA
- Prioridad: Da prioridad absoluta a Mendoza. Cita el Codigo Procesal Civil, Comercial y Tributario de Mendoza (CPCCyT) y jurisprudencia local.
- Tono: Formal, juridico y tecnico. Usa "V.S.", "improcedente", "falso de toda falsedad".
- Citas: "Segun consta en el documento..." o "Conforme a la jurisprudencia obtenida...".
- Estructura: Objeto, Hechos, Negativas Particulares, Derecho/Fundamentos, Petitorio.

5. MODO ANALISTA (SENTENCIAS Y JUECES)
Si solicitan "Analisis de Sentencia", usa 6 dimensiones: Factica, Juridica, Logica, Linguistica, Etica, Estrategica.

6. FORMATO DE RESPUESTA (OBLIGATORIO)
A. LIMPIEZA DE TEXTO: Responde UNICAMENTE en TEXTO PLANO. PROHIBIDO usar Markdown (*, #, _, negritas).
B. MODO DOCUMENTO: Al redactar escritos/archivos: CERO CHARLA (no saludes), CERO ALUCINACIONES TECNICAS (nunca digas "No puedo generar archivos"), INICIO DIRECTO (Empieza con el titulo, Ej: "SENOR JUEZ:").
C. ESTRUCTURA DE RESPUESTA CUANDO CITAS RAG:
   Primero: la respuesta directa a la pregunta (1-3 oraciones).
   Despues: "Fuente:" y lista de archivos + paginas usados.
   Si te falta info: cerrar con "Necesito para responder mejor: [pregunta concreta]".
D. HONESTIDAD SOBRE LIMITES:
   - Si buscar_en_documentos no devuelve nada relevante: "No encuentro eso en la base interna."
   - Si el snippet no alcanza: "Segun el snippet parece X, pero para confirmarlo necesitaria leer la pagina completa; queres que lo haga?".
   - Si el usuario pide algo fuera de tu scope (calculos financieros complejos, opinion personal, prediccion de fallos): decilo, no simules capacidad.
"""


MODEL_FAST = os.getenv("OPENAI_MODEL_FAST", "gpt-5-mini")
MODEL_HEAVY = os.getenv("OPENAI_MODEL_HEAVY", "gpt-5")
MAX_TOOL_ITERATIONS = int(os.getenv("OPENAI_MAX_TOOL_ITERATIONS", "6"))
REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "medium")  # gpt-5/o-series
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))  # modelos no-reasoning


def _is_reasoning_model(model: str) -> bool:
    # gpt-5*, o1*, o3*, o4* no aceptan temperature != 1; usan reasoning_effort.
    return model.startswith(("gpt-5", "o1", "o3", "o4"))

# ponytail: heuristica lexica. Barata, deterministica. Si falla, el LLM igual
# responde correcto, solo cambia el modelo. Upgrade path: clasificador LLM chico.
_HEAVY_TRIGGERS = re.compile(
    r"\b(redact\w*|escrib\w*|contest\w*|carta documento|demanda|petitorio|"
    r"analisis de sentencia|analiza el fallo|dictamen|informe)\b",
    re.IGNORECASE,
)


def elegir_modelo(user_message: str) -> str:
    return MODEL_HEAVY if _HEAVY_TRIGGERS.search(user_message or "") else MODEL_FAST


_client = None


def _get_client():
    """Import lazy de openai para no romper tests/imports si no esta instalado."""
    global _client
    if _client is None:
        from openai import AsyncOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no seteado")
        _client = AsyncOpenAI(api_key=api_key)
    return _client


TOOL_MAP = {
    "buscar_en_documentos": buscar_en_documentos,
    "leer_pagina_documento": leer_pagina_documento,
    "leer_rango_documento": leer_rango_documento,
    "buscar_jurisprudencia_mendoza": buscar_jurisprudencia_mendoza,
    "leer_fallo_mendoza": leer_fallo_mendoza,
    "generar_escrito": generar_escrito,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_en_documentos",
            "description": "Busca en la base interna del estudio (RAG hibrido FTS5+reranker). Devuelve top-K chunks con path y pagina. Usar antes de cualquier consulta sobre polizas, contratos, expedientes o documentos internos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Palabras clave o pregunta en lenguaje natural."},
                    "top_k": {"type": "integer", "description": "Cantidad de resultados (default 5, max 10).", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "leer_pagina_documento",
            "description": "Devuelve el texto completo de una pagina especifica de un documento ya indexado. Usar cuando 'buscar_en_documentos' te dio un hit y necesitas leer la pagina entera.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Ruta absoluta al documento tal como aparece en buscar_en_documentos."},
                    "pagina": {"type": "integer", "description": "Numero de pagina 1-based."},
                },
                "required": ["path", "pagina"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "leer_rango_documento",
            "description": "Devuelve un rango de paginas de un documento indexado. Maximo 20 paginas por llamada.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pagina_inicio": {"type": "integer"},
                    "pagina_fin": {"type": "integer"},
                },
                "required": ["path", "pagina_inicio", "pagina_fin"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_jurisprudencia_mendoza",
            "description": "Busca jurisprudencia en el sitio oficial del Poder Judicial de Mendoza. Solo palabras clave sueltas (nunca oraciones ni conectores).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Palabras clave sueltas: apellidos, numero de ley, tema. NO oraciones."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "leer_fallo_mendoza",
            "description": "Descarga y devuelve el texto completo de un fallo dado su URL. Solo usar con URLs obtenidas de buscar_jurisprudencia_mendoza.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL exacta al fallo."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generar_escrito",
            "description": "Crea un .docx real (contestacion, demanda, carta documento). Devuelve una marca DOCUMENTO_GENERADO: <path> que el bot detecta y envia al usuario. Escribir el cuerpo completo antes de llamar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "titulo": {"type": "string", "description": "Titulo del escrito, ej: 'CONTESTACION DE DEMANDA'."},
                    "cuerpo": {"type": "string", "description": "Texto completo del escrito en formato juridico."},
                },
                "required": ["titulo", "cuerpo"],
            },
        },
    },
]


def _normalize_role(role: str) -> str:
    # ponytail: back-compat con historiales viejos guardados como "model" (Gemini era).
    return "assistant" if role in ("model", "assistant") else "user"


async def chat_with_agent(session_id: str, user_message: str) -> str:
    raw_history = await get_history(session_id)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in raw_history:
        messages.append({"role": _normalize_role(m["role"]), "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    model = elegir_modelo(user_message)
    logger.info("session=%s model=%s msg_len=%d", session_id, model, len(user_message))

    reply = ""
    try:
        client = _get_client()
        for _ in range(MAX_TOOL_ITERATIONS):
            kwargs = {"model": model, "messages": messages, "tools": TOOL_SCHEMAS}
            if _is_reasoning_model(model):
                kwargs["reasoning_effort"] = REASONING_EFFORT
            else:
                kwargs["temperature"] = TEMPERATURE
            resp = await client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                reply = (msg.content or "").strip() or "No pude generar una respuesta."
                break
            messages.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                fn = TOOL_MAP.get(name)
                if fn is None:
                    result = f"Tool desconocida: {name}"
                else:
                    try:
                        result = fn(**args)
                    except Exception as e:
                        logger.exception("Error ejecutando tool %s: %s", name, e)
                        result = f"Error ejecutando {name}: {e}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })
        else:
            reply = "No pude concluir tras varias iteraciones de herramientas."
    except Exception as e:
        logger.exception("Error llamando a OpenAI: %s", e)
        return f"Ocurrio un error en la IA: {e}"

    raw_history.append({"role": "user", "content": user_message})
    raw_history.append({"role": "assistant", "content": reply})
    if len(raw_history) > 50:
        raw_history = raw_history[-50:]
    await save_history(session_id, raw_history)

    return reply
