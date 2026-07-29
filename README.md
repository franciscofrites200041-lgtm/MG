# Bot Montoya-Gherzi

Migración del workflow n8n (`MontoyaGherzi.json`) a Python puro. Reemplaza la
collection Qdrant perdida por **SQLite FTS5 + re-ranker semántico** — cero
infra de vectores, un solo archivo `.db`, corre en cualquier NAS con Docker.

## Arquitectura

```
[PC potente + RTX 3050]                     [NAS Docker]                [Telegram]
 index_full.py             ── copiar        mg-bot container   <────>   usuarios
 (una vez, 12-24h)         index.db         ├─ aiogram polling
   ├─ walk NAS                              ├─ agent.py (OpenAI gpt-5 / gpt-5-mini)
   ├─ extract PDFs/DOCX/DOC                 ├─ rag/search.py
   ├─ FTS5 upsert                           │    ├─ FTS5 top-20
   └─ embeddings GPU                        │    └─ reranker: embed query
      (bge-mini multilingual)               │        + cosine top-K
                                            ├─ tools/saij_tools.py
                                            ├─ tools/gdocs_tools.py
                                            └─ cron reindex.sh (nocturno)
                                                    │
                                                    └─> lee NAS_ROOT (SMB/mount)
```

**Pipeline de búsqueda por default:**

1. La query pasa por FTS5 → top-20 chunks candidatos (léxico + BM25).
2. Se embed la query con el reranker (MiniLM multilingual, 384-dim).
3. Cosine similarity contra los 20 embeddings pre-computados.
4. Reordenar → top-5 al agente.

Esto arregla las queries semánticas ("el asegurado no avisó a tiempo") que
FTS5 puro no encontraría, sin pagar API de embeddings ni levantar Qdrant.

**Adjuntos por Telegram**: además de RAG interno, el usuario puede mandar
PDF/DOCX/DOC/TXT directamente al chat (hasta 20MB). El bot los extrae con el
mismo pipeline (`rag/extractor.py`), los inyecta al mensaje como
`[ARCHIVO ADJUNTO: nombre.pdf (N paginas)]` y el agente responde con las
mismas reglas anti-alucinación. Si el archivo supera `DOC_MAX_CHARS` (default
100k caracteres), se trunca y se avisa al agente.

**Ruteo automático de modelos**: `elegir_modelo()` manda queries de búsqueda a
`gpt-5-mini` (rápido/barato) y redacciones/análisis de sentencia a `gpt-5`
(más caro y mejor razonando). Ambos overrideables por env
(`OPENAI_MODEL_FAST`, `OPENAI_MODEL_HEAVY`).

**Tool calling**: loop manual con `openai.AsyncOpenAI` sobre Chat Completions.
Máximo `OPENAI_MAX_TOOL_ITERATIONS` (default 6) ciclos herramienta→respuesta.
Si el modelo pide una tool inexistente o los argumentos rompen, se le
devuelve el error como resultado de la tool y sigue.

## Setup

### 1. Instalación

```bash
pip install -r requirements.txt
cp .env.example .env
# editar .env: TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, NAS_ROOT
```

Nota: `sentence-transformers` arrastra `torch` (~2GB). En Windows con GPU se
instala el sabor CUDA automáticamente; en el NAS (sin GPU) baja la variante
CPU.

### 2. Extracción total (PC potente, una vez)

```bash
# Con tu i5-12400 + 16GB + RTX 3050, 8 workers = ~12-24h para 400GB.
# Los embeddings agregan ~15-30 min más sobre la RTX 3050.
python scripts/index_full.py \
    --root "Z:\estudio_juridico" \
    --db ./index.db \
    --workers 8
```

Podés cortarlo y retomar con `--incremental`: no reprocesa lo hecho.

Si sólo querés FTS5 (sin re-ranker) por alguna razón: `--no-embed` en la CLI o
`USE_RERANKER=0` en el .env.

Cuando termina, `index.db` pesa aprox el texto extraído (~5-15% del original)
más los embeddings (~1.5KB por chunk × cantidad de chunks). Copiás el `.db` al
NAS a `data/index.db`.

### 3. Deploy al NAS (Synology/QNAP con Docker)

Estructura sugerida en el NAS:

```
/volume1/docker/mg-bot/
├── (código Python del repo)
├── .env
└── data/
    ├── index.db              ← copiado desde la PC
    ├── memory.db             ← se crea solo
    ├── escritos_generados/   ← docx generados por el agente
    └── logs/
```

`.env` en el NAS:
```
NAS_ROOT=/volume1/estudio_juridico
USE_RERANKER=1
```

Levantar:
```bash
cd /volume1/docker/mg-bot
docker compose up -d --build
docker logs -f mg-bot
```

El modelo del re-ranker (~470MB) se baja la primera vez que el bot lo usa y
queda cacheado en `~/.cache/huggingface/` del container (persistido si montás
un volume). En CPU el embed de la query tarda 30-100ms — imperceptible.

### 4. Cron nocturno de reindex incremental

En Panel de Control > Programador de tareas del Synology (o `crontab -e`):
```
0 3 * * * /volume1/docker/mg-bot/scripts/reindex.sh
```

El script detecta si el container `mg-bot` está corriendo y ejecuta el reindex
adentro. Solo procesa archivos con `mtime` o `size` distinto al indexado, y
computa embeddings para los chunks nuevos.

### 5. Google Docs (opcional)

Si querés que `generar_escrito` use un template propio del estudio en lugar
del fallback local:

1. Creá un Google Doc con placeholders `{{TITULO}}`, `{{CUERPO}}`, `{{FECHA}}`.
2. Copiá el ID del doc a `GDOCS_TEMPLATE_ID` en `.env`.
3. Creá una service account en GCP con scopes Drive + Docs, descargá el JSON.
4. Compartí el template con el email de la service account.
5. Poné el JSON en `data/service_account.json` y `GOOGLE_SA_JSON=/data/service_account.json`.

Si no configurás nada, el bot genera un `.docx` local usando python-docx y lo
manda igual — funciona, solo pierde el template corporativo.

## Tests

```bash
python tests/test_smoke.py
```

17/17 tests OK en <2s, sin llamadas LLM ni red. El re-ranker corre en modo mock
(bag-of-chars determinístico) para no bajar el modelo real en los smoke tests.
El agente OpenAI no se golpea en tests (validado solo que los tool schemas
coincidan con `TOOL_MAP` y que el ruteo Fast/Heavy funcione).

**Cobertura de extracción validada por tests:**
- DOCX con tablas + headers + footers → todo el texto queda en FTS5.
- TXT en encoding no-UTF (latin-1, cp1252, utf-16) → se detecta y decodea.
- PDF sin capa de texto (equivalente a escaneado) → se marca `status='sin_texto'`
  y queda listado, sin ensuciar el conteo de errores reales.

Para validar el modelo real (una vez instalado sentence-transformers):
```bash
python -m rag.reranker
```

## Módulos

| Archivo | Responsabilidad |
|---|---|
| `main.py` | Entry point aiogram. Init DBs + registra router. |
| `agent.py` | OpenAI Chat Completions + system prompt + ruteo Fast/Heavy + tool loop + tools registry. |
| `handlers.py` | Handlers Telegram: texto, voz (Whisper), adjuntos PDF/DOCX/DOC/TXT, envío de docs generados. |
| `rag/extractor.py` | Walk NAS + PyMuPDF (texto+tablas+orden por columnas) / python-docx (párrafos+tablas+headers+footers en orden natural) / antiword / TXT con auto-detección de encoding. Multiprocessing + FTS5 upsert + embeddings. Marca `status='sin_texto'` los PDFs sin capa de texto. |
| `rag/reranker.py` | Modelo sentence-transformers + cosine reranking + modo mock. |
| `rag/search.py` | Tools de búsqueda y lectura de páginas expuestos al agente. |
| `rag/schema.sql` | DDL FTS5 + tabla chunk_embeddings. |
| `tools/saij_tools.py` | Scraping SAIJ (buscar + leer). Selectors conservadores, pendientes exactos. |
| `tools/gdocs_tools.py` | Generación de escritos (Google Docs con fallback local). |
| `db/sqlite_db.py` | Memoria de chat por session_id. |
| `utils/text_cleaner.py` | Limpieza markdown + división para Telegram. |
| `utils/voice.py` | Whisper transcribe. |
| `scripts/index_full.py` | CLI para extracción total inicial. |
| `scripts/reindex_incremental.py` | CLI para el cron nocturno del NAS. |
| `scripts/reindex.sh` | Wrapper del cron. |

## Deuda técnica conocida

- `tools/saij_tools.py`: los selectors son conservadores. Para que devuelvan
  resultados exactos como el n8n original, hay que exportar los subworkflows
  `PJM` (isM1jzN03Sks9q9r) y `Leer sentencias SAIJ` (9qotp0bG30y4b869) del n8n
  y portar sus selectors/URLs reales.
