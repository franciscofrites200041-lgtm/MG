# MG Bot — Documentación operativa completa

Bot de consulta legal del Estudio Montoya-Gherzi. Corre en un VPS con Docker
Compose y expone una UI tipo ChatGPT (Open WebUI) que consulta ~500k chunks de
documentos internos del estudio (PDFs, DOCX, DOC, TXT) via RAG semántico +
GPT‑4o.

Esta doc está pensada para que **alguien que nunca vio el proyecto** pueda:

1. Entender qué corre dónde y por qué.
2. Diagnosticar los problemas más frecuentes.
3. Volver a levantar el stack desde cero si el VPS se pierde.
4. Saber qué respaldar para no perder datos.

---

## Índice

1. [Arquitectura del stack](#1-arquitectura-del-stack)
2. [Servicios (uno por uno)](#2-servicios-uno-por-uno)
3. [Flujo de una consulta](#3-flujo-de-una-consulta)
4. [Flujo de ingesta de documentos](#4-flujo-de-ingesta-de-documentos-nas--qdrant)
5. [Estructura del repositorio](#5-estructura-del-repositorio)
6. [Variables de entorno (.env)](#6-variables-de-entorno-env)
7. [Deploy desde cero en un VPS nuevo](#7-deploy-desde-cero-en-un-vps-nuevo)
8. [Backups (qué, dónde, cómo)](#8-backups-qué-dónde-cómo)
9. [Restauración desde backup](#9-restauración-desde-backup)
10. [Operación diaria](#10-operación-diaria)
11. [Cron jobs](#11-cron-jobs)
12. [Troubleshooting](#12-troubleshooting)
13. [Inventario de secretos](#13-inventario-de-secretos)

---

## 1. Arquitectura del stack

Todo corre en **un solo VPS Hostinger KVM 4** (4 vCPU / 16 GB RAM / 200 GB
NVMe / Ubuntu 24.04). Orquestado por **Docker Compose** (`docker-compose.yml`).

```
                             +----- Internet ------+
                             |                     |
                             v                     |
        +----------------------------------+       |
        |  nginx (80/443)                  |<------+
        |  - TLS Let's Encrypt             |
        |  - proxy a openwebui y gateway   |
        +---+--------------------------+---+
            |                          |
            | / (todo lo demas)        | /download/  (DOCX generados)
            v                          v
        +---------+              +-----------+
        | openwebui| ---LLM API--->|  gateway  |
        | :8080    |              |  :8000    |
        | (UI +   |              | (FastAPI, |
        | auth +  |              | OpenAI-   |
        | historial)              | compat)   |
        +---------+              +-----+-----+
                                       |
                                       | vector search + payload
                                       v
                                 +-----------+
                                 |  qdrant   |
                                 |  :6333    |
                                 | (~500k    |
                                 |  points)  |
                                 +-----+-----+
                                       |
                                       ^
                                       |  upsert cada hora
                                       |
                     +-----------------------------------+
                     | cron host: process_nas_ondemand   |
                     | (docker exec gateway)             |
                     |   - rclone lsjson SFTP → NAS      |
                     |   - filtra vs Qdrant              |
                     |   - baja, extrae, embedea, upsert |
                     +-----------------+-----------------+
                                       |
                                       | rclone SFTP
                                       v
                                 +-----------+
                                 |  NAS      |
                                 | Synology  |
                                 | (fuente   |
                                 |  de docs) |
                                 +-----------+
```

**Idea clave:** el VPS **no** guarda los PDFs originales. Solo tiene los
**chunks de texto extraídos + embeddings** en Qdrant. El NAS es la fuente de
verdad de los documentos; el VPS es la fuente de verdad del índice.

---

## 2. Servicios (uno por uno)

Todos definidos en `docker-compose.yml`. Levantan con `docker compose up -d`.

### 2.1 `qdrant` — Vector DB

- **Imagen:** `qdrant/qdrant:v1.11.3` (pineada, no bumpear sin probar).
- **Puerto:** `127.0.0.1:6333` (solo localhost del VPS; no expuesto a
  internet).
- **Datos persistidos en volúmenes Docker:**
  - `qdrant_data` → `/qdrant/storage` (la base entera, ~5–10 GB con 500k
    chunks + vectores 384-dim).
  - `qdrant_snapshots` → `/qdrant/snapshots` (dumps para backup).
- **Colección:** `mg_docs` (nombre configurable con `QDRANT_COLLECTION`).
- **Un punto (point) por chunk**, con:
  - `vector`: float32[384] (MiniLM multilingual, cosine distance).
  - `payload`: `{path, filename, ext, page, chunk_idx, snippet, mtime, size, carpeta_top}`.
  - `id`: UUID5 determinístico de `(path, page, chunk_idx)` → upsert idempotente.
- **Índices sobre payload** (creados por `qdrant_backend.ensure_collection()`
  al arranque del gateway):
  - `path`: KEYWORD + TEXT (match exacto + substring).
  - `filename`: TEXT (hybrid search por nombre de archivo).
  - `carpeta_top`: KEYWORD (filtro por carpeta top-level del NAS).
  - `page`: INTEGER (para range `page ± N`).
  - `mtime`: FLOAT (dedup por `(path, mtime)` al ingestar).

### 2.2 `gateway` — RAG Gateway

- **Imagen:** built local desde `gateway/Dockerfile`.
- **Base:** `python:3.11-slim` + `antiword` + `rclone` + `openssh-client`.
- **Framework:** FastAPI + uvicorn.
- **Puerto interno:** `8000` (NO expuesto al host — solo red interna Docker).
- **Función:** expone endpoints **OpenAI-compat** para que Open WebUI lo
  consuma como si fuera OpenAI, pero por atrás mete RAG semántico y valida
  citas.
- **Endpoints:**
  - `POST /v1/chat/completions` — el que Open WebUI llama en cada mensaje.
    Streaming SSE.
  - `GET /v1/models` — lista mock con un solo modelo (`mg-bot`).
  - `GET /health` — estado (Qdrant OK, cantidad de points, OpenAI config).
  - `POST /debug/query` — dump de hits para debugging.
  - `POST /download/docx` + `GET /download/docx/{id}` — genera y sirve un
    `.docx` con formato legal argentino (Times 12pt, márgenes procesales).
    Los DOCX se generan en `/tmp/mg_docx` con cleanup > 1h.
- **Volúmenes:**
  - `hf_cache` → `/root/.cache/huggingface` (persiste el modelo del reranker
    y del cross-encoder, ~500MB c/u; primer arranque los baja de Hugging
    Face).
  - `/data:/data` (bind del host) — logs de `process_nas_ondemand`.
  - `/home/mg/.config/rclone:/root/.config/rclone:ro` (config de rclone).
  - `/home/mg/.ssh:/home/mg/.ssh:ro` (llave SSH para SFTP al NAS).
- **Dependencia:** `qdrant` (compose lo levanta antes).

### 2.3 `openwebui` — UI + auth + historial

- **Imagen:** `ghcr.io/open-webui/open-webui:main` (main, se auto-actualiza en
  cada rebuild).
- **Puerto:** `127.0.0.1:3000:8080` (solo localhost; nginx lo proxea).
- **Volumen:** `openwebui_data` → `/app/backend/data` (SQLite con **usuarios,
  chats, historial, avatares**). **Este es el volumen más crítico a
  respaldar** — perderlo = perder toda la conversación histórica de todos
  los abogados.
- **Config clave** (variables en compose):
  - `OPENAI_API_BASE_URL=http://gateway:8000/v1` (le habla al gateway).
  - `OPENAI_API_KEY=dummy-not-used-but-required` (Open WebUI exige la
    variable, pero el gateway no la valida).
  - `WEBUI_NAME=Estudio Montoya-Gherzi` (branding).
  - `ENABLE_SIGNUP=true` + `DEFAULT_USER_ROLE=pending`: cualquiera puede
    registrarse pero queda **pending** hasta que un admin lo apruebe desde
    Panel > Users. **Primer usuario que se registra es admin automáticamente.**
  - `ENABLE_OLLAMA_API=false`, `ENABLE_IMAGE_GENERATION=false`,
    `ENABLE_WEB_SEARCH=false` (features apagadas, no las necesitamos).

### 2.4 `nginx` — Reverse proxy + TLS

- **Imagen:** `nginx:1.27-alpine`.
- **Puertos:** `80` (redirect a HTTPS + ACME challenge) y `443` (TLS).
- **Config:** `nginx/nginx.conf`.
- **TLS:** certificados de Let's Encrypt en `./certbot_certs` (bind mount).
- **Rutas:**
  - `/` → `openwebui:8080` (con WebSocket upgrade para streaming).
  - `/download/` → `gateway:8000` (para descargar los DOCX generados).
  - `/.well-known/acme-challenge/` → filesystem (para renovación certbot).
  - `/healthz` → devuelve `ok\n`.
- **sub_filter:** reescribe "Open WebUI" a "Estudio Montoya-Gherzi" en HTML
  y JS servidos (branding). Requiere `proxy_set_header Accept-Encoding "";`
  para que el upstream no comprima.
- **Dominio:** `bot-estudio.montoya-gherzi.com.ar` (hardcoded en la config).
- **Body size:** `client_max_body_size 50M` (para subir PDFs a chats).

### 2.5 `certbot` — Renovación TLS (comentado por default)

Se activa recién después del primer `certonly` manual (ver [Deploy](#7-deploy-desde-cero-en-un-vps-nuevo)).
Corre `certbot renew` cada 12h. Persiste en el mismo bind mount
`certbot_certs`.

---

## 3. Flujo de una consulta

Cuando un abogado tipea "qué dice la cláusula de exclusión por alcoholemia":

```
1. Browser  ── HTTPS ──>  nginx (443)
2. nginx    ──────────>  openwebui:8080  (WebSocket para el stream)
3. openwebui llama gateway:8000/v1/chat/completions (OpenAI-compat)
4. gateway.main.chat_completions():
   a. Detecta el intent:
      - "trivial" (saludo, gracias) → skip RAG, gpt-4o-mini directo
      - "escrito" (redactar demanda) → activa template de escrito legal
      - "consulta" (default) → RAG completo
   b. Si NO es trivial, llama _rag_context(query):
      - _expand_query(): pide 3 variantes a gpt-4o-mini
      - Por cada variante: reranker.embed_query() + qdrant.search(limit=60)
      - _filename_hits(): matches por keyword en filename (hybrid)
      - Merge, dedupe por (path, page), boost por keyword
      - _rerank() con cross-encoder → top 10 hits
      - _expand_context(): para cada hit, trae ± CONTEXT_RADIUS páginas
   c. Arma el contexto:
      SYSTEM_PROMPT_LEGAL + "## CONTEXTO — extractos" + hits formateados
   d. _maybe_compact(): si historial > COMPACT_THRESHOLD_TOKENS,
      gpt-4o-mini resume los viejos y quedan últimos 4 + resumen
   e. Stream de gpt-4o con temperature=0.1
   f. Post-stream: _validate_citations() verifica que cada
      "(archivo.pdf, pag. X)" del texto matchee un filename real
      de los hits. Si no, agrega footer ⚠️ con las inválidas.
   g. Si intent=escrito, genera el DOCX y agrega link al final:
      "📄 [Descargar como Word](https://.../download/docx/{id})"
5. Cada chunk SSE vuelve por el mismo camino (gateway → openwebui → nginx → browser)
6. openwebui persiste el chat completo en openwebui_data
```

**Modelo por operación:**

| Operación                | Modelo         | Por qué                                    |
|--------------------------|----------------|--------------------------------------------|
| Query expansion          | `gpt-4o-mini`  | Reformular es tarea barata                 |
| Historial compactación   | `gpt-4o-mini`  | Resumir texto en español es fácil          |
| Respuesta legal / RAG    | `gpt-4o`       | Precisión importa, no puede alucinar       |
| Trivial (hola, gracias)  | `gpt-4o-mini`  | Un saludo no necesita 4o                   |

---

## 4. Flujo de ingesta de documentos (NAS → Qdrant)

Los abogados dropean archivos al NAS (Synology). El VPS no ve el NAS
directamente; usa **rclone SFTP** para bajarlos on-demand.

**Script:** `scripts/process_nas_ondemand.py`.
**Corre:** vía cron del host, `docker exec mg-gateway python /app/scripts/process_nas_ondemand.py`
cada hora (ver [Cron jobs](#11-cron-jobs)).

**Por cada corrida:**

```
1. rclone lsf --dirs-only nas:Publico  →  top-level dirs
   (excluye #recycle, #snapshot, @eaDir, Thumbs.db, ~$*, Publico.zip)

2. Para cada dir top-level:
   rclone lsjson --recursive --files-only → lista de archivos con mtime/size
   (timeout 30 min por dir; los que timeoutean se saltean)

3. Por cada archivo listado:
   a. Filtro por extensión soportada: .pdf .docx .doc .txt
   b. Traduce path local → virtual (agrega prefijo VIRTUAL_ROOT
      = /volume1/Publico/Estudio para consistencia con puntos
      históricos ya en Qdrant)
   c. qdrant_backend.exists_by_path_mtime(virtual, mtime):
      - Si ya está indexado con ese mtime → SKIP
      - Si no o cambió → CANDIDATO

4. Por cada candidato:
   a. rclone copyto  nas:Publico/… → /tmp/nas_ondemand/{uuid}.ext
   b. rag.extractor.process_one(tmp):
      - PDF: PyMuPDF (fitz), un chunk por página, con tablas + orden
        multi-columna
      - DOCX: python-docx (párrafos + tablas + headers/footers en
        orden natural), chunked por ~500 palabras
      - DOC: antiword (binario), chunked por ~500 palabras
      - TXT: auto-detección de encoding (utf-8/latin-1/cp1252/utf-16)
      - Si el PDF no tiene capa de texto → status='sin_texto', se
        registra pero NO se sube a Qdrant
   c. reranker.embed_texts(chunks) → vector 384-dim por chunk
      (MiniLM multilingual, CPU en el VPS ~50ms/chunk)
   d. qdrant_backend.upsert_chunks(): batch de 64 con ChunkPoint
      (id determinístico → si un archivo cambió, upsert reemplaza
      los chunks viejos automáticamente)
   e. Borra /tmp/nas_ondemand/{uuid}.ext
```

**Checkpoint:** en `/data/logs/nas_ondemand_last.json` se persiste
la última ejecución con contadores (listados, chequeados, candidatos,
ok, err). Útil para monitoreo.

**Timeouts y resiliencia:**
- Chunk timeout: 1800s por dir top-level (`RCLONE_CHUNK_TIMEOUT`).
- Un `flock` externo (en cron) impide que dos corridas se apilen si
  una anterior está corriendo.

---

## 5. Estructura del repositorio

```
MG/                                     (raíz del repo git)
├── MontoyaGherzi.json                 archivo n8n histórico (legacy)
└── bot/                                todo el stack corre desde acá
    ├── docker-compose.yml              orquesta los 4 servicios
    ├── .env                            secretos y config (NO se commitea)
    ├── .env.example                    template de referencia
    │
    ├── gateway/                        el corazón: RAG + LLM proxy
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   ├── main.py                     FastAPI app
    │   ├── test_gateway.py             checks estructurales (compaction, SSE)
    │   ├── test_integration.py         checks end-to-end contra Qdrant real
    │   └── test_precision.py           checks de precisión de citas
    │
    ├── nginx/
    │   └── nginx.conf                  TLS + reverse proxy + branding
    │
    ├── rag/                            librería compartida
    │   ├── __init__.py
    │   ├── qdrant_backend.py           wrapper de qdrant-client (idx, upsert, search)
    │   ├── reranker.py                 sentence-transformers (embed_query/embed_texts)
    │   ├── extractor.py                PDF/DOCX/DOC/TXT → chunks
    │   ├── search.py                   API de búsqueda de alto nivel (legacy compat SQLite)
    │   ├── query_analyzer.py           extrae carpeta/expedientes/nombres de la query (legacy)
    │   └── schema.sql                  DDL SQLite FTS5 (legacy)
    │
    ├── scripts/                        utilidades operativas
    │   ├── process_nas_ondemand.py     CRON: ingesta rclone → Qdrant (activo)
    │   ├── backfill_carpeta_top.py     one-off: rellena payload carpeta_top
    │   └── count_nas.py                utilitario: contar archivos indexables del NAS
    │
    └── data/                           artefactos locales (persistidos en git como muestra)
        ├── index.db                    SQLite FTS5 histórico (compat legacy)
        └── index_test.db               idem, versión de test
```

**Volúmenes Docker gestionados** (no en el repo, viven en el VPS):

| Volumen              | Contenido                                       | Peso aprox. |
|----------------------|-------------------------------------------------|-------------|
| `qdrant_data`        | Base Qdrant completa                            | 5–10 GB     |
| `qdrant_snapshots`   | Dumps para restore                              | 5–10 GB     |
| `openwebui_data`     | SQLite de Open WebUI (usuarios, chats)          | crece con uso |
| `hf_cache`           | Modelos MiniLM + cross-encoder                  | ~1 GB       |

**Bind mounts del host:**

- `/data` (host) → `/data` (gateway). Guarda `logs/`, checkpoints,
  `service_account.json`, DOCX temporales fallback.
- `/home/mg/.config/rclone` → `/root/.config/rclone` (config rclone).
- `/home/mg/.ssh` → `/home/mg/.ssh` (llave privada SSH del NAS).
- `./certbot_certs` → `/etc/letsencrypt` (certificados TLS).
- `./certbot_www` → `/var/www/certbot` (ACME challenges).

---

## 6. Variables de entorno (.env)

Ver `.env.example` para el template. Referencia completa:

### Obligatorias

| Variable                | Ejemplo / default                | Descripción                                            |
|-------------------------|----------------------------------|--------------------------------------------------------|
| `OPENAI_API_KEY`        | `sk-...`                         | Key de OpenAI (proyecto MG). **Sin esto no arranca.**  |
| `OPENAI_MODEL`          | `gpt-4o`                         | Modelo principal para respuestas legales.              |
| `OPENAI_SUMMARIZE_MODEL`| `gpt-4o-mini`                    | Modelo para resumir historial al compactar.            |
| `OPENAI_QUERY_MODEL`    | `gpt-4o-mini`                    | Modelo para expansion de queries y saludos.            |
| `RAG_BACKEND`           | `qdrant`                         | Fijo en prod. `sqlite` es legacy.                      |
| `QDRANT_URL`            | `http://qdrant:6333`             | URL interna al container Qdrant.                       |
| `QDRANT_COLLECTION`     | `mg_docs`                        | Nombre de la colección.                                |
| `MODEL_NAME_EXPOSED`    | `mg-bot`                         | Nombre que ve el usuario en Open WebUI.                |
| `PUBLIC_BASE_URL`       | `https://bot-estudio…com.ar`     | Se usa para armar links de descarga de DOCX.           |

### Retrieval tuning (sanos por default)

| Variable                 | Default | Qué hace                                              |
|--------------------------|--------:|-------------------------------------------------------|
| `QDRANT_TOP_K`           | 10      | Cantidad final de hits que van al LLM.                |
| `QDRANT_POOL`            | 60      | Candidatos que trae Qdrant antes del dedupe.          |
| `RERANK_TOP_N`           | 30      | Cuántos pasan por el cross-encoder.                   |
| `SNIPPET_CHARS`          | 1800    | Chars máx por snippet enviado al LLM.                 |
| `CONTEXT_RADIUS`         | 1       | Trae páginas ± N alrededor del hit.                   |
| `COMPACT_THRESHOLD_TOKENS`| 12000  | Al pasar este umbral, resume historial.               |
| `ENABLE_QUERY_EXPANSION` | 1       | 0 = una sola query, 1 = 3 variantes.                  |
| `ENABLE_CROSS_ENCODER`   | 1       | 0 = sin rerank, 1 = con cross-encoder.                |
| `USE_RERANKER`           | 1       | 0 = sin embeddings (no funciona el search).           |

### LLM behavior

| Variable                  | Default | Qué hace                                                                                    |
|---------------------------|--------:|---------------------------------------------------------------------------------------------|
| `OPENAI_TEMPERATURE`      | 0.1     | Solo se aplica a modelos NO-reasoning (gpt-4o, gpt-4.1). Ignorado en gpt-5/o1/o3/o4.        |
| `OPENAI_REASONING_EFFORT` | (vacío) | Solo se aplica a modelos reasoning. Valores: `minimal` / `low` / `medium` / `high`. |

### Web search (Open WebUI)

| Variable                          | Default      | Qué hace                                          |
|-----------------------------------|--------------|---------------------------------------------------|
| `WEB_SEARCH_ENGINE`               | `duckduckgo` | Backend de búsqueda (ver [sección Web search](#106-web-search-en-open-webui)). |
| `WEB_SEARCH_RESULT_COUNT`         | 5            | Cantidad de resultados por búsqueda.              |
| `WEB_SEARCH_CONCURRENT_REQUESTS`  | 10           | Requests concurrentes al scrapear resultados.     |

### Ingesta NAS

| Variable                | Default                     | Qué hace                                       |
|-------------------------|-----------------------------|------------------------------------------------|
| `RCLONE_REMOTE`         | `nas`                       | Nombre del remote en `rclone.conf`.            |
| `NAS_SHARE_PATH`        | `Publico`                   | Share que se sincroniza.                       |
| `VIRTUAL_ROOT`          | `/volume1/Publico/Estudio`  | Prefix que ve el usuario en las citas.         |
| `TMP_DIR`               | `/tmp/nas_ondemand`         | Working dir para descargas temporales.         |
| `RCLONE_CHUNK_TIMEOUT`  | 1800 (s)                    | Timeout por top-level dir.                     |

---

## 7. Deploy desde cero en un VPS nuevo

Asumiendo: VPS Ubuntu 24.04 fresco, dominio ya apuntando a la IP.

### 7.1 SSH + hardening

```bash
ssh root@TU_IP_VPS
apt update && apt upgrade -y
apt install -y ufw fail2ban curl git

# Usuario no-root
adduser mg
usermod -aG sudo mg

# Firewall
ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw enable

# Reboot si el kernel se actualizó
[ -f /var/run/reboot-required ] && reboot
```

### 7.2 Docker

```bash
curl -fsSL https://get.docker.com | sh
usermod -aG docker mg
docker compose version   # verifica que quedó instalado
```

### 7.3 Clonar repo

```bash
sudo mkdir -p /opt/mg-bot
sudo chown -R mg:mg /opt/mg-bot
su - mg
cd /opt/mg-bot
git clone https://github.com/franciscofrites200041-lgtm/MG.git .
cd bot/
```

### 7.4 Configurar `.env`

```bash
cp .env.example .env
nano .env
```

Setear como mínimo:
- `OPENAI_API_KEY` (obligatorio).
- Verificar `PUBLIC_BASE_URL` con el dominio real.

### 7.5 Config rclone (para ingesta del NAS)

Necesitamos que el container gateway pueda hablar por SFTP al NAS. rclone
config vive en el **host**, se bind-monta read-only al container.

```bash
# En el host, como usuario mg
mkdir -p ~/.config/rclone ~/.ssh

# Generar keypair (o copiar la existente)
ssh-keygen -t ed25519 -f ~/.ssh/nas_sftp -N ""

# Copiar la clave pública al NAS (autorizar acceso SFTP)
# En Synology: Panel > Usuario > Editar > Servicios avanzados > agregar SSH key
# O manualmente vía DSM

# Configurar rclone
rclone config
# n → new remote
# name: nas
# type: sftp
# host: IP_DEL_NAS
# user: TU_USUARIO_SFTP
# port: 22
# key_file: /home/mg/.ssh/nas_sftp
# use_insecure_cipher: false
# md5sum_command / sha1sum_command: none
# q → quit

# Testear
rclone lsd nas:
# Debería listar Publico y demás shares
```

### 7.6 Estructura de dirs del host

```bash
sudo mkdir -p /data/logs /data/nas_mirror
sudo chown -R mg:mg /data
```

### 7.7 Levantar el stack (sin TLS todavía)

Para el primer arranque, el TLS aún no está. Comentá temporalmente el bloque
HTTPS de `nginx/nginx.conf` (o dejalo — sin certificados, nginx no arranca).

```bash
cd /opt/mg-bot/bot
docker compose up -d qdrant
docker compose logs -f qdrant  # esperar "Qdrant HTTP listening on 6333"
# Ctrl+C

docker compose up -d gateway
docker compose logs -f gateway  # esperar "Application startup complete"
# Ctrl+C

docker exec mg-gateway curl -s http://localhost:8000/health
# {"status":"ok","qdrant_ok":true,"qdrant_points":0,"openai_configured":true,...}
```

### 7.8 Restaurar snapshot Qdrant (si existe)

Si venís de un VPS anterior, subí el snapshot antes de meter openwebui/nginx.
Ver [Restauración](#9-restauración-desde-backup).

Si es instalación fresca sin datos, saltate este paso — el cron de
`process_nas_ondemand.py` va a llenar Qdrant con el tiempo.

### 7.9 Certificado TLS (Let's Encrypt)

Con nginx **apagado**:

```bash
docker compose stop nginx  # (por si estaba)
docker run --rm -it -p 80:80 \
  -v /opt/mg-bot/bot/certbot_certs:/etc/letsencrypt \
  certbot/certbot certonly --standalone \
  -d bot-estudio.montoya-gherzi.com.ar \
  --email TU_EMAIL@dominio.com --agree-tos --no-eff-email
```

Verificar que quedaron en `/opt/mg-bot/bot/certbot_certs/live/bot-estudio.…/`.

Descomentá el bloque HTTPS de `nginx/nginx.conf` si lo habías comentado.

### 7.10 Levantar todo

```bash
docker compose up -d
docker compose ps
# los 4 servicios deberían estar Up y healthy
```

Test:

```bash
curl -I https://bot-estudio.montoya-gherzi.com.ar/
# HTTP/2 200
```

### 7.11 Crear primer usuario admin

Abrir `https://bot-estudio.montoya-gherzi.com.ar/` en el browser. El primer
signup queda como **admin** automáticamente. Los siguientes signups quedan
como `pending` y hay que aprobarlos desde `Menu > Admin Panel > Users`.

### 7.12 Habilitar cron de ingesta

```bash
sudo crontab -u mg -e
# Agregar:
0 * * * * flock -n /tmp/nas_ondemand.lock docker exec mg-gateway python /app/scripts/process_nas_ondemand.py >> /data/logs/nas_ondemand.log 2>&1
```

Ejecutar una vez manual para probar:

```bash
docker exec mg-gateway python /app/scripts/process_nas_ondemand.py --preflight
# Todos los checks deben decir [OK]
```

### 7.13 (Opcional) Auto-renovación TLS

Descomentar el servicio `certbot` en `docker-compose.yml` y:

```bash
docker compose up -d certbot
```

---

## 8. Backups (qué, dónde, cómo)

Prioridad de qué se pierde si no se respalda:

| Recurso                       | Qué se pierde                            | Recuperable?                    |
|-------------------------------|------------------------------------------|---------------------------------|
| Volume `openwebui_data`       | Usuarios + historial de todos los chats  | **NO** — irrecuperable          |
| Volume `qdrant_data`          | Índice vectorial (500k chunks)           | Sí, pero ~48h de re-indexado    |
| `/opt/mg-bot/bot/.env`        | Config + API keys                        | Sí (regenerar keys)             |
| `~/.config/rclone/rclone.conf`| Config SFTP al NAS                       | Sí (regenerar)                  |
| `~/.ssh/nas_sftp`             | Llave privada para el NAS                | Sí (regenerar + reautorizar)    |
| `certbot_certs/`              | Certificados TLS                         | Sí (regenerar con certbot)      |
| NAS Synology                  | **Los documentos originales**            | Backup propio del NAS           |

### 8.1 Backup diario recomendado (cron)

Guión (`/usr/local/bin/mg-backup.sh`):

```bash
#!/bin/bash
set -euo pipefail
STAMP=$(date +%Y%m%d-%H%M)
DEST=/opt/backups/$STAMP
mkdir -p "$DEST"

# 1. Snapshot Qdrant (via API, evita tocar files locked)
docker exec mg-qdrant sh -c \
  "curl -s -X POST http://localhost:6333/collections/mg_docs/snapshots" \
  | tee "$DEST/qdrant_snapshot_meta.json"
SNAP=$(docker exec mg-qdrant ls -1t /qdrant/snapshots/ | head -1)
docker cp mg-qdrant:/qdrant/snapshots/$SNAP "$DEST/"

# 2. Volume openwebui_data (SQLite de usuarios/chats)
docker run --rm \
  -v openwebui_data:/data:ro \
  -v "$DEST":/backup \
  alpine tar czf /backup/openwebui_data.tgz -C /data .

# 3. Config plana (.env, rclone, ssh, certbot)
tar czf "$DEST/config.tgz" \
  -C /opt/mg-bot/bot .env docker-compose.yml nginx/nginx.conf certbot_certs \
  -C /home/mg .config/rclone .ssh/nas_sftp

# 4. Retención: borrar snapshots > 14 días
find /opt/backups -maxdepth 1 -type d -mtime +14 -exec rm -rf {} +
```

Cron:

```
0 4 * * * /usr/local/bin/mg-backup.sh >> /var/log/mg-backup.log 2>&1
```

**Además:** copiar el backup a un **almacenamiento fuera del VPS** (S3, otro
NAS, Google Drive via rclone). Un backup en el mismo VPS no sirve si el VPS
se pierde.

Ejemplo con rclone:

```bash
# Agregar al final del script:
rclone copy /opt/backups/$STAMP remotebackup:mg-bot-backups/$STAMP
```

### 8.2 Backup on-demand rápido

```bash
docker exec mg-qdrant sh -c "curl -s -X POST http://localhost:6333/collections/mg_docs/snapshots"
docker cp mg-qdrant:/qdrant/snapshots/. /tmp/qdrant-$(date +%Y%m%d)/
docker run --rm -v openwebui_data:/data:ro -v /tmp:/out alpine tar czf /out/owui-$(date +%Y%m%d).tgz -C /data .
```

---

## 9. Restauración desde backup

Asumiendo VPS nuevo con el stack levantado (ver [Deploy](#7-deploy-desde-cero-en-un-vps-nuevo))
y **Qdrant vacío**.

### 9.1 Restaurar Qdrant

```bash
# Copiar el .snapshot al VPS
scp qdrant_snapshot-YYYYMMDD.snapshot mg@VPS_IP:/tmp/

# Levantar Qdrant
docker compose up -d qdrant

# Copiar snapshot dentro del container
docker cp /tmp/qdrant_snapshot-YYYYMMDD.snapshot mg-qdrant:/qdrant/snapshots/

# Restore vía API
SNAP=$(docker exec mg-qdrant ls -1 /qdrant/snapshots/ | head -1)
curl -X PUT "http://127.0.0.1:6333/collections/mg_docs/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d "{\"location\": \"file:///qdrant/snapshots/$SNAP\"}"

# Verificar
curl -s http://127.0.0.1:6333/collections/mg_docs \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['result']['points_count'])"
```

### 9.2 Restaurar Open WebUI (usuarios + chats)

```bash
# Con openwebui APAGADO
docker compose stop openwebui

# Crear el volumen vacío si no existe
docker volume create openwebui_data

# Vaciar y restaurar
docker run --rm \
  -v openwebui_data:/data \
  -v /tmp/backup:/backup:ro \
  alpine sh -c "rm -rf /data/* /data/..?* /data/.[!.]* 2>/dev/null; tar xzf /backup/openwebui_data.tgz -C /data"

# Levantar
docker compose up -d openwebui
```

### 9.3 Verificación post-restore

```bash
docker exec mg-gateway curl -s http://localhost:8000/health
# qdrant_points debe coincidir con el conteo previo
```

Loguearse con la cuenta admin previa (los users vinieron con el volumen).

---

## 10. Operación diaria

### 10.1 Ver logs

```bash
# Todo en vivo
docker compose logs -f

# Solo un servicio
docker compose logs -f gateway
docker compose logs -f qdrant
docker compose logs -f openwebui
docker compose logs -f nginx

# Últimas N líneas
docker compose logs --tail=200 gateway

# Cron ingesta
tail -f /data/logs/nas_ondemand.log
```

### 10.2 Restart un servicio

```bash
docker compose restart gateway    # aplica cambios de .env
docker compose restart nginx      # aplica cambios de nginx.conf
```

### 10.3 Deploy de cambios de código

```bash
cd /opt/mg-bot
git pull
cd bot/
docker compose build gateway
docker compose up -d gateway
docker compose logs -f gateway
```

Cambios que **no** requieren rebuild:
- `.env` → `docker compose restart gateway`
- `nginx/nginx.conf` → `docker compose restart nginx`
- Templates de scripts en `scripts/*.py` → surten efecto en la próxima
  ejecución del cron (se leen desde el bind mount `/data` no, están dentro
  de la imagen — sí requiere rebuild)

### 10.4 Panel admin Open WebUI

- Aprobar signups pendientes: `Menu > Admin Panel > Users` → cambiar rol
  `pending` → `user`.
- Bloquear signups nuevos: setear `ENABLE_SIGNUP=false` en `.env` y
  `docker compose up -d openwebui` (recreate).
- Modelo visible en el selector: `mg-bot` (el `MODEL_NAME_EXPOSED`).

### 10.5 Cambiar el modelo / el nivel de razonamiento

Todos los modelos vienen de OpenAI. El stack tiene 3 slots:

| Slot                    | Env var                  | Uso                                          |
|-------------------------|--------------------------|----------------------------------------------|
| Principal (respuestas)  | `OPENAI_MODEL`           | Respuestas legales, generación de escritos.  |
| Query expansion         | `OPENAI_QUERY_MODEL`     | 3 variantes semánticas por query.            |
| Historial compactación  | `OPENAI_SUMMARIZE_MODEL` | Resumen al superar el threshold.             |

**Para cambiar el modelo principal** (ej. subir de `gpt-4o` a `gpt-5`):

```bash
nano /opt/mg-bot/bot/.env
# Setear:
# OPENAI_MODEL=gpt-5
# OPENAI_REASONING_EFFORT=medium
docker compose restart gateway
docker compose logs -f gateway   # verificar que arranca sin errores
```

**Cómo funciona el ruteo de parámetros:**

El gateway detecta automáticamente si el modelo es de razonamiento
(`gpt-5*`, `o1*`, `o3*`, `o4*`) o no. Según eso:

- **Modelos NO-reasoning** (`gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, etc.):
  se manda `temperature=$OPENAI_TEMPERATURE` (default 0.1). `OPENAI_REASONING_EFFORT`
  se ignora.
- **Modelos reasoning** (`gpt-5`, `o1`, `o3`, `o4`, sus variantes `-mini`, etc.):
  se manda `reasoning_effort=$OPENAI_REASONING_EFFORT`. `OPENAI_TEMPERATURE`
  se ignora (los reasoning solo aceptan `temperature=1` y OpenAI rechaza otros).

**Niveles válidos de reasoning:**

| Nivel       | Latencia | Calidad     | Costo   | Cuándo usarlo                                     |
|-------------|----------|-------------|---------|---------------------------------------------------|
| `minimal`   | rápida   | básica      | mínimo  | tareas triviales, prototipos                      |
| `low`       | moderada | correcta    | bajo    | consultas simples, resúmenes                      |
| `medium`    | media    | buena       | medio   | **default recomendado** para respuestas legales   |
| `high`      | lenta    | máxima      | alto    | casos difíciles, análisis con muchos snippets     |

Regla práctica: para análisis legales complejos con muchos documentos en el
contexto, `medium` alcanza casi siempre. Reservá `high` para consultas
puntuales donde precisión > velocidad.

**Para cambiar solo el modelo auxiliar** (ej. bajar costos en query expansion):

```bash
# Ej. usar gpt-5-mini en vez de gpt-4o-mini para queries
nano .env
# OPENAI_QUERY_MODEL=gpt-5-mini
# OPENAI_REASONING_EFFORT=low   # el mismo se aplica a todos los reasoning
docker compose restart gateway
```

**Verificación:** el modelo activo se ve en `/health`:

```bash
docker exec mg-gateway curl -s http://localhost:8000/health | jq
# {"model": "gpt-5", ...}
```

### 10.6 Web search en Open WebUI

El stack tiene web search **habilitado por default** con **DuckDuckGo**
(no requiere API key). El usuario activa/desactiva por mensaje con el ícono
del globo 🌐 en la caja de texto del chat.

**Cambiar de engine:**

Setear `WEB_SEARCH_ENGINE` en `.env` y `docker compose up -d openwebui`
(recreate). Opciones soportadas:

| Engine        | Requiere API key      | Cómo configurar                                                                 |
|---------------|-----------------------|--------------------------------------------------------------------------------|
| `duckduckgo`  | No                    | Default. Nada más que setear.                                                  |
| `tavily`      | Sí (Tavily)           | Agregar `TAVILY_API_KEY=...`                                                    |
| `serper`      | Sí (Serper)           | Agregar `SERPER_API_KEY=...`                                                    |
| `brave`       | Sí (Brave Search)     | Agregar `BRAVE_SEARCH_API_KEY=...`                                              |
| `google_pse`  | Sí (Google Custom Search) | Agregar `GOOGLE_PSE_API_KEY=...` + `GOOGLE_PSE_ENGINE_ID=...`               |
| `jina`        | Sí (Jina)             | Agregar `JINA_API_KEY=...`                                                      |
| `searxng`     | No (self-hosted)      | Agregar `SEARXNG_QUERY_URL=https://tu-searxng/search?q=<query>`                 |
| `mojeek`      | Sí                    | `MOJEEK_SEARCH_API_KEY=...`                                                     |
| `bing`        | Sí                    | `BING_SEARCH_V7_ENDPOINT=...` + `BING_SEARCH_V7_SUBSCRIPTION_KEY=...`           |
| `exa`         | Sí                    | `EXA_API_KEY=...`                                                               |
| `serply`      | Sí                    | `SERPLY_API_KEY=...`                                                            |

Tras cambiar cualquier valor:

```bash
nano .env
docker compose up -d openwebui
docker compose logs openwebui | grep -i "web.search"
```

**Deshabilitar web search por completo:**

En `docker-compose.yml`, servicio `openwebui`:

```yaml
ENABLE_WEB_SEARCH: "false"
ENABLE_RAG_WEB_SEARCH: "false"
```

Y `docker compose up -d openwebui`.

**Ajustar cantidad de resultados / concurrencia:**

- `WEB_SEARCH_RESULT_COUNT` (default 5): más = más contexto pero más lento.
- `WEB_SEARCH_CONCURRENT_REQUESTS` (default 10): cuántas páginas scrapea en
  paralelo. Bajarlo si el ancho de banda del VPS es limitado.

**Advertencia** para el estudio: los resultados de web search NO tienen la
validación de citas del RAG interno. El LLM puede citar páginas web sin
verificar. Si es importante, aclarar en el prompt del usuario que solo cite
lo que aparece textual en los resultados.

### 10.7 Health check

```bash
# Interno
docker exec mg-gateway curl -s http://localhost:8000/health

# Externo
curl -I https://bot-estudio.montoya-gherzi.com.ar/
curl -s https://bot-estudio.montoya-gherzi.com.ar/healthz  # "ok"
```

### 10.8 Espacio en disco

```bash
df -h /
docker system df
du -sh /var/lib/docker/volumes/*
```

Volúmenes que crecen:
- `qdrant_data`: crece con la ingesta.
- `qdrant_snapshots`: crece con cada backup. Purgar con:
  ```bash
  docker exec mg-qdrant sh -c "find /qdrant/snapshots -type f -mtime +7 -delete"
  ```
- `hf_cache`: fijo (~1 GB).
- `/data/logs/*.log`: pueden crecer sin límite. Agregar rotación:
  ```
  # /etc/logrotate.d/mg-bot
  /data/logs/*.log {
      daily
      rotate 14
      compress
      missingok
      notifempty
  }
  ```

---

## 11. Cron jobs

Todos corren como usuario `mg` en el **host** (no en containers).

```cron
# /var/spool/cron/crontabs/mg

# Ingesta hourly del NAS
0 * * * * flock -n /tmp/nas_ondemand.lock docker exec mg-gateway python /app/scripts/process_nas_ondemand.py >> /data/logs/nas_ondemand.log 2>&1

# Backup diario 04:00
0 4 * * * /usr/local/bin/mg-backup.sh >> /var/log/mg-backup.log 2>&1

# Cleanup snapshots viejos del volumen Qdrant (opcional si el backup ya los baja)
30 4 * * * docker exec mg-qdrant sh -c "find /qdrant/snapshots -type f -mtime +7 -delete"
```

**Notas:**
- `flock -n` en la ingesta evita que dos corridas se apilen si una tarda >1h.
- El renew de certbot no está en cron: lo hace el propio container `certbot`
  cada 12h (una vez descomentado en compose).

---

## 12. Troubleshooting

### 12.1 El bot no responde / timeout en el browser

```bash
docker compose ps
```

- Si `nginx` está caído → ver `docker compose logs nginx`. Común: certificado
  expirado o config inválida.
- Si `openwebui` está caído → ver logs. Suele arrancar solo con `docker compose restart openwebui`.
- Si `gateway` está caído → ver logs. Causas comunes:
  - `OPENAI_API_KEY` inválida o vencida.
  - Qdrant caído (gateway se salva pero degrada — chequear
    `curl http://localhost:6333/`).

### 12.2 Las respuestas vienen sin RAG ("no consta en los documentos")

```bash
docker exec mg-gateway curl -s http://localhost:8000/health
```

Si `qdrant_ok=false` o `qdrant_points=0`:
- Qdrant se rompió: `docker compose restart qdrant`, si no arranca, restore
  de snapshot.
- La colección se perdió: recrear con `docker exec mg-gateway python -c "from rag import qdrant_backend; qdrant_backend.ensure_collection()"`.

Si `qdrant_points` es normal pero igual no hay hits:
- Test directo del retrieval:
  ```bash
  docker exec mg-gateway curl -s -X POST http://localhost:8000/debug/query \
    -H "Content-Type: application/json" \
    -d '{"query": "clausula alcoholemia"}'
  ```
- Si `n_hits=0` → problema en el embedder o filtros muy agresivos.
- Si hay hits pero el LLM los ignora → subir `CONTEXT_RADIUS` a 2 o `QDRANT_TOP_K` a 15.

### 12.3 El cron de ingesta no procesa nada nuevo

```bash
tail -100 /data/logs/nas_ondemand.log
```

- **"rclone lsf dirs fallo"** → problema de conectividad al NAS. Verificar
  manualmente: `rclone lsd nas:`.
- **"TIMEOUT >1800s"** → un dir muy grande. Subir `RCLONE_CHUNK_TIMEOUT`
  en `.env` o excluir el dir específico agregando a `RCLONE_EXCLUDES` en
  `scripts/process_nas_ondemand.py`.
- **"embed fallo"** → el reranker no cargó. Ver logs del gateway al arranque
  para el mensaje "Cargando reranker …". Si falta, verificar que `hf_cache`
  esté montado y con permisos de escritura.

Preflight ad-hoc:

```bash
docker exec mg-gateway python /app/scripts/process_nas_ondemand.py --preflight
```

Dry-run (lista candidatos sin bajar):

```bash
docker exec mg-gateway python /app/scripts/process_nas_ondemand.py --dry-run --limit 100
```

### 12.4 Certificado TLS expirado

```bash
sudo docker run --rm \
  -v /opt/mg-bot/bot/certbot_certs:/etc/letsencrypt \
  certbot/certbot certificates
```

Si expiró:

```bash
docker compose stop nginx
docker run --rm -p 80:80 \
  -v /opt/mg-bot/bot/certbot_certs:/etc/letsencrypt \
  certbot/certbot renew
docker compose start nginx
```

### 12.5 Alguien no puede iniciar sesión

- Verificar que el rol no sea `pending`: Admin Panel > Users.
- Si el usuario admin perdió la contraseña: hoy no hay reset por email
  configurado. Solución rápida:
  ```bash
  docker exec -it mg-openwebui sh
  # Adentro del container:
  python -c "from open_webui.models.users import Users; from open_webui.utils.auth import get_password_hash; \
             Users.update_user_by_id('USER_ID', {'password': get_password_hash('nuevo_password')})"
  ```
  (El USER_ID sale del SQLite en `/app/backend/data/webui.db`.)

### 12.6 Se llenó el disco

```bash
df -h /
du -sh /var/lib/docker/*
du -sh /data/*
```

- **Purgar snapshots Qdrant:** `docker exec mg-qdrant find /qdrant/snapshots -type f -mtime +7 -delete`
- **Logs del cron:** `truncate -s 0 /data/logs/*.log`
- **Docker overhead:** `docker system prune -af --volumes` (⚠️ NO borra
  volúmenes con nombre en uso, pero verificar antes).
- **Temp files:** `docker exec mg-gateway sh -c "find /tmp -type f -mmin +60 -delete"`

### 12.7 Un archivo específico no aparece en las búsquedas

Verificar si está indexado:

```bash
docker exec mg-gateway python -c "
from rag import qdrant_backend
from qdrant_client.models import Filter, FieldCondition, MatchText
c = qdrant_backend.get_client()
r, _ = c.scroll(collection_name='mg_docs',
  scroll_filter=Filter(must=[FieldCondition(key='filename', match=MatchText(text='NOMBRE_ARCHIVO'))]),
  limit=10, with_payload=True)
for p in r: print(p.payload.get('path'), p.payload.get('page'))
"
```

Si no aparece:
- Verificar que exista en el NAS: `rclone lsl nas:Publico/RUTA/`.
- Correr la ingesta manualmente limitada:
  ```bash
  docker exec mg-gateway python /app/scripts/process_nas_ondemand.py --limit 5
  ```
- Si es un PDF sin capa de texto (escaneado), aparece con `status=sin_texto`
  y **no se indexa**. Habría que OCRear antes de subirlo al NAS.

### 12.8 El campo `carpeta_top` está mal (todos "volume1" o vacíos)

Ejecutar el backfill:

```bash
docker exec mg-gateway python /app/scripts/backfill_carpeta_top.py --dry-run
# Ver cuántos requieren fix

docker exec mg-gateway python /app/scripts/backfill_carpeta_top.py
# Aplicar
```

Idempotente, seguro correrlo varias veces.

---

## 13. Inventario de secretos

Qué secretos existen y dónde viven:

| Secreto                     | Ubicación en el VPS                         | Uso                             |
|-----------------------------|---------------------------------------------|---------------------------------|
| OpenAI API Key              | `/opt/mg-bot/bot/.env` (`OPENAI_API_KEY`)   | LLM del gateway                 |
| Clave SSH SFTP al NAS       | `/home/mg/.ssh/nas_sftp`                    | rclone → ingesta NAS            |
| rclone.conf                 | `/home/mg/.config/rclone/rclone.conf`       | Remote `nas:` configurado       |
| Certificados TLS            | `/opt/mg-bot/bot/certbot_certs/live/…`      | HTTPS del dominio               |
| SQLite Open WebUI           | volumen `openwebui_data`                    | Hashes de contraseñas de users  |

**Rotación recomendada:**

- OpenAI: cada 6 meses o si sospechás filtración. Nueva key en
  https://platform.openai.com/api-keys, actualizar `.env`, `docker compose restart gateway`.
- Llave SSH del NAS: cada 12 meses. Generar nueva, agregar la pública al
  NAS, actualizar `~/.ssh/nas_sftp`, sacar la vieja del NAS.
- Contraseñas de usuarios Open WebUI: los propios usuarios las cambian desde
  `Settings > Account`.

**Qué NO commitear al repo:**

- `.env`
- `certbot_certs/`
- Nada dentro de `data/` que tenga datos reales (los `.db` del repo actual
  son solo estructura vieja de referencia).

---

## Apéndice A — Comandos de uso frecuente

```bash
# Levantar todo
docker compose up -d

# Bajar todo (mantiene volúmenes)
docker compose down

# Bajar todo y BORRAR VOLÚMENES ⚠️
docker compose down -v

# Rebuild sólo del gateway (cambios de código)
docker compose build gateway && docker compose up -d gateway

# Entrar a un container
docker exec -it mg-gateway bash
docker exec -it mg-openwebui sh
docker exec -it mg-qdrant sh

# Ejecutar un script one-off
docker exec mg-gateway python /app/scripts/count_nas.py /data/nas_mirror
docker exec mg-gateway python /app/scripts/backfill_carpeta_top.py

# Verificar Qdrant
curl -s http://127.0.0.1:6333/collections/mg_docs | jq .result.points_count
curl -s http://127.0.0.1:6333/collections/mg_docs/exists | jq

# Snapshot manual
docker exec mg-qdrant sh -c "curl -s -X POST http://localhost:6333/collections/mg_docs/snapshots"
docker exec mg-qdrant ls -1 /qdrant/snapshots/

# Test de retrieval sin gastar OpenAI
docker exec mg-gateway curl -s -X POST http://localhost:8000/debug/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"clausula exclusion alcoholemia"}' | jq
```

## Apéndice B — Contactos y recursos externos

- **Dominio:** `bot-estudio.montoya-gherzi.com.ar`
- **VPS:** Hostinger KVM 4 (Ubuntu 24.04)
- **NAS:** Synology (IP interna, acceso vía SFTP)
- **Repo:** `https://github.com/franciscofrites200041-lgtm/MG`
- **OpenAI:** cuenta del estudio, dashboard en `https://platform.openai.com/`
- **Let's Encrypt:** email de registro para renovación (ver en
  `/opt/mg-bot/bot/certbot_certs/accounts/…/regr.json`).
