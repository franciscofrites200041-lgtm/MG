# Deploy MG Bot al VPS Hostinger KVM 4

Stack: **Qdrant + RAG Gateway (Gemini) + Open WebUI + Nginx**.

VPS: **Hostinger KVM 4** — 4 vCPU / 16 GB RAM / 200 GB NVMe / Ubuntu 24.04.

---

## Fase 0 — En tu PC (antes de tocar el VPS)

1. Terminar `bg_worker.py` (JSONs en `E:\MG_drop`).
2. Levantar Qdrant local en WSL, correr `scripts/drain_json_to_qdrant.py` para pasar los JSONs a Qdrant.
3. Crear snapshot en Qdrant local:

```bash
# En WSL Ubuntu, con Qdrant corriendo
curl -X POST "http://localhost:6333/collections/mg_docs/snapshots"
ls ~/qdrant_snapshots/mg_docs/  # deberia haber un .snapshot
```

4. Verificar count:

```bash
curl -s http://localhost:6333/collections/mg_docs | python3 -c "import sys,json;print(json.load(sys.stdin)['result']['points_count'])"
```

---

## Fase 1 — Setup del VPS

### 1.1. SSH + hardening

```bash
ssh root@TU_IP_VPS
apt update && apt upgrade -y
apt install -y ufw fail2ban curl git

adduser mg
usermod -aG sudo mg

ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw enable
[ -f /var/run/reboot-required ] && reboot
```

### 1.2. Docker

```bash
curl -fsSL https://get.docker.com | sh
usermod -aG docker mg
docker compose version
```

### 1.3. Dirs

```bash
sudo mkdir -p /opt/mg-bot
sudo chown -R mg:mg /opt/mg-bot
```

---

## Fase 2 — Clonar repo y configurar

```bash
su - mg
cd /opt/mg-bot
git clone https://github.com/franciscofrites200041-lgtm/MG.git .
cp .env.example .env
nano .env
```

Setear en `.env`:

- `GEMINI_API_KEY=<tu key AI Studio>`
- `GEMINI_MODEL=gemini-2.0-flash-exp`
- `COMPACT_THRESHOLD_TOKENS=8000`
- `RAG_BACKEND=qdrant`
- `QDRANT_URL=http://qdrant:6333`

`TELEGRAM_BOT_TOKEN` y `OPENAI_API_KEY` pueden quedar vacios (no se usan en este stack).

---

## Fase 3 — Qdrant + restore snapshot

```bash
docker compose up -d qdrant
docker compose logs -f qdrant  # esperar "Qdrant HTTP listening on 6333"
```

Subir snapshot desde tu PC (PowerShell):

```powershell
scp "\\wsl$\Ubuntu\home\franc\qdrant_snapshots\mg_docs\<TU_SNAP>.snapshot" mg@TU_IP_VPS:/tmp/
```

En el VPS:

```bash
sudo docker cp /tmp/*.snapshot mg-qdrant:/qdrant/snapshots/
SNAP=$(ls /tmp/*.snapshot | xargs -n1 basename | head -1)
curl -X PUT "http://127.0.0.1:6333/collections/mg_docs/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d "{\"location\": \"file:///qdrant/snapshots/$SNAP\"}"

curl -s http://127.0.0.1:6333/collections/mg_docs | python3 -c "import sys,json;print(json.load(sys.stdin)['result']['points_count'])"
```

---

## Fase 4 — Gateway + Open WebUI + Nginx

```bash
cd /opt/mg-bot
docker compose up -d gateway openwebui nginx
docker compose logs -f gateway    # "Application startup complete"
docker compose logs -f openwebui  # "Uvicorn running"
```

Test:

```bash
docker exec mg-gateway curl -s http://localhost:8000/health
# {"status":"ok","qdrant_ok":true,"qdrant_points":N,"gemini_configured":true,...}
```

**Desde tu maquina:** `http://TU_IP_VPS/`

1. Crear cuenta (email + password). **Primer usuario = admin**.
2. Nuevo chat → consulta legal → Gemini responde con contexto RAG y cita `(archivo.pdf, pag. X)`.
3. Sidebar: historial de chats persistido en volumen Docker.

---

## Fase 5 — TLS (cuando tengas dominio)

Apuntar A record `TU_DOMINIO.com` -> IP_VPS. Despues:

```bash
docker compose stop nginx
docker run --rm -it -p 80:80 \
  -v /opt/mg-bot/certbot_certs:/etc/letsencrypt \
  certbot/certbot certonly --standalone \
  -d TU_DOMINIO.com --email tu@email.com --agree-tos --no-eff-email
```

Editar `nginx/nginx.conf`: descomentar bloque HTTPS, reemplazar `TU_DOMINIO.com`.

```bash
docker compose up -d nginx
curl -I https://TU_DOMINIO.com/
```

Renovacion auto: descomentar servicio `certbot` en `docker-compose.yml` y `docker compose up -d certbot`.

---

## Operacion

### Logs
```bash
docker compose logs -f gateway openwebui qdrant nginx
```

### Actualizar codigo
```bash
cd /opt/mg-bot && git pull && docker compose build gateway && docker compose up -d gateway
```

### Backup snapshot Qdrant
```bash
docker exec mg-qdrant sh -c "curl -X POST http://localhost:6333/collections/mg_docs/snapshots"
sudo docker cp mg-qdrant:/qdrant/snapshots/ /opt/backups/qdrant-$(date +%Y%m%d)/
```

### Backup usuarios + chats
```bash
sudo docker cp mg-openwebui:/app/backend/data /opt/backups/openwebui-$(date +%Y%m%d)/
```

### Admin Open WebUI
- Aprobar signups, cambiar roles: menu Users
- Cerrar signups: `ENABLE_SIGNUP=false` en `.env` + restart openwebui
- Modelo: `mg-bot-gemini` (nuestro gateway)

---

## Checklist final

- [ ] `curl -s http://127.0.0.1:6333/collections/mg_docs` responde con `points_count` esperado
- [ ] `docker exec mg-gateway curl -s http://localhost:8000/health` responde `qdrant_ok:true, gemini_configured:true`
- [ ] `http://TU_IP_VPS/` carga login de Open WebUI
- [ ] Signup: cuenta admin creada
- [ ] Consulta legal responde con citas `(archivo.pdf, pag. X)`
- [ ] Al superar `COMPACT_THRESHOLD_TOKENS` gateway loguea `Compactando N mensajes...`
- [ ] Fail2ban activo (`sudo fail2ban-client status`)
- [ ] (Con dominio) TLS grado "A" en https://www.ssllabs.com/ssltest/
