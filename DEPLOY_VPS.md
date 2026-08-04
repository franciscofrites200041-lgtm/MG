# Deploy MG Bot al VPS Hostinger KVM 4

Stack: **Qdrant + Bot Telegram + API FastAPI + Frontend estatico + Nginx**.

VPS recomendado: **Hostinger KVM 4** — 4 vCPU / 16 GB RAM / 200 GB NVMe / Ubuntu 24.04.

Costo total estimado: ~14 USD/mes VPS + dominio + backup opcional.

---

## Fase 0 — Antes de tocar el VPS

En tu PC, dejar terminado el procesamiento full del corpus (bg_worker + drain_pending). Verificar:

```bash
python -c "from rag.search import stats_index; print(stats_index())"
# Deberia mostrar files_ok ~= 381000, embeddings ~= N chunks total
```

**No sigas si `embeddings` es 0.** El export a Qdrant depende de eso.

---

## Fase 1 — Setup del VPS

### 1.1. Contratar y primer SSH

```bash
ssh root@TU_IP_VPS
```

### 1.2. Hardening minimo

```bash
apt update && apt upgrade -y
apt install -y ufw fail2ban curl git

# Crear usuario no-root
adduser mg
usermod -aG sudo mg

# Firewall
ufw allow OpenSSH
ufw allow 80
ufw allow 443
ufw enable

# Reboot si hubo kernel updates
[ -f /var/run/reboot-required ] && reboot
```

### 1.3. Docker + docker-compose

```bash
curl -fsSL https://get.docker.com | sh
usermod -aG docker mg

# Compose plugin ya viene incluido en docker-ce moderno.
docker compose version  # verificar
```

### 1.4. rclone

```bash
curl https://rclone.org/install.sh | sudo bash

# Config del remote NAS. Interactivo, elegir sftp.
# Al terminar, editar ~/.config/rclone/rclone.conf a mano si querés.
rclone config
# > n (new remote)
# > name: nas
# > type: sftp
# > host: IP publica o Tailscale del NAS
# > user: usuario Synology
# > pass: password
# > shell: no (skip)

# Test
rclone lsd nas:/volume1/Publico/Estudio | head
```

### 1.5. Directorios

```bash
sudo mkdir -p /data/nas_mirror /data/logs /opt/mg-bot
sudo chown -R mg:mg /data /opt/mg-bot
```

---

## Fase 2 — Clonar el repo y configurar

```bash
su - mg
cd /opt/mg-bot
git clone <REPO_URL> .

cp .env.example .env
nano .env
```

Editar en `.env`:

- `TELEGRAM_BOT_TOKEN=<tu token>`
- `OPENAI_API_KEY=<tu key>`
- `RAG_BACKEND=qdrant`
- `QDRANT_URL=http://qdrant:6333`
- `RCLONE_REMOTE=nas`
- `RCLONE_SOURCE_PATH=/volume1/Publico/Estudio`
- `RCLONE_LOCAL_MIRROR=/data/nas_mirror`
- `API_CORS_ORIGINS=https://TU_DOMINIO.com`

---

## Fase 3 — Levantar solo Qdrant y restaurar snapshot

### 3.1. Arrancar Qdrant

```bash
docker compose up -d qdrant
docker compose logs -f qdrant  # esperar "Qdrant HTTP listening on 6333"
```

### 3.2. Exportar snapshot desde tu PC

En tu PC (donde tenés el `index.db` full):

```bash
# 1. Levantar Qdrant local
docker run -d --name qdrant_local -p 6333:6333 \
  -v qdrant_pc_data:/qdrant/storage qdrant/qdrant:v1.11.3

# 2. Exportar SQLite -> Qdrant local
cd C:\Users\franc\OneDrive\Escritorio\WEB\MG\bot
python scripts/export_to_qdrant.py \
  --db ./data/index.db \
  --qdrant-url http://localhost:6333 \
  --collection mg_docs \
  --batch 512 \
  --snapshot-dir ./snapshots

# Al terminar hay un .tar en ./snapshots/
```

### 3.3. Subir snapshot al VPS

```bash
scp ./snapshots/*.snapshot mg@TU_IP_VPS:/tmp/
```

### 3.4. Restore en el VPS

```bash
# En el VPS
ssh mg@TU_IP_VPS

# Copiar el snapshot al volume de qdrant
docker cp /tmp/*.snapshot mg-qdrant:/qdrant/snapshots/

# Recover
SNAP=$(ls /tmp/*.snapshot | xargs -n1 basename | head -1)
curl -X PUT "http://127.0.0.1:6333/collections/mg_docs/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d "{\"location\": \"file:///qdrant/snapshots/$SNAP\"}"

# Verificar
curl http://127.0.0.1:6333/collections/mg_docs | jq .result.points_count
# Deberia mostrar los ~4M puntos
```

---

## Fase 4 — Levantar bot + api + nginx

```bash
cd /opt/mg-bot
docker compose up -d

# Verificar
docker compose logs -f bot          # deberia loguear "Bot arrancando (long polling)"
docker compose logs -f api          # uvicorn escuchando 8000
curl http://127.0.0.1/healthz       # ok
curl http://127.0.0.1/api/health    # {"status":"ok","backend":"qdrant",...}
```

Probar el bot desde Telegram: mandale una consulta, deberia buscar en la base y citar.

Probar el frontend: `http://TU_IP_VPS/` en el navegador.

---

## Fase 5 — TLS con Let's Encrypt

### 5.1. Apuntar el dominio

En tu registrador (Namecheap, Cloudflare, etc), crear A record:

```
TU_DOMINIO.com  A  <IP_DEL_VPS>
```

Esperar propagacion (~5-30 min). Verificar con `dig TU_DOMINIO.com`.

### 5.2. Certbot inicial (standalone)

```bash
docker compose stop nginx

docker run --rm -it \
  -p 80:80 \
  -v /opt/mg-bot/certbot_certs:/etc/letsencrypt \
  certbot/certbot certonly --standalone \
  -d TU_DOMINIO.com \
  --email tu@email.com \
  --agree-tos --no-eff-email
```

### 5.3. Descomentar bloque TLS en `nginx/nginx.conf`

Reemplazar `TU_DOMINIO.com` en el archivo. Despues:

```bash
docker compose up -d nginx
curl https://TU_DOMINIO.com/api/health  # deberia devolver JSON via HTTPS
```

### 5.4. Renovacion automatica

Descomentar el servicio `certbot` en `docker-compose.yml` y:

```bash
docker compose up -d certbot
```

---

## Fase 6 — Sync automatico del NAS

### 6.1. Instalar el script

```bash
sudo cp /opt/mg-bot/scripts/sync_nas.sh /usr/local/bin/mg-sync-nas
sudo chmod +x /usr/local/bin/mg-sync-nas

# Test manual (primer sync completo, puede tardar)
sudo -u mg /usr/local/bin/mg-sync-nas
```

### 6.2. Cron cada 6h

```bash
sudo crontab -u mg -e
```

Agregar:

```cron
0 */6 * * * /usr/local/bin/mg-sync-nas >> /var/log/mg-bot/sync.log 2>&1
```

---

## Fase 7 — Backups

### 7.1. Snapshot Qdrant diario a Backblaze B2

```bash
# Setup B2 remote en rclone
rclone config  # elegir type: b2
# > name: b2
# > account: <tu KEY_ID>
# > key: <tu APP_KEY>

# Script de backup
cat > /usr/local/bin/mg-backup <<'EOF'
#!/bin/bash
set -e
DATE=$(date +%Y%m%d)
docker exec mg-qdrant sh -c "curl -X POST http://localhost:6333/collections/mg_docs/snapshots"
sleep 5
SNAP=$(docker exec mg-qdrant ls /qdrant/snapshots | grep mg_docs | tail -1)
docker cp mg-qdrant:/qdrant/snapshots/$SNAP /tmp/$SNAP
rclone copy /tmp/$SNAP b2:mg-bot-backups/qdrant/ --progress
rm /tmp/$SNAP
# Retener solo 30 dias en B2
rclone delete b2:mg-bot-backups/qdrant/ --min-age 30d
EOF
sudo chmod +x /usr/local/bin/mg-backup

# Cron diario 4 AM
sudo crontab -u mg -e
```

Agregar:

```cron
0 4 * * * /usr/local/bin/mg-backup >> /var/log/mg-bot/backup.log 2>&1
```

---

## Operacion habitual

### Ver logs
```bash
docker compose logs -f bot api nginx qdrant
tail -f /var/log/mg-bot/sync.log
```

### Reindexar todo desde cero
```bash
docker exec mg-bot python /app/scripts/bg_worker_qdrant.py --root /data/nas_mirror
```

### Actualizar codigo
```bash
cd /opt/mg-bot
git pull
docker compose build bot api
docker compose up -d bot api
```

### Restore desde backup B2
```bash
rclone copy b2:mg-bot-backups/qdrant/<snap>.snapshot /tmp/
docker cp /tmp/<snap>.snapshot mg-qdrant:/qdrant/snapshots/
curl -X PUT "http://127.0.0.1:6333/collections/mg_docs/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d "{\"location\": \"file:///qdrant/snapshots/<snap>.snapshot\"}"
```

---

## Checklist final antes de dar por listo

- [ ] `curl https://TU_DOMINIO.com/api/health` responde `backend: qdrant`, `points > 3000000`
- [ ] Bot Telegram responde a `/start` y a consultas reales
- [ ] Frontend en `https://TU_DOMINIO.com/` busca y muestra resultados
- [ ] `mg-sync-nas` corrio al menos una vez y proceso archivos (revisar logs)
- [ ] `mg-backup` corrio y hay un snapshot en B2
- [ ] TLS con nota "A" en https://www.ssllabs.com/ssltest/
- [ ] Fail2ban activo (`sudo fail2ban-client status`)
