#!/bin/bash
# Sync NAS -> VPS local mirror + procesar nuevos con bg_worker_qdrant.
#
# Instalacion:
#   1. rclone config  # setear remote "nas" (SFTP a Synology)
#   2. cp scripts/sync_nas.sh /usr/local/bin/mg-sync-nas
#   3. chmod +x /usr/local/bin/mg-sync-nas
#   4. Cron:
#      0 */6 * * * /usr/local/bin/mg-sync-nas >> /var/log/mg-sync-nas.log 2>&1

set -euo pipefail

# Config via env (o .env del bot)
REMOTE="${RCLONE_REMOTE:-nas}"
SRC="${RCLONE_SOURCE_PATH:-/volume1/Publico/Estudio}"
DEST="${RCLONE_LOCAL_MIRROR:-/data/nas_mirror}"
BOT_DIR="${BOT_DIR:-/opt/mg-bot}"
LOG_DIR="${LOG_DIR:-/var/log/mg-bot}"
LOCK="/tmp/mg-sync-nas.lock"

mkdir -p "$DEST" "$LOG_DIR"

# Un solo sync a la vez (cron paralelo = desastre).
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "[$(date -Is)] otro sync corriendo, salgo"
  exit 0
fi

echo "[$(date -Is)] === Sync NAS -> $DEST ==="

rclone sync "${REMOTE}:${SRC}" "$DEST" \
  --sftp-disable-hashcheck \
  --timeout 10m \
  --sftp-idle-timeout 3m \
  --transfers 4 \
  --checkers 8 \
  --stats 60s \
  --filter '- /#recycle/**' \
  --filter '- /#snapshot/**' \
  --filter '- **/#recycle/**' \
  --filter '- **/#snapshot/**' \
  --filter '- **/@eaDir/**' \
  --filter '- **/.DS_Store' \
  --filter '- **/Thumbs.db' \
  --log-file "$LOG_DIR/rclone-$(date +%Y%m%d).log" \
  --log-level INFO

echo "[$(date -Is)] === Procesar nuevos con bg_worker_qdrant ==="

# Corre el worker dentro del container del bot (ya tiene deps + modelo cacheado).
docker exec mg-bot python /app/scripts/bg_worker_qdrant.py \
  --root "$DEST" \
  --virtual-root "$SRC" \
  --workers 1 \
  2>&1 | tee -a "$LOG_DIR/bg_worker-$(date +%Y%m%d).log"

echo "[$(date -Is)] === Fin ==="
