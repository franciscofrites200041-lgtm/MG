#!/bin/bash
# Sync NAS -> VPS local mirror + procesar nuevos con bg_worker_qdrant.
#
# Instalacion:
#   sudo cp /opt/mg-bot/scripts/sync_nas.sh /usr/local/bin/mg-sync-nas
#   sudo chmod +x /usr/local/bin/mg-sync-nas
#   Cron user mg: 0 */6 * * * /usr/local/bin/mg-sync-nas

set -euo pipefail

REMOTE="${RCLONE_REMOTE:-nas}"
SRC="${RCLONE_SOURCE_PATH:-Publico}"                            # share directo (sin /volume1/)
VIRTUAL_ROOT="${VIRTUAL_ROOT:-/volume1/Publico/Estudio}"        # prefix en Qdrant (compat con puntos existentes)
DEST="${RCLONE_LOCAL_MIRROR:-/data/nas_mirror}"
LOG_DIR="${LOG_DIR:-/data/logs}"
LOCK="/tmp/mg-sync-nas.lock"
# ponytail: solo bajamos archivos nuevos/modificados en el ultimo dia.
# 1d = margen de 24h por si un sync se salta. Los historicos ya estan en Qdrant.
MAX_AGE="${MAX_AGE:-1d}"
# Borrar del mirror local los archivos con mtime > 7d (ya se procesaron).
LOCAL_CLEANUP_DAYS="${LOCAL_CLEANUP_DAYS:-7}"

mkdir -p "$DEST" "$LOG_DIR"

exec 200>"$LOCK"
if ! flock -n 200; then
  echo "[$(date -Is)] otro sync corriendo, salgo"
  exit 0
fi

echo "[$(date -Is)] === Copy NAS (max-age=$MAX_AGE) -> $DEST ==="

# rclone copy (NO sync): baja solo lo que falta, no borra remotos.
# --max-age limita a archivos modificados en la ventana. Los historicos NO se bajan.
rclone copy "${REMOTE}:${SRC}" "$DEST" \
  --max-age "$MAX_AGE" \
  --sftp-disable-hashcheck \
  --timeout 30m \
  --sftp-idle-timeout 5m \
  --transfers 4 \
  --checkers 8 \
  --stats 60s \
  --filter '- Publico.zip*' \
  --filter '- /#recycle/**' \
  --filter '- /#snapshot/**' \
  --filter '- **/#recycle/**' \
  --filter '- **/#snapshot/**' \
  --filter '- **/@eaDir/**' \
  --filter '- **/.DS_Store' \
  --filter '- **/Thumbs.db' \
  --log-file "$LOG_DIR/rclone-$(date +%Y%m%d).log" \
  --log-level INFO

echo "[$(date -Is)] === Procesar con bg_worker_qdrant (skipea lo ya indexado) ==="

# ponytail: corre en el container gateway (ya tiene torch, sentence-transformers, qdrant-client)
docker exec mg-gateway python /app/scripts/bg_worker_qdrant.py \
  --root "$DEST" \
  --virtual-root "$VIRTUAL_ROOT" \
  --workers 1 \
  2>&1 | tee -a "$LOG_DIR/bg_worker-$(date +%Y%m%d).log"

echo "[$(date -Is)] === Cleanup mirror local (mtime > ${LOCAL_CLEANUP_DAYS}d) ==="
# Libera espacio: los procesados hace mas de N dias se borran del mirror local.
# Siguen indexados en Qdrant. Si se modifican en el NAS, rclone los baja de nuevo.
find "$DEST" -type f -mtime "+${LOCAL_CLEANUP_DAYS}" -delete 2>/dev/null || true
find "$DEST" -type d -empty -delete 2>/dev/null || true

echo "[$(date -Is)] === Fin ==="
df -h "$DEST" | tail -1
