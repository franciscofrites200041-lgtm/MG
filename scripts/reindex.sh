#!/bin/sh
# Cron nocturno del NAS. Correr con:
#   0 3 * * * /volume1/docker/mg-bot/scripts/reindex.sh >> /volume1/docker/mg-bot/logs/reindex.log 2>&1

set -eu

cd "$(dirname "$0")/.."

# Si el bot corre en Docker, ejecutamos el reindex dentro del container.
if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | grep -q '^mg-bot$'; then
    exec docker exec mg-bot python scripts/reindex_incremental.py
else
    exec python scripts/reindex_incremental.py
fi
