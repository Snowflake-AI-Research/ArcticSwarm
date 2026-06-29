#!/usr/bin/env bash
# One-time push of the prebuilt global fetch cache to a cluster pod's
# /data/soyoon/fetch_cache (the path ArcticSwarm uses by default in-cluster).
#
# Streams gzip -> pod -> gunzip in a single pipe, which is far more robust for a
# multi-GB SQLite file than `kubectl cp`, and self-resolves the (suffixed) pod
# name. SQLite content is text, so gzip typically shrinks the transfer ~3-4x.
#
# Usage:
#   snowflake/scripts/push_fetch_cache.sh [SRC_SQLITE] [JOB_NAME] [NAMESPACE] [DEST_PATH]
# Defaults:
#   SRC  = /home/soyoon/fetch_cache/fetch_cache.sqlite
#   JOB  = soyoung-dev-cpu     (matches `kontrol connect soyoung-dev-cpu`)
#   NS   = mltraining-dev
#   DEST = /data/soyoon/fetch_cache/fetch_cache.sqlite
set -euo pipefail

SRC="${1:-/home/soyoon/fetch_cache/fetch_cache.sqlite}"
JOB="${2:-soyoung-dev-cpu}"
NS="${3:-mltraining-dev}"
DEST="${4:-/data/soyoon/cache/fetch_cache.sqlite}"

[ -f "$SRC" ] || { echo "ERROR: source not found: $SRC" >&2; exit 1; }

POD="$(kubectl get pods -n "$NS" -o name 2>/dev/null | sed 's#^pod/##' | grep "^${JOB}" | head -1)"
[ -n "$POD" ] || { echo "ERROR: no running pod matching '${JOB}' in namespace '${NS}'." >&2; exit 1; }

echo "Source : $SRC ($(du -h "$SRC" | cut -f1))"
echo "Target : ${NS}/${POD}:${DEST}"
echo "Streaming gzip -> pod -> gunzip ..."
gzip -c "$SRC" | kubectl exec -i -n "$NS" "$POD" -- sh -c \
  "mkdir -p \$(dirname '$DEST') && gunzip -c > '$DEST'"

echo "Verifying on pod ..."
kubectl exec -n "$NS" "$POD" -- sh -c \
  "ls -la '$DEST' && python3 -c \"import sqlite3; print('entries on pod:', sqlite3.connect('$DEST').execute('SELECT count(*) FROM entries').fetchone()[0])\""
echo "Done. The pod will now serve cached fetches from $DEST (default fetch_cache_path)."
