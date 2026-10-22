#!/bin/sh
# Copyright © 2026 440 Labs LLC
# Prompt-only ablation: attach po_fence.sh to every run container of this matrix.
#
# Runs ON each fleet host (orchestrator/launch.sh starts it with ssh + nohup), not on the
# Mac, so a sleeping controller never leaves a container unfenced. Every POLL
# seconds it lists running `ce-*<suffix>` containers and, for any whose fence
# loop is not alive (no /tmp/.po_fence pid, or that pid is gone), starts
#   docker exec -d -u root <ctr> sh /opt/po/po_fence.sh
# (/opt/po is bind-mounted read-only by matrix.po*.yaml). The entrypoint,
# clean-slate inspector and seed step all run before the first architect
# session, so a POLL-second attach delay never reaches an agent edit.
#
#   po_fence_watch.sh <run_suffix>        e.g. po_fence_watch.sh -po
set -u
SUFFIX=${1:?usage: po_fence_watch.sh <run_suffix>}
POLL=${PO_WATCH_POLL:-15}
PO_DIR=${PO_DIR:-/opt/ce/po}
LOG=$PO_DIR/po_fence_watch${SUFFIX}.log

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }

echo $$ > "$PO_DIR/watch${SUFFIX}.pid"
log "watcher started pid=$$ suffix=$SUFFIX poll=${POLL}s"
while :; do
  for c in $(docker ps --format '{{.Names}}' | grep -E "^ce-.*${SUFFIX}\$"); do
    if docker exec -u root "$c" sh -c 'p=$(cat /tmp/.po_fence 2>/dev/null) && kill -0 "$p" 2>/dev/null' ; then
      continue
    fi
    if docker exec -d -u root "$c" sh /opt/po/po_fence.sh; then
      log "attached fence to $c"
    else
      log "FAILED to attach fence to $c"
    fi
  done
  sleep "$POLL"
done
