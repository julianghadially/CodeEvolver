#!/bin/sh
# Copyright © 2026 440 Labs LLC
# Prompt-only ablation: kernel write fence for ONE run container.
#
# Runs as root INSIDE the container (injected by po_fence_watch.sh with
# `docker exec -d -u root`), bind-mounted read-only at /opt/po/po_fence.sh by
# matrix.po*.yaml. Every agent process runs as the unprivileged `ce-agent`
# uid; this loop makes /workspace read-only to that uid except for:
#   * the benchmark's signatures file (+ its directory, so `sed -i`,
#     `git checkout -- <f>` and `git reset --hard` can replace it),
#   * codeevolver.md,
#   * .git/ (the agent commits and resets itself) -- minus hooks/, info/ and
#     config, so it cannot plant a hook, an exclude or core.fsmonitor that the
#     root-run engine git would honour.
# Memory (/artifacts/.codeevolver/memory) is outside /workspace: untouched.
#
# Root has CAP_FOWNER (chmod anything) but not CAP_CHOWN, so the fence is mode
# bits only: group/other write is stripped, the root owner keeps write, so the
# engine's own (root) git and evaluator are unaffected. Executable bits are
# never touched, so git sees no mode change.
#
# It re-applies every TICK seconds because the entrypoint first opens the tree
# (`chmod -R a+rwX`) and the whole container runs with umask 0: any file a root
# git checkout rewrites comes back 0666. Each tick only chmods files that
# drifted (`-perm /022`). Hard-coded paths: this is a research fence.
set -u
W=/workspace
TICK=${PO_FENCE_TICK:-10}
LOG=/artifacts/po_fence.log
MARK=/tmp/.po_fence

case "${CE_BENCH:-}" in
  hover)       SIG=langProBe/hover/prompts/hover_signatures.py ;;
  phantomwiki) SIG=src/program/prompts/phantomwiki_signatures.py ;;
  *) echo "po_fence: refusing, unknown CE_BENCH='${CE_BENCH:-}'" >&2; exit 1 ;;
esac
SIGDIR=$(dirname "$SIG")

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" >> "$LOG"; }

fence() {
  [ -d "$W/.git" ] || return 0   # before clone_workspace: nothing to fence yet
  n=$(find "$W" \( -path "$W/.git" -o -path "$W/$SIGDIR" -o -path "$W/codeevolver.md" \) -prune \
        -o -perm /022 -print -exec chmod go-w {} + | wc -l)
  if [ -d "$W/$SIGDIR" ]; then
    chmod 0777 "$W/$SIGDIR"
    [ -f "$W/$SIG" ] && chmod 0666 "$W/$SIG"
  fi
  [ -f "$W/codeevolver.md" ] && chmod 0666 "$W/codeevolver.md"
  for d in "$W/.git/hooks" "$W/.git/info"; do
    [ -d "$d" ] || continue
    chmod 0755 "$d"
    find "$d" -type f -perm /022 -exec chmod go-w {} +
  done
  [ -f "$W/.git/config" ] && chmod go-w "$W/.git/config"
  echo "$n"
}

if [ "${1:-}" = "--once" ]; then
  fence >/dev/null
  exit 0
fi

echo $$ > "$MARK"
log "po_fence started pid=$$ bench=$CE_BENCH sig=$SIG tick=${TICK}s"
first=1
while :; do
  n=$(fence)
  if [ "$first" = 1 ] && [ -d "$W/.git" ]; then
    log "first fence applied ($n paths tightened)"
    first=0
  elif [ "${n:-0}" -gt 0 ]; then
    log "re-fenced $n drifted paths"
  fi
  sleep "$TICK"
done
