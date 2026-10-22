# Copyright © 2026 440 Labs LLC
# Prompt-only ablation hooks, sourced by orchestrator/launch.sh when the matrix
# sets `launch: {prompt_only: true}` (matrix.po.yaml).
#
# What the prompt-only arm adds on top of a normal launch:
#   * seed: the `prompt-only` branch bundles (runner/bundles-po/, built by
#     prompt_only/build_bundles.py), bind-mounted over /opt/seed/<bench>.bundle
#     by the matrix's docker_args;
#   * instructions: CE_SET_HOVER / CE_SET_PHANTOMWIKI = additional_instructions
#     from prompt_only/{hover,phantomwiki}.md, which the LM constraint gate
#     enforces;
#   * kernel fence: po_fence_watch.sh runs on each host and injects po_fence.sh
#     into every `ce-*<run_suffix>` container, leaving only the signatures file
#     and codeevolver.md writable to the ce-agent uid.
# Every run is audited offline afterwards by scripts/prompt_only_audit.py.
# Expects launch.sh's variables: CE_MATRIX, CE_IMAGE_LOCK, CE_BENCHMARKS,
# FLEET_FILE, RUN_SUFFIX, HOSTS, and its host_benches/step helpers.

po_preflight() {
  if [[ "${CE_SET:-}" == *coding_harness_model=* || "${CE_SET:-}" == *reflection_lm=* ]]; then
    echo "refusing: the prompt-only arm runs the input.py models; do not override them in CE_SET" >&2
    exit 1
  fi
  if [[ -n "${CE_SET_HOVER:-}${CE_SET_PHANTOMWIKI:-}" ]]; then
    echo "refusing: CE_SET_HOVER / CE_SET_PHANTOMWIKI are set from orchestrator/prompt_only/*.md" >&2
    exit 1
  fi
  local f
  for f in runner/bundles-po/hover.bundle runner/bundles-po/phantomwiki.bundle \
           orchestrator/prompt_only/po_fence.sh orchestrator/prompt_only/po_fence_watch.sh; do
    [[ -f "$f" ]] || { echo "refusing: $f missing (run orchestrator/prompt_only/build_bundles.py)" >&2; exit 1; }
  done
  CE_SET_HOVER=$(po_instructions orchestrator/prompt_only/hover.md)
  CE_SET_PHANTOMWIKI=$(po_instructions orchestrator/prompt_only/phantomwiki.md)
  export CE_SET_HOVER CE_SET_PHANTOMWIKI
  shasum -a 256 runner/bundles-po/*.bundle
}

# additional_instructions -> one-line JSON string (experiments/run.py parses
# JSON first). The runner splits CE_SET on ';', so the text must not contain one.
po_instructions() {
  python3 - "$1" <<'PY'
import json, sys
text = open(sys.argv[1]).read().strip()
if ";" in text:
    sys.exit(f"{sys.argv[1]}: contains ';' (the CE_SET separator)")
print("additional_instructions=" + json.dumps(text))
PY
}

po_stage_hosts() {
  step "pull the locked digests on every host"
  local entry host ip hb b digest addr
  for entry in "${HOSTS[@]}"; do
    read -r host ip <<< "$entry"
    hb=$(host_benches "$host")
    for b in $(echo "${hb:-$CE_BENCHMARKS}" | tr ',' ' '); do
      digest=$(python3 -c "import json,os; print(json.load(open(os.environ['CE_IMAGE_LOCK']))['$b']['digest'])")
      echo "--- $host: $digest"
      docker --context "$host" pull "$digest"
    done
  done

  step "stage prompt-only seeds + fence, start the fence watcher on every host"
  # Both bundles go to every host: the matrix mounts both, and a missing bind
  # source would make docker create a directory there instead.
  for entry in "${HOSTS[@]}"; do
    read -r host ip <<< "$entry"
    addr="root@$ip"
    echo "--- $host ($ip)"
    ssh "$addr" "mkdir -p /opt/ce/po/seed"
    scp runner/bundles-po/hover.bundle runner/bundles-po/phantomwiki.bundle "$addr:/opt/ce/po/seed/"
    scp orchestrator/prompt_only/po_fence.sh orchestrator/prompt_only/po_fence_watch.sh "$addr:/opt/ce/po/"
    ssh "$addr" "chmod 644 /opt/ce/po/seed/*.bundle && chmod 755 /opt/ce/po/*.sh && sha256sum /opt/ce/po/seed/*.bundle"
    # One watcher per suffix per host, tracked by pidfile (not pgrep/pkill -f,
    # which would match this very ssh command line); a relaunch replaces it.
    ssh "$addr" "pf=/opt/ce/po/watch$RUN_SUFFIX.pid
      [ -f \$pf ] && kill \$(cat \$pf) 2>/dev/null
      nohup /opt/ce/po/po_fence_watch.sh $RUN_SUFFIX > /dev/null 2>&1 < /dev/null &
      sleep 1
      kill -0 \$(cat \$pf) 2>/dev/null && echo 'watcher up' || { echo 'watcher FAILED'; exit 1; }"
  done
}
