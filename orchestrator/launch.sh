#!/usr/bin/env bash
# Copyright © 2026 440 Labs LLC
# Run one experiment matrix end to end: create the droplets, provision them,
# (optionally) rebuild the images, run the matrix to completion, and tear the
# droplets down once every run is collected.
#
#   CE_MATRIX=$PWD/orchestrator/matrix.main.yaml \
#     nohup ./orchestrator/launch.sh > ~/.codeevolver/logs/launch.main.log 2>&1 &
#   tail -f ~/.codeevolver/logs/launch.main.log
#
# Inputs (environment):
#   CE_MATRIX       required. The experiment matrix (matrix.main.yaml,
#                   matrix.evo.yaml, matrix.abl.yaml, matrix.po.yaml).
#   CE_STATE        controller state file (default orchestrator/state.<fleet tag>.json).
#   CE_IMAGE_LOCK   digest pins (default images.lock.json for image_tag `latest`,
#                   else images.<image_tag>.lock.json).
#   CE_BENCHMARKS   benchmarks to provision (default: every benchmark named in
#                   the matrix's host_benchmarks, else all five).
#   CE_REPO_ROOT    parent folder of the benchmark repos (default ~/Documents/code).
#   CE_BUILD=1      rebuild, push and re-pin every image on the build host between
#                   provisioning and dispatch (~45 min). CE_BUILD_HOST picks the
#                   host (default: the matrix's first host).
#   CE_QUARTER=1    quarter budgets (max_metric_calls / 4 per benchmark), for a
#                   load test.
#
# Optional `launch:` block in the matrix:
#   set: "<k=v;k=v>"   default CE_SET for every dispatch (e.g. the coder model
#                      of a coder ablation). An explicit CE_SET wins.
#   prompt_only: true  stage the prompt-only ablation before dispatch: seed
#                      bundles, kernel fence and per-benchmark instructions
#                      (orchestrator/prompt_only/).
#
# The controller machine must stay awake (caffeinate is used when present). If
# it sleeps the droplets keep running and `ce status` re-attaches on wake; only
# dispatch and collection pause. `fleet.py down` refuses while anything is
# uncollected.
set -euo pipefail
exec < /dev/null
cd "$(dirname "$0")/.."
source env/bin/activate

: "${CE_MATRIX:?set CE_MATRIX to the experiment matrix (e.g. orchestrator/matrix.main.yaml)}"
export CE_MATRIX
mval() { python3 -c "import os,yaml; m=yaml.safe_load(open(os.environ['CE_MATRIX'])); print($1)"; }

FLEET_TAG=$(mval "m['fleet']['tag']")
FLEET_FILE=$PWD/orchestrator/fleet.${FLEET_TAG}.json
IMAGE_TAG=$(mval "m['image_tag']")
RUN_SUFFIX=$(mval "m.get('run_suffix', '')")
FIRST_HOST=$(mval "m['hosts'][0]")
PROMPT_ONLY=$(mval "bool((m.get('launch') or {}).get('prompt_only'))")
MATRIX_SET=$(mval "(m.get('launch') or {}).get('set', '')")
MATRIX_BENCHES=$(mval "','.join(dict.fromkeys(b for bs in (m.get('host_benchmarks') or {}).values() for b in bs))")

export CE_STATE=${CE_STATE:-$PWD/orchestrator/state.${FLEET_TAG}.json}
if [[ "$IMAGE_TAG" == "latest" ]]; then
  export CE_IMAGE_LOCK=${CE_IMAGE_LOCK:-$PWD/orchestrator/images.lock.json}
else
  export CE_IMAGE_LOCK=${CE_IMAGE_LOCK:-$PWD/orchestrator/images.${IMAGE_TAG}.lock.json}
fi
export CE_REPO_ROOT=${CE_REPO_ROOT:-$HOME/Documents/code}
export CE_BENCHMARKS=${CE_BENCHMARKS:-${MATRIX_BENCHES:-hover,phantomwiki,ragqa,arcagi2,paperreview}}
if [[ -z "${CE_SET:-}" && -n "$MATRIX_SET" ]]; then
  export CE_SET="$MATRIX_SET"
fi

# One controller per state file. A state file that already records runs means a
# controller is (or was) driving this matrix; a second one would double-dispatch.
if [[ -f "$CE_STATE" ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$CE_STATE')).get('runs') else 1)"; then
  echo "refusing: $CE_STATE already records runs -- a controller already owns this matrix." >&2
  echo "Use \`ce status\` with CE_MATRIX/CE_STATE exported, or move the state file aside." >&2
  exit 1
fi
for pid in $(pgrep -f "controller.py start" || true); do
  if ps eww "$pid" 2>/dev/null | tr ' ' '\n' | grep -qx "CE_STATE=$CE_STATE"; then
    echo "refusing: controller pid $pid already runs with CE_STATE=$CE_STATE" >&2
    exit 1
  fi
done

if [[ "${CE_QUARTER:-}" == "1" ]]; then
  # max_metric_calls / 4 per benchmark (input.py budgets: hover 9000,
  # phantomwiki 5000, ragqa 25000, arcagi2 6000, paperreview 9000). arcagi2 is
  # set below 6000/4 because its iteration-1 fixed cost (seed 250 + variance
  # 250 + comparison 200) already buys only ~2 usable iterations at 1250.
  export CE_SET_HOVER=max_metric_calls=2250
  export CE_SET_PHANTOMWIKI=max_metric_calls=1250
  export CE_SET_RAGQA=max_metric_calls=6250
  export CE_SET_ARCAGI2=max_metric_calls=1250
  export CE_SET_PAPERREVIEW=max_metric_calls=2250
  echo "quarter budgets: CE_SET_* set for $CE_BENCHMARKS"
fi

if [[ "$PROMPT_ONLY" == "True" ]]; then
  # shellcheck source=prompt_only/launch_hooks.sh
  source orchestrator/prompt_only/launch_hooks.sh
  po_preflight
fi

step() { echo; echo "##### $(date '+%Y-%m-%d %H:%M:%S') $*"; }
host_benches() { python3 -c "import os,yaml; print(','.join(yaml.safe_load(open(os.environ['CE_MATRIX'])).get('host_benchmarks',{}).get('$1',[])))"; }

echo "matrix=$CE_MATRIX fleet=$FLEET_TAG image_tag=$IMAGE_TAG lock=$CE_IMAGE_LOCK"
echo "state=$CE_STATE benchmarks=$CE_BENCHMARKS run_suffix=$RUN_SUFFIX prompt_only=$PROMPT_ONLY"
echo "CE_SET=${CE_SET:-}"
env | grep '^CE_SET_' | cut -c1-160 || true

step "fleet up ($CE_MATRIX)"
python3 orchestrator/fleet.py up

step "provision (TAG=$IMAGE_TAG)"
# Hosts are read into an array first: provision.sh's ssh calls read stdin and
# would swallow a piped host list. (bash 3.2 on macOS has no mapfile.)
HOSTS=()
while IFS= read -r line; do HOSTS+=("$line"); done \
  < <(python3 -c "import json; [print(h, v['ip']) for h, v in json.load(open('$FLEET_FILE')).items()]")
for entry in "${HOSTS[@]}"; do
  read -r host ip <<< "$entry"
  # Hosts listed in host_benchmarks provision exactly those benchmarks (only
  # the ragqa host fetches the retriever and its index); others get the full set.
  hb=$(host_benches "$host")
  echo "--- $host ($ip)  benchmarks=${hb:-$CE_BENCHMARKS}"
  TAG="$IMAGE_TAG" CE_BENCHMARKS="${hb:-$CE_BENCHMARKS}" ./orchestrator/provision.sh "$host" "$ip" < /dev/null
done

if [[ "${CE_BUILD:-}" == "1" ]]; then
  BUILD_HOST=${CE_BUILD_HOST:-$FIRST_HOST}
  step "ce build --host $BUILD_HOST (rebuild @ $IMAGE_TAG, push, pin into $CE_IMAGE_LOCK, pull)"
  python3 orchestrator/controller.py build --host "$BUILD_HOST"
fi

step "lock check ($CE_IMAGE_LOCK)"
# Without a pin, dispatch falls back to the mutable tag and the run loses its
# provenance record.
export IMAGE_TAG_CHECK="$IMAGE_TAG"
python3 - <<PY
import json, os, sys
lock = json.load(open(os.environ["CE_IMAGE_LOCK"]))
want = os.environ["IMAGE_TAG_CHECK"]
benches = os.environ["CE_BENCHMARKS"].split(",")
bad = [b for b in benches if not lock.get(b, {}).get("image", "").endswith(f"{b}-{want}")]
if bad:
    sys.exit(f"lock missing or mismatched for {bad}: refusing to dispatch unpinned")
for b in benches:
    print(f"  {b}: {lock[b]['digest'].rsplit('@', 1)[1][:19]}...")
PY

if [[ "$PROMPT_ONLY" == "True" ]]; then
  po_stage_hosts
fi

step "ce start (blocks until the matrix is complete)"
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i python3 orchestrator/controller.py start
else
  python3 orchestrator/controller.py start
fi

step "fleet down (gated on every run being collected)"
python3 orchestrator/fleet.py down

step "done"
