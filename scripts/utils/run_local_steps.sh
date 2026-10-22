# Steps for scripts/run_local.sh, with their checks. Sourced, not run; the
# main script stays a readable list of steps and this file holds the details.

die() { echo "run_local: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null || die "$1 not found on PATH. $2"; }
json_field() { python3 -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"; }
api_is_up() { curl -sf "$api/health" 2>/dev/null | grep -q '"status":"healthy"'; }
mongo_is_up() { nc -z 127.0.0.1 27017 2>/dev/null; }

usage() { sed -n '2,7p' "$root/scripts/run_local.sh"; }

# Sets docker, port, repo, config and api.
parse_args() {
  docker=0
  port=8080
  while [[ $# -gt 0 ]]; do
    case $1 in
      --docker) docker=1; shift ;;
      --port) port=$2; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) break ;;
    esac
  done
  [[ $# -eq 2 ]] || { usage >&2; exit 1; }
  repo=$(cd "$1" 2>/dev/null && pwd) || die "repo not found: $1"
  [[ -d $repo/.git ]] || die "$repo is not a git checkout"
  [[ -f $2 ]] || die "config not found: $2"
  config=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
  api=http://127.0.0.1:$port
}

check_prereqs() {
  [[ -f .env ]] || die "no .env; copy env.example to .env and fill in your keys"
  need python3 "It reads and writes the job JSON."
  if [[ $docker == 1 ]]; then
    need docker "Install Docker Desktop."
    return
  fi
  need python3.11 "brew install python@3.11"
  need uv "brew install uv (it builds your repo's venv)"
  need node "brew install node"
  need opencode "npm install -g opencode-ai"
  local version
  version=$(opencode --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
  [[ $(printf '%s\n1.17.11\n' "$version" | sort -V | head -1) == 1.17.11 ]] \
    || die "opencode $version is too old; need >= 1.17.11 (npm install -g opencode-ai@latest)"
  command -v rg >/dev/null || echo "run_local: rg not on PATH; OpenCode will download it on first use" >&2
}

create_venv() {
  [[ -x env/bin/python ]] && return
  python3.11 -m venv env
  env/bin/pip install -q -e ".[local]"
}

start_mongo() {
  mongo_is_up && return
  if command -v brew >/dev/null && brew list mongodb-community@7.0 >/dev/null 2>&1; then
    brew services start mongodb-community@7.0 >/dev/null
  else
    need mongod "Install MongoDB Community 7.0 (see README)."
    mkdir -p ~/.codeevolver/mongodb ~/.codeevolver/logs
    mongod --dbpath ~/.codeevolver/mongodb --bind_ip 127.0.0.1 --fork \
      --logpath ~/.codeevolver/logs/mongod.log >/dev/null
  fi
  for _ in $(seq 30); do mongo_is_up && return; sleep 1; done
  die "MongoDB did not start"
}

# Both start_api_* reuse an API that is already up, and refuse a port held by
# something else.
ensure_port_free() {
  ! curl -s -o /dev/null "$api/health" 2>/dev/null || die "port $port is used by another service; pass --port"
}

start_api_docker() {
  api_is_up && return
  ensure_port_free
  TASK_REPO=$repo CE_PORT=$port docker compose -f docker-compose.local.yml up --build -d
}

start_api_native() {
  api_is_up && return
  ensure_port_free
  set -a; . ./.env; set +a
  mkdir -p ~/.codeevolver/logs
  # The optimizer reports job status to CALLBACK_URL, which defaults to port 8080.
  APP_MODE=local SANDBOX_MODE=local CALLBACK_URL=$api \
    nohup env/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port "$port" \
    >~/.codeevolver/logs/api.log 2>&1 &
}

wait_for_health() {
  for _ in $(seq 60); do api_is_up && return; sleep 2; done
  if [[ $docker == 1 ]]; then
    die "API did not come up on $api; see: docker compose -f docker-compose.local.yml logs"
  fi
  die "API did not come up on $api; see ~/.codeevolver/logs/api.log"
}

# Prints the file:// repo_url the API registered for <path>.
connect_repo() {
  curl -sf -X POST "$api/local/connect-repo" -H 'Content-Type: application/json' \
    -d "{\"local_path\": \"$1\"}" | json_field repo_url \
    || die "POST /local/connect-repo failed"
}

# Posts <config.json> with repo_url filled in; prints the job id.
submit_job() {
  local body response
  body=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); c["repo_url"]=sys.argv[2]; print(json.dumps(c))' "$1" "$2")
  response=$(curl -s -X POST "$api/optimize" -H 'Content-Type: application/json' -d "$body")
  json_field job_id <<<"$response" 2>/dev/null || die "POST /optimize failed: $response"
}

# Polls every 30s; prints the terminal status.
wait_for_job() {
  echo "Submitted $1; polling every 30s (Ctrl-C stops polling, not the job)" >&2
  local status
  while true; do
    status=$(curl -sf "$api/job/$1" | json_field status 2>/dev/null || echo unknown)
    case $status in completed|failed|cancelled) echo "$status"; return ;; esac
    sleep 30
  done
}

print_results() {
  local job_id=$1 status=$2 results=$3 stop
  if [[ $docker == 1 ]]; then
    stop="TASK_REPO=$repo docker compose -f docker-compose.local.yml down"
  else
    stop="pkill -f 'uvicorn src.api.main:app'"
  fi
  echo
  echo "Job $job_id: $status"
  echo "  Result:  $results/history/$job_id.final.json"
  echo "  History: $results/history/$job_id.jsonl"
  echo "  Memory:  $results/memory/$job_id/memory.md"
  echo "  Log:     $results/logs/$job_id.log"
  echo "The API is still running (post-run agents may still be working). To stop it: $stop"
}
