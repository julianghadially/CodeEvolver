#!/usr/bin/env bash
# Run one CodeEvolver job on your repo, natively or in Docker.
#
#   scripts/run_local.sh [--docker] [--port N] <repo-path> <config.json>
#
# <config.json> is the /optimize body from the README; repo_url is filled in.
# Each step's checks live in scripts/utils/run_local_steps.sh.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
source "$root/scripts/utils/run_local_steps.sh"

parse_args "$@"
cd "$root"

# 1. Check prerequisites and start the API (reused if one is already up).
check_prereqs
if [[ $docker == 1 ]]; then
  start_api_docker              # API, MongoDB and agents in one container
  workspace=/workspace          # where the container mounts your repo
  results=$HOME/.codeevolver-mount/.codeevolver
else
  create_venv                   # env/, if missing
  start_mongo
  start_api_native              # uvicorn in the background
  workspace=$repo
  results=$HOME/.codeevolver
fi
wait_for_health

# 2. Register the repo, submit the job and wait for it to finish.
repo_url=$(connect_repo "$workspace")
job_id=$(submit_job "$config" "$repo_url")
status=$(wait_for_job "$job_id")

# 3. Show where the results landed.
print_results "$job_id" "$status" "$results"
[[ $status == completed ]]
