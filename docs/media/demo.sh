#!/usr/bin/env bash
# Drives a real DevWaste session for recording. Typing is simulated; every command really runs.
#
# Usage: docs/media/demo.sh [VENV_DIR]
#   VENV_DIR  optional virtual environment to activate. If omitted, the currently
#             active environment is used (`devwaste` must be on PATH).
# Needs Docker, the FlakeRadar stack, and flakeradar_baseline.csv (from `devwaste collect`)
# next to this script.
set -u
if [ $# -ge 1 ]; then
  # shellcheck disable=SC1091
  source "$1/bin/activate" || { echo "cannot activate venv: $1" >&2; exit 1; }
fi
command -v devwaste >/dev/null || { echo "devwaste not found; activate a venv or pass its path" >&2; exit 1; }
cd "$(dirname "$0")"
PS='\e[1;32mahmed@devbox\e[0m:\e[1;34m~/DevWaste\e[0m$ '
type_cmd() {
  printf "$PS"
  local s="$1"; for ((i=0;i<${#s};i++)); do printf '%s' "${s:i:1}"; sleep 0.035; done
  sleep 0.4; printf '\n'
}
run() { type_cmd "$1"; eval "$1"; sleep 1.5; }
clear
printf '\e[2m# FlakeRadar stack running locally (Postgres, Redpanda, Redis, MLflow, Dagster)\e[0m\n'
run "devwaste scan"
run "devwaste train --data flakeradar_baseline.csv --output-dir model --epochs 30"
clear
printf '\e[2m# live monitoring; a real redis-benchmark load spike starts mid-run\e[0m\n'
( sleep "${LOAD_AT:-28}"; docker exec flakeradar-redis-1 timeout 18 redis-benchmark -q -n 100000000 -c 50 -P 16 -t set,get >/dev/null 2>&1 ) &
run "devwaste monitor --model model --interval 2 --duration ${MON_DUR:-52} --consecutive 2"
wait
