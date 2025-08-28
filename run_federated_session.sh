#!/usr/bin/env bash
set -euo pipefail

USER="nviduser"
HOSTS=(192.168.1.2 192.168.1.3 192.168.1.4)

REMOTE_PROJECT_DIR="/home/nviduser/YOLOv8-pt-fedn"
REMOTE_VENV_ACTIVATE="/home/nviduser/fedn_venv/bin/activate"
SCRIPT="main_fedn_version.py"
ARGS="--epochs 1 --batch-size 4 --local_updates 10"

for host in "${HOSTS[@]}"; do
  echo "===> ${host} :: starting client"
  ssh -o StrictHostKeyChecking=no "${USER}@${host}" \
    REMOTE_PROJECT_DIR="$REMOTE_PROJECT_DIR" \
    REMOTE_PYTHON="$REMOTE_PYTHON" \
    SCRIPT="$SCRIPT" \
    ARGS="$ARGS" \
    'bash -s' <<'REMOTE'
set -e
cd "$REMOTE_PROJECT_DIR"

# sanity checks
if [ ! -x "$REMOTE_PYTHON" ]; then
  echo "ERROR: python not found/executable at $REMOTE_PYTHON" >&2
  exit 2
fi
if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: script not found: $SCRIPT" >&2
  exit 2
fi

mkdir -p logs

# run in background and log
nohup "$REMOTE_PYTHON" "$SCRIPT" $ARGS >> "logs/$(hostname)-fedn.log" 2>&1 &
echo $! > logs/fedn.pid
sleep 1

if ps -p "$(cat logs/fedn.pid)" >/dev/null 2>&1; then
  echo "Started on $(hostname) PID=$(cat logs/fedn.pid)"
else
  echo "Process died immediately. Last 50 log lines:"
  tail -n 50 "logs/$(hostname)-fedn.log" || true
  exit 1
fi
REMOTE
done

echo "All start commands sent."