#!/usr/bin/env bash
set -euo pipefail

USER="nviduser"
HOSTS=(192.168.1.2 192.168.1.3 192.168.1.4)
REMOTE_PROJECT_DIR="/home/nviduser/YOLOv8-pt-fedn"
REMOTE_PYTHON="/home/nviduser/fedn_venv/bin/python"   # or /home/nviduser/fedn_venv/bin/python
SCRIPT="main_fedn_version.py"

# Pass args as separate vars to avoid $ARGS parsing issues
EPOCHS=1
BATCH_SIZE=4
LOCAL_UPDATES=10

for host in "${HOSTS[@]}"; do
  echo "===> ${host} :: starting client"
  ssh -o StrictHostKeyChecking=no "${USER}@${host}" \
    REMOTE_PROJECT_DIR="$REMOTE_PROJECT_DIR" \
    REMOTE_PYTHON="$REMOTE_PYTHON" \
    SCRIPT="$SCRIPT" \
    EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" LOCAL_UPDATES="$LOCAL_UPDATES" \
    'bash -s' <<'REMOTE'
set -euo pipefail
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
nohup_test='python.*main_fedn_version\.py'

# skip if already running
if pgrep -f "$nohup_test" >/dev/null 2>&1; then
  echo "Already running on $(hostname):"
  pgrep -fa "$nohup_test"
  exit 0
fi

mkdir -p logs
nohup "$REMOTE_PYTHON" "$SCRIPT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --local_updates "$LOCAL_UPDATES" \
  >> "logs/$(hostname)-fedn.log" 2>&1 &
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
