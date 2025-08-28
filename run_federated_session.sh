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

  # Build a single-line, well-quoted remote command
  remote_cmd=$'set -e\n'\
'cd '"$REMOTE_PROJECT_DIR"$'\n'\
'mkdir -p logs\n'\
'source '"$REMOTE_VENV_ACTIVATE"$'\n'\
'nohup python '"$SCRIPT"' '"$ARGS"$' >> logs/$(hostname)-fedn.log 2>&1 &\n'\
'echo $! > logs/fedn.pid\n'\
'sleep 1\n'\
'if ps -p $(cat logs/fedn.pid) >/dev/null 2>&1; then\n'\
'  echo "Started on $(hostname) PID=$(cat logs/fedn.pid)"\n'\
'else\n'\
'  echo "Process died immediately. Tail last 50 lines:"\n'\
'  tail -n 50 logs/$(hostname)-fedn.log || true\n'\
'  exit 1\n'\
'fi'

  # Run over SSH. Remove BatchMode=yes if you want password prompts.
  ssh -o StrictHostKeyChecking=no "${USER}@${host}" /bin/bash -lc "$remote_cmd"
done

echo "All start commands sent."