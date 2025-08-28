#!/usr/bin/env bash
set -euo pipefail

USER="nviduser"
HOSTS=(192.168.1.2 192.168.1.3 192.168.1.4)

REMOTE_PROJECT_DIR="/home/nviduser/YOLOv8-pt-fedn"
REMOTE_VENV_ACTIVATE="$REMOTE_PROJECT_DIR/fedn_env/bin/activate"
SCRIPT="main_fedn_version.py"
ARGS="--epochs 1 --batch-size 4 --local_updates 10"

for host in "${HOSTS[@]}"; do
  echo "===> ${host} :: starting client"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${USER}@${host}" bash -lc "
    set -e
    cd '$REMOTE_PROJECT_DIR'
    mkdir -p logs
    source '$REMOTE_VENV_ACTIVATE'
    nohup python '$SCRIPT' $ARGS >> logs/\$(hostname)-fedn.log 2>&1 &
    echo \$! > logs/fedn.pid
    echo 'Started on ' \$(hostname) ' PID=' \$(cat logs/fedn.pid)
  "
done

echo "All start commands sent."