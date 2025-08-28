#!/usr/bin/env bash
set -euo pipefail

# === EDIT THESE ===
USER="nviduser"                               # SSH user on the Orins
HOSTS=(orin-a.local orin-b.local orin-c.local)  # hostnames/IPs of the 3 Orins
DATA_PATHS=(/data/ds_a /data/ds_b /data/ds_c)   # dataset path on each Orin (one per host)

REMOTE_PROJECT_DIR="/home/nviduser/YOLOv8-pt-fedn"           # repo location on each Orin
REMOTE_VENV_ACTIVATE="$REMOTE_PROJECT_DIR/fedn_env/bin/activate"
SCRIPT="main_fedn_version.py"
ARGS="--epochs 1 --batch-size 4 --local_updates 10"     # tune per your needs
# ===================

for i in "${!HOSTS[@]}"; do
  host="${HOSTS[$i]}"
  data="${DATA_PATHS[$i]}"

  echo "===> ${host} :: starting client (DATA_PATH=${data})"
  # Pass variables as environment to the remote shell, then run a small script there.
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${USER}@${host}" \
  "PROJECT_URL='${PROJECT_URL}' FEDN_AUTH_TOKEN='${FEDN_AUTH_TOKEN}' DATA_PATH='${data}' \
   REMOTE_PROJECT_DIR='${REMOTE_PROJECT_DIR}' REMOTE_VENV_ACTIVATE='${REMOTE_VENV_ACTIVATE}' \
   SCRIPT='${SCRIPT}' ARGS='${ARGS}' bash -lc '
      set -e
      cd \"\$REMOTE_PROJECT_DIR\"
      mkdir -p logs
      # activate venv
      source \"\$REMOTE_VENV_ACTIVATE\"
      # export FEDn env
      export PROJECT_URL=\"\$PROJECT_URL\"
      export FEDN_AUTH_TOKEN=\"\$FEDN_AUTH_TOKEN\"
      export DATA_PATH=\"\$DATA_PATH\"
      # run in background and log
      nohup python \"\$SCRIPT\" \$ARGS >> logs/\$(hostname)-fedn.log 2>&1 &
      echo \$! > logs/fedn.pid
      echo \"Started FEDn client on \$(hostname) PID=\$(cat logs/fedn.pid) DATA_PATH=\$DATA_PATH\"
   '"
done

echo "All start commands sent."
