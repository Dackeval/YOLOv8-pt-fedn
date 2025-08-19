import os, glob, subprocess, time, hashlib, sys

REMOTE="nviduser@vtt-scaleout-server"
REMOTE_DIR="/home/nviduser/data_partitions"
LOCAL_DIR = "/Users/sigvard/Desktop/VTT-test-send"
SSH_OPTS=["-o","StrictHostKeyChecking=accept-new"]  # first connect convenience

os.makedirs(LOCAL_DIR, exist_ok=True)
subprocess.run(
    ["rsync", "-avz", "--partial", "--progress", "-e", "ssh",
     f"{REMOTE}:{REMOTE_DIR}/", LOCAL_DIR],
    check=True
)

