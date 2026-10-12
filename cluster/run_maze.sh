#!/bin/bash
# Maze run in one command, from the login node: setup (~15 min), then submit.sh maze
# (smoke test, then the full run if every smoke job succeeds).
set -euo pipefail
cd "$(dirname "$0")/.."
bash cluster/setup/clone.sh
bash cluster/setup/setup_jax.sh
bash cluster/setup/setup_sfl.sh
bash cluster/submit.sh maze
