#!/bin/bash
# Packs scores, the exam, final checkpoints and logs into maze_results.tgz.
set -euo pipefail
cd "$(dirname "$0")/.."
bash cluster/collect.sh "$PWD/maze_results.tgz"
