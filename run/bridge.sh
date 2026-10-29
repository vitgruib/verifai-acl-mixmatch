#!/bin/bash
# CartPole check, CPU only: 3 arms x 20 seeds, then grades them and compares. Rerunning resumes.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p runs/slurm
exec > >(tee -a runs/slurm/run_bridge.log) 2>&1
for i in $(seq 0 59); do SLURM_ARRAY_TASK_ID=$i bash cluster/1_maze/bridge.sbatch; done
BRIDGE_GRADE=1 bash cluster/1_maze/bridge.sbatch
