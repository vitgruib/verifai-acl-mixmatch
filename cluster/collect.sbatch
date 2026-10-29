#!/bin/bash
# Packs the results (cluster/collect.sh) once the Maze run ends, whether or not it succeeded.
#SBATCH --job-name=atlas-collect
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
bash cluster/collect.sh "${OUT:-$ATLAS/maze_results.tgz}"
