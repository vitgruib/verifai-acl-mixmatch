#!/bin/bash
# Builds the exam's VerifAI section from the 10 reference agents (after run/ref.sh).
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p runs/slurm
bash cluster/1_maze/exam.sbatch 2>&1 | tee -a runs/slurm/run_exam.log
