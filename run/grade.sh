#!/bin/bash
# Grades the 60 trained runs on the exam (after run/train.sh and run/exam.sh).
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p runs/slurm
bash cluster/1_maze/grade.sbatch 2>&1 | tee -a runs/slurm/run_grade.log
