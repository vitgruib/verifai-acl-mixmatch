#!/bin/bash
# ~30 min smoke test on the GPUs in CUDA_VISIBLE_DEVICES: tiny runs of every method through the exam
# and grading. Prints the memory per job, a suggested JOBS_PER_GPU and the full run's expected time.
set -euo pipefail
cd "$(dirname "$0")/.."
LOCAL=1 bash cluster/submit.sh maze-smoke
