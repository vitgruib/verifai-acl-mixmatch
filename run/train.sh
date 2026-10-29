#!/bin/bash
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 bash run/train.sh [<first>-<last>] [ref] [bridge]
# Trains, JOBS_PER_GPU at a time (default 1) on each GPU listed:
#   <first>-<last>  the six methods (SFL, DR, PLR, Robust PLR, ACCEL, minimax) for those seeds of 0-9
#   ref             the 10 reference DR agents (100-104 are copied from GitHub when they are there)
#   bridge          the CartPole check, on the CPU next to the GPU jobs
# With no arguments: 0-9 ref bridge. Finished runs are skipped and unfinished ones restart, so
# rerunning resumes. Ctrl-C stops everything. One log per run in runs/slurm/.
set -euo pipefail
cd "$(dirname "$0")/.."; source cluster/env.sh; mkdir -p runs/slurm
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv >&2 || true
  echo "Pick the GPUs: CUDA_VISIBLE_DEVICES=0,1 bash $0 $*" >&2; exit 1
fi
[ $# -gt 0 ] || set -- 0-9 ref bridge

# one "algo seed index" line per run; index is the task number train.sbatch reads for that algo
tasks=$(mktemp); claims=$(mktemp -d); bridge=""
trap 'rm -rf "$tasks" "$claims"' EXIT
trap 'trap - TERM; kill 0' INT TERM   # background jobs ignore Ctrl-C in a script; stop them all
for a; do case $a in
  [0-9]-[0-9]) for algo in sfl dr plr rplr accel minimax; do
      for s in $(seq "${a%-*}" "${a#*-}"); do echo "$algo $s $s"; done; done >> "$tasks";;
  ref) maze_tasks ref | awk '{print $1, $2, NR - 1}' >> "$tasks";;
  bridge) bridge=1;;
  *) sed -n '2,8p' "$0" >&2; exit 2;;
esac; done

k=${JOBS_PER_GPU:-1}
[ "$k" -gt 1 ] && export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-$(awk "BEGIN {print 0.8 / $k}")}
n=$(wc -l < "$tasks")
say() { echo "$(date '+%a %d %b %H:%M') $*"; }
worker() {  # <gpu>: takes the next unclaimed run until none are left
  local i algo seed idx log
  for i in $(seq "$n"); do
    mkdir "$claims/$i" 2>/dev/null || continue
    read -r algo seed idx <<< "$(sed -n "${i}p" "$tasks")"; log=runs/slurm/train_${algo}_$seed.out
    say "GPU $1: $algo seed $seed started"
    if CUDA_VISIBLE_DEVICES=$1 ALGOS=$algo SLURM_ARRAY_TASK_ID=$idx nice -n 10 bash cluster/1_maze/train.sbatch > "$log" 2>&1
    then say "GPU $1: $algo seed $seed done"
    else say "GPU $1: $algo seed $seed FAILED, see $log"; touch "$claims/$i/failed"; fi
  done; }

say "$n runs on GPUs $CUDA_VISIBLE_DEVICES, $k at a time on each${bridge:+, and the CartPole check on the CPU}"
for g in ${CUDA_VISIBLE_DEVICES//,/ }; do for _ in $(seq "$k"); do worker "$g" & done; done
if [ -n "$bridge" ]; then
  ( for i in $(seq 0 59); do SLURM_ARRAY_TASK_ID=$i nice -n 10 bash cluster/1_maze/bridge.sbatch; done \
      && BRIDGE_GRADE=1 bash cluster/1_maze/bridge.sbatch ) > runs/slurm/bridge.out 2>&1 \
    && say "CartPole check done" || { say "CartPole check FAILED, see runs/slurm/bridge.out"; touch "$claims/bridge_failed"; } &
fi
wait
f=$(find "$claims" -name '*failed' | wc -l | tr -d ' ')
[ "$f" = 0 ] && say "all done. Next: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} bash run/finish.sh" \
  || { say "$f failed (logs above); rerun the same command to retry them"; exit 1; }
