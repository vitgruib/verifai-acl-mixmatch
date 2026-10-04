#!/bin/bash
# One command per part. Prints every sbatch line it runs; DRY_RUN=1 prints without submitting.
#   bash cluster/submit.sh smoke      # ~10 min sanity check of Parts 1 and 4 (tiny budgets; JaxNav SFL + minimax)
#   bash cluster/submit.sh 1          # Maze: train (60 tasks) -> falsify (60 tasks, after train)
#   bash cluster/submit.sh 2          # CarRacing train (60 tasks, Xvfb)
#   bash cluster/submit.sh 3          # BipedalWalker train (60 tasks, requeues)
#   bash cluster/submit.sh 4          # JaxNav train (60 tasks)
#   bash cluster/submit.sh all        # 1-4
# Extra sbatch flags (partition, account, ...) go in SBATCH_ARGS, e.g. SBATCH_ARGS="-p gpu -A lab".
set -euo pipefail
cd "$(dirname "$0")/.."; source cluster/env.sh; mkdir -p "$RUNS/slurm" runs/slurm
NSEEDS=${NSEEDS:-10}
sb() {  # sbatch wrapper -> prints job id
  echo "sbatch ${SBATCH_ARGS:-} $*" >&2
  if [ -n "${DRY_RUN:-}" ]; then echo DRYRUN; else sbatch --parsable ${SBATCH_ARGS:-} "$@"; fi
}
count() { set -- $1; echo $(( $# * NSEEDS - 1 )); }

part1() {
  local A=${ALGOS:-"dr plr rplr accel minimax sfl"}
  local j; j=$(ALGOS="$A" sb --export=ALL --array=0-$(count "$A") cluster/1_maze/train.sbatch)
  ALGOS="$A" sb --export=ALL --array=0-$(count "$A") --dependency=afterok:$j cluster/1_maze/falsify.sbatch
}
dcd() {  # dcd() <part dir> <domain> <xvfb> <configs...>
  local p=$1 dom=$2 x=$3; shift 3
  local f=$RUNS/${dom}_cmds.txt
  bash cluster/dcd_cmds.sh $dom $x "$@" > "$f"
  echo "$(wc -l < "$f") commands -> $f" >&2
  sb --array=0-$(( $(wc -l < "$f") - 1 )) cluster/$p/train.sbatch "$f"
}
part2() { dcd 2_carracing car_racing 1 ${CONFIGS:-cr_dr cr_minimax cr_plr cr_robust_plr cr_accel cr_sfl}; }
part3() { dcd 3_bipedal bipedal 0 ${CONFIGS:-bipedal_dr bipedal_minimax bipedal_plr bipedal_robust_plr bipedal_accel bipedal_sfl}; }
part4() { local A=${ALGOS:-"dr minimax plr rplr accel sfl"}; ALGOS="$A" sb --export=ALL --array=0-$(count "$A") cluster/4_jaxnav/train.sbatch; }

smoke() {  # one seed, one algo per codebase, tiny budgets; outputs under $RUNS/smoke
  local S="learning.NUM_ENVS=16 learning.NUM_ENVS_FROM_SAMPLED=8 learning.NUM_ENVS_TO_GENERATE=8 learning.NUM_STEPS=32 learning.TOTAL_TIMESTEPS=4096 learning.EVAL_FREQ=2 learning.NUM_CHECKPOINTS=2 BATCH_SIZE=64 NUM_BATCHES=1 ROLLOUT_STEPS=50 NUM_TO_SAVE=32"
  export RUNS=$RUNS/smoke NSEEDS=1 NUM_UPDATES=50 JAXUED_ARGS="--eval_freq 10 --checkpoint_save_interval 1" BUDGET=50 FSEEDS=0 SAMPLERS=random
  local j1 j2
  j1=$(ALGOS=rplr sb --export=ALL --time=00:30:00 --array=0 cluster/1_maze/train.sbatch)
  j2=$(ALGOS=sfl SFL_ARGS="$S learning.WARMUP_UPDATES=1" sb --export=ALL --time=00:30:00 --array=0 cluster/1_maze/train.sbatch)
  ALGOS="rplr sfl" sb --export=ALL --time=00:30:00 --array=0-1 --dependency=afterok:$j1:$j2 cluster/1_maze/falsify.sbatch
  ALGOS=sfl SFL_ARGS="$S" sb --export=ALL --time=00:30:00 --array=0 cluster/4_jaxnav/train.sbatch
  ALGOS=minimax SFL_ARGS="learning.NUM_ENVS=16 learning.NUM_STEPS=32 learning.TOTAL_TIMESTEPS=4096 learning.EVAL_FREQ=2 learning.NUM_CHECKPOINTS=2" \
    sb --export=ALL --time=00:30:00 --array=0 cluster/4_jaxnav/train.sbatch
  echo "smoke jobs submitted; check runs/slurm/*.out, then: python -m atlas.prelim $RUNS/maze_falsify" >&2
}

case ${1:-} in
  smoke) smoke;; 1) part1;; 2) part2;; 3) part3;; 4) part4;;
  all) part1; part2; part3; part4;;
  *) sed -n '2,10p' "$0"; exit 2;;
esac
