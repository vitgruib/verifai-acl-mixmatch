#!/bin/bash
# One command per part. Prints every sbatch line it runs; DRY_RUN=1 prints without submitting.
#   bash cluster/submit.sh smoke      # ~1 h sanity check of all 3 parts (tiny budgets; train -> falsify), Kinetix probe up to 3 h
#   bash cluster/submit.sh 1          # Maze: train (60 tasks) -> falsify (60 tasks, after train)
#   bash cluster/submit.sh 2          # Kinetix: train (50 GPU tasks) -> falsify (50 CPU tasks, after train)
#   bash cluster/submit.sh 3          # JaxNav: train (60 tasks) -> falsify (60 tasks, after train)
#   bash cluster/submit.sh all        # 1-3
# Extra sbatch flags (partition, account, ...) go in SBATCH_ARGS, e.g. SBATCH_ARGS="-p gpu -A lab".
set -euo pipefail
cd "$(dirname "$0")/.."; source cluster/env.sh; mkdir -p runs/slurm
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
part2() {  # SFL gets its own array: its learnability rollouts need a longer limit (SFL_TIME)
  local A=${ALGOS:-dr plr rplr accel sfl} B j d=""; B=$(echo " $A " | sed 's/ sfl / /; s/^ *//; s/ *$//')
  [ -n "$B" ] && { j=$(ALGOS="$B" sb --export=ALL --array=0-$(count "$B") cluster/2_kinetix/train.sbatch); d=$d:$j; }
  [ "$B" != "$A" ] && { j=$(ALGOS=sfl sb --export=ALL --time=${SFL_TIME:-24:00:00} --array=0-$(count sfl) cluster/2_kinetix/train.sbatch); d=$d:$j; }
  ALGOS="$A" sb --export=ALL --array=0-$(count "$A") --dependency=afterok$d cluster/2_kinetix/falsify.sbatch
}
part3() { chain 3_jaxnav "${ALGOS:-dr minimax plr rplr accel sfl}"; }
chain() {  # chain <part dir> <algos>: train array -> falsify array (afterok)
  local j; j=$(ALGOS="$2" sb --export=ALL --array=0-$(count "$2") cluster/$1/train.sbatch)
  ALGOS="$2" sb --export=ALL --array=0-$(count "$2") --dependency=afterok:$j cluster/$1/falsify.sbatch
}

smoke() {  # one seed, one or two algos per codebase, tiny budgets; outputs under $RUNS/smoke
  local S="learning.NUM_ENVS=16 learning.NUM_ENVS_FROM_SAMPLED=8 learning.NUM_ENVS_TO_GENERATE=8 learning.NUM_STEPS=32 learning.TOTAL_TIMESTEPS=4096 learning.EVAL_FREQ=2 learning.NUM_CHECKPOINTS=2 BATCH_SIZE=64 NUM_BATCHES=1 ROLLOUT_STEPS=50 NUM_TO_SAVE=32"
  local R=$RUNS
  export RUNS=$RUNS/smoke NSEEDS=1 NUM_UPDATES=50 JAXUED_ARGS="--eval_freq 10 --checkpoint_save_interval 1" BUDGET=50 FSEEDS=0 SAMPLERS=ce; mkdir -p "$RUNS"
  local j1 j2
  j1=$(ALGOS=rplr sb --export=ALL --time=00:30:00 --array=0 cluster/1_maze/train.sbatch)
  j2=$(ALGOS=sfl SFL_ARGS="$S learning.WARMUP_UPDATES=1" sb --export=ALL --time=00:30:00 --array=0 cluster/1_maze/train.sbatch)
  ALGOS="rplr sfl" sb --export=ALL --time=00:30:00 --array=0-1 --dependency=afterok:$j1:$j2 cluster/1_maze/falsify.sbatch
  # Kinetix: ACCEL (plr.py) and SFL (sfl.py) at 32 envs, 16 / 64 updates (sfl has no level_buffer_capacity)
  local K="learning.num_train_envs=32 learning.num_minibatches=4 learning.update_epochs=2 eval.eval_num_attempts=2"
  local KS="learning.num_steps=64 ued.batch_size=64 ued.rollout_steps=64 ued.num_to_save=32"
  local k1 k2
  k1=$(ALGOS=accel STEPS=131072 EVAL_FREQ=4 CKPT_FREQ=8 KINETIX_ARGS="$K ued.level_buffer_capacity=64" sb --export=ALL --time=00:30:00 --array=0 cluster/2_kinetix/train.sbatch)
  k2=$(ALGOS=sfl STEPS=131072 EVAL_FREQ=4 CKPT_FREQ=8 KINETIX_ARGS="$K $KS" sb --export=ALL --time=00:30:00 --array=0 cluster/2_kinetix/train.sbatch)
  ALGOS="accel sfl" sb --export=ALL --time=00:30:00 --array=0-1 --dependency=afterok:$k1:$k2 cluster/2_kinetix/falsify.sbatch
  # Kinetix GPU probe at full size; the .out ends with steps/s and full_run_h (projected 201M-step run).
  # DR: 32 updates (1/12 of a run). SFL: 48 updates (1/8 of a run, incl. one full 134M-step learnability refresh).
  ALGOS=dr RUNS=$R/probe STEPS=16777216 CKPT_FREQ=32 sb --export=ALL --time=01:00:00 --array=0 cluster/2_kinetix/train.sbatch >/dev/null
  ALGOS=sfl RUNS=$R/probe STEPS=25165824 CKPT_FREQ=48 sb --export=ALL --time=03:00:00 --array=0 cluster/2_kinetix/train.sbatch >/dev/null
  local j3; j3=$(ALGOS=sfl SFL_ARGS="$S" sb --export=ALL --time=00:30:00 --array=0 cluster/3_jaxnav/train.sbatch)
  ALGOS=sfl sb --export=ALL --time=00:30:00 --array=0 --dependency=afterok:$j3 cluster/3_jaxnav/falsify.sbatch
  ALGOS=minimax SFL_ARGS="learning.NUM_ENVS=16 learning.NUM_STEPS=32 learning.TOTAL_TIMESTEPS=4096 learning.EVAL_FREQ=2 learning.NUM_CHECKPOINTS=2" \
    sb --export=ALL --time=00:30:00 --array=0 cluster/3_jaxnav/train.sbatch >/dev/null
  echo "smoke jobs submitted; check runs/slurm/*.out (probe: full_run_h on the last line of runs/slurm/atlas-2-kinetix_*.out), then: python -m atlas.prelim $RUNS/maze_falsify" >&2
}

case ${1:-} in
  smoke) smoke;; 1) part1;; 2) part2;; 3) part3;;
  all) part1; part2; part3;;
  *) sed -n '2,8p' "$0"; exit 2;;
esac
