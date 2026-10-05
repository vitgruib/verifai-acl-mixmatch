#!/bin/bash
# One command per part. Prints every sbatch line it runs; DRY_RUN=1 prints without submitting.
#   bash cluster/submit.sh smoke      # <1 h sanity check of all 4 parts (tiny budgets; train -> falsify per codebase)
#   bash cluster/submit.sh 1          # Maze: train (30 tasks) -> falsify (30 tasks, after train)
#   bash cluster/submit.sh 2          # CarRacing: train (30 tasks, Xvfb) -> falsify (30 CPU tasks, after train)
#   bash cluster/submit.sh 3          # BipedalWalker: train (30 tasks) -> falsify (30 CPU tasks, after train)
#   bash cluster/submit.sh 4          # JaxNav: train (30 tasks) -> falsify (30 tasks, after train)
#   bash cluster/submit.sh all        # 1-4
# Part 2 chains SEGMENTS=2 24 h train arrays (resume after timeout) before falsify; Part 3 trains
# Bipedal for 300M env steps (paper: 2B) via DCD_ARGS, in one segment.
# Extra sbatch flags (partition, account, ...) go in SBATCH_ARGS, e.g. SBATCH_ARGS="-p gpu -A lab".
set -euo pipefail
cd "$(dirname "$0")/.."; source cluster/env.sh; mkdir -p "$RUNS/slurm" runs/slurm
NSEEDS=${NSEEDS:-5}
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
  local j; j=$(sb --array=0-$(( $(wc -l < "$f") - 1 )) cluster/$p/train.sbatch "$f")
  # SEGMENTS>1 chains continuation arrays on the same command file (afterany = also after a timeout):
  # DCD resumes each run from its model.tar; a finished run resumes into an empty loop and exits.
  local k; for ((k = 1; k < ${SEGMENTS:-1}; k++)); do
    j=$(sb --array=0-$(( $(wc -l < "$f") - 1 )) --dependency=afterany:$j cluster/$p/train.sbatch "$f")
  done
  # falsify algo names from the configs (cr_robust_plr -> rplr); dcd_ckpt.sh maps them back
  local A; A=$(for c in "$@"; do c=${c#*_}; [ $c = robust_plr ] && c=rplr; echo -n "$c "; done)
  ALGOS="$A" sb --export=ALL --array=0-$(count "$A") --dependency=afterok:$j cluster/$p/falsify.sbatch
}
part2() { SEGMENTS=${SEGMENTS:-2} DCD_ARGS=${DCD_ARGS:-} dcd 2_carracing car_racing 1 ${CONFIGS:-cr_dr cr_minimax cr_plr cr_robust_plr cr_accel cr_sfl}; }
part3() { SEGMENTS=${SEGMENTS:-1} DCD_ARGS=${DCD_ARGS:---num_env_steps=300000000} dcd 3_bipedal bipedal 0 ${CONFIGS:-bipedal_dr bipedal_minimax bipedal_plr bipedal_robust_plr bipedal_accel bipedal_sfl}; }
part4() {
  local A=${ALGOS:-"dr minimax plr rplr accel sfl"}
  local j; j=$(ALGOS="$A" sb --export=ALL --array=0-$(count "$A") cluster/4_jaxnav/train.sbatch)
  ALGOS="$A" sb --export=ALL --array=0-$(count "$A") --dependency=afterok:$j cluster/4_jaxnav/falsify.sbatch
}

dsmoke() {  # dsmoke <part dir> <domain> <xvfb> <config> <overrides>: one DCD train -> falsify
  local p=$1 dom=$2 x=$3 c=$4 f=$RUNS/${2}_smoke_cmds.txt
  bash cluster/dcd_cmds.sh $dom $x $c | sed "s|\$| $5|" > "$f"
  local j; j=$(sb --export=ALL --time=01:00:00 --array=0 cluster/$p/train.sbatch "$f")
  ALGOS=sfl BUDGET=20 sb --export=ALL --time=01:00:00 --array=0 --dependency=afterok:$j cluster/$p/falsify.sbatch
}
smoke() {  # one seed, one algo per codebase, tiny budgets; outputs under $RUNS/smoke
  local S="learning.NUM_ENVS=16 learning.NUM_ENVS_FROM_SAMPLED=8 learning.NUM_ENVS_TO_GENERATE=8 learning.NUM_STEPS=32 learning.TOTAL_TIMESTEPS=4096 learning.EVAL_FREQ=2 learning.NUM_CHECKPOINTS=2 BATCH_SIZE=64 NUM_BATCHES=1 ROLLOUT_STEPS=50 NUM_TO_SAVE=32"
  export RUNS=$RUNS/smoke NSEEDS=1 NUM_UPDATES=50 JAXUED_ARGS="--eval_freq 10 --checkpoint_save_interval 1" BUDGET=50 FSEEDS=0 SAMPLERS=ce; mkdir -p "$RUNS"
  local j1 j2
  j1=$(ALGOS=rplr sb --export=ALL --time=00:30:00 --array=0 cluster/1_maze/train.sbatch)
  j2=$(ALGOS=sfl SFL_ARGS="$S learning.WARMUP_UPDATES=1" sb --export=ALL --time=00:30:00 --array=0 cluster/1_maze/train.sbatch)
  ALGOS="rplr sfl" sb --export=ALL --time=00:30:00 --array=0-1 --dependency=afterok:$j1:$j2 cluster/1_maze/falsify.sbatch
  local j3; j3=$(ALGOS=sfl SFL_ARGS="$S" sb --export=ALL --time=00:30:00 --array=0 cluster/4_jaxnav/train.sbatch)
  ALGOS=sfl sb --export=ALL --time=00:30:00 --array=0 --dependency=afterok:$j3 cluster/4_jaxnav/falsify.sbatch
  # DCD: the SFL port per domain (paper config + tiny-budget overrides; argparse keeps the last value)
  local T="--num_processes=4 --test_interval=2 --test_num_episodes=1 --test_num_processes=1 --sfl_eval_interval=1 --sfl_num_eval_levels=8 --sfl_buffer_size=4"
  dsmoke 2_carracing car_racing 1 cr_sfl "$T --num_env_steps=2000 --sfl_eval_steps=125"
  dsmoke 3_bipedal bipedal 0 bipedal_sfl "$T --num_env_steps=16384 --sfl_eval_steps=200"
  ALGOS=minimax SFL_ARGS="learning.NUM_ENVS=16 learning.NUM_STEPS=32 learning.TOTAL_TIMESTEPS=4096 learning.EVAL_FREQ=2 learning.NUM_CHECKPOINTS=2" \
    sb --export=ALL --time=00:30:00 --array=0 cluster/4_jaxnav/train.sbatch
  echo "smoke jobs submitted; check runs/slurm/*.out, then: python -m atlas.prelim $RUNS/maze_falsify" >&2
}

case ${1:-} in
  smoke) smoke;; 1) part1;; 2) part2;; 3) part3;; 4) part4;;
  all) part1; part2; part3; part4;;
  *) sed -n '2,10p' "$0"; exit 2;;
esac
