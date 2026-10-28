#!/bin/bash
# Usage: bash cluster/submit.sh <target>   (DRY_RUN=1 prints the sbatch lines without submitting)
#   maze        Maze smoke test, then the full Maze run if it passes, then collect
#   maze-smoke  Maze smoke test only (~30 min)
#   smoke       smoke test of all three environments (~1 h; Kinetix speed probe up to 3 h)
#   1 | 2 | 3   full run of Maze | Kinetix | JaxNav;  all = 1-3
# Extra sbatch flags go in SBATCH_ARGS. LOCAL=1 runs the jobs on this machine instead of SLURM.
set -euo pipefail
cd "$(dirname "$0")/.."; source cluster/env.sh; mkdir -p runs/slurm
NSEEDS=${NSEEDS:-10}
sb() {  # prints the job id
  echo "${LOCAL:+[local] }sbatch ${SBATCH_ARGS:-} $*" >&2
  if [ -n "${DRY_RUN:-}" ]; then echo DRYRUN
  elif [ -n "${LOCAL:-}" ]; then sb_local "$@"
  else sbatch --parsable ${SBATCH_ARGS:-} "$@"; fi
}
# Runs a job script here and returns when it is done, so dependencies hold by order. Array tasks
# run one per GPU (one per cpus-per-task cores for CPU jobs); --time is enforced with timeout.
hms() { local IFS=:; set -- $1; echo $(( 10#$1 * 3600 + 10#$2 * 60 + 10#$3 )); }
clock() { date -d "@$1" '+%a %d %b %H:%M' 2>/dev/null || date -r "$1" '+%a %d %b %H:%M'; }
sb_local() {
  local a=0 t="" f="" x
  for x; do case $x in --array=*) a=${x#--array=};; --time=*) t=${x#--time=};; esac; f=$x; done
  local name c slots tmo="" tasks n k=0 st tmp pids=""
  name=$(sed -n 's/^#SBATCH --job-name=//p' "$f"); c=$(sed -n 's/^#SBATCH --cpus-per-task=//p' "$f")
  if grep -q '^#SBATCH --gres=gpu' "$f"; then
    slots=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
    [ -n "$slots" ] || slots=cpu
  else
    slots=$(seq 1 $(( $(getconf _NPROCESSORS_ONLN) / ${c:-1} > 0 ? $(getconf _NPROCESSORS_ONLN) / ${c:-1} : 1 )) | sed 's/^/cpu/')
  fi
  [ -n "$t" ] && command -v timeout >/dev/null && tmo="timeout $(hms "$t")"
  tasks=$(seq ${a%-*} ${a#*-}); n=$(echo $tasks | wc -w); st=$(date +%s); tmp=$(mktemp -d)
  for s in $slots; do
    ( j=0
      for i in $tasks; do
        [ $(( j++ % $(echo $slots | wc -w) )) -eq $k ] || continue
        t0=$(date +%s); rc=0
        ( case $s in cpu*) ;; *) export CUDA_VISIBLE_DEVICES=$s;; esac
          export SLURM_ARRAY_TASK_ID=$i SLURM_CPUS_PER_TASK=${c:-1} SLURM_SUBMIT_DIR=$PWD
          $tmo bash "$f" ) > "runs/slurm/${name}_${st}_$i.out" 2>&1 || rc=$?
        echo "$i $rc" >> "$tmp/done"; now=$(date +%s); d=$(wc -l < "$tmp/done")
        printf '%s  %s task %s %s in %d min (%d/%d done; this stage ends ~%s)\n' "$(date +%H:%M)" "$name" "$i" \
          "$([ $rc = 0 ] && echo ok || echo "FAILED rc=$rc")" $(( (now - t0) / 60 )) $d $n \
          "$(clock $(( now + (now - st) * (n - d) / d )))" >&2
      done ) &
    pids="$pids $!"; k=$((k + 1))
  done
  for x in $pids; do wait $x || true; done
  echo "local$st"
  ! awk '$2 != 0 {bad=1} END {exit !bad}' "$tmp/done"
}
count() { set -- $1; echo $(( $# * NSEEDS - 1 )); }

part1() {
  local A=${ALGOS:-"dr plr rplr accel minimax sfl"}
  local j; j=$(ALGOS="$A" sb --export=ALL ${GATE:-} --array=0-$(count "$A") cluster/1_maze/train.sbatch)
  ALGOS="$A" sb --export=ALL ${KD:-} --array=0-$(count "$A") --dependency=afterok:$j cluster/1_maze/falsify.sbatch
}
part2() {  # SFL runs as its own array with a longer time limit
  local A=${ALGOS:-dr plr rplr accel sfl} B j d=""; B=$(echo " $A " | sed 's/ sfl / /; s/^ *//; s/ *$//')
  [ -n "$B" ] && { j=$(ALGOS="$B" sb --export=ALL --array=0-$(count "$B") cluster/2_kinetix/train.sbatch); d=$d:$j; }
  [ "$B" != "$A" ] && { j=$(ALGOS=sfl sb --export=ALL --time=${SFL_TIME:-24:00:00} --array=0-$(count sfl) cluster/2_kinetix/train.sbatch); d=$d:$j; }
  ALGOS="$A" sb --export=ALL --array=0-$(count "$A") --dependency=afterok$d cluster/2_kinetix/falsify.sbatch
}
part3() { chain 3_jaxnav "${ALGOS:-dr minimax plr rplr accel sfl}"; }
chain() {  # <part dir> <algos>
  local j; j=$(ALGOS="$2" sb --export=ALL --array=0-$(count "$2") cluster/$1/train.sbatch)
  ALGOS="$2" sb --export=ALL --array=0-$(count "$2") --dependency=afterok:$j cluster/$1/falsify.sbatch
}

# Smoke test settings: one seed, tiny budgets, outputs under $RUNS/smoke.
S="learning.NUM_ENVS=16 learning.NUM_ENVS_FROM_SAMPLED=8 learning.NUM_ENVS_TO_GENERATE=8 learning.NUM_STEPS=32 learning.TOTAL_TIMESTEPS=4096 learning.EVAL_FREQ=2 learning.NUM_CHECKPOINTS=2 BATCH_SIZE=64 NUM_BATCHES=1 ROLLOUT_STEPS=50 NUM_TO_SAVE=32"
smoke_env() {
  export RUNS=$RUNS/smoke NSEEDS=1 NUM_UPDATES=50 JAXUED_ARGS="--eval_freq 10 --checkpoint_save_interval 1" BUDGET=50 FSEEDS=0 SAMPLERS="random ce"; mkdir -p "$RUNS"
}
smoke_maze() {
  local j1 j2
  j1=$(ALGOS=rplr sb --export=ALL --time=00:30:00 --array=0 cluster/1_maze/train.sbatch)
  j2=$(ALGOS=sfl SFL_ARGS="$S learning.WARMUP_UPDATES=1" sb --export=ALL --time=00:30:00 --array=0 cluster/1_maze/train.sbatch)
  ALGOS="rplr sfl" sb --export=ALL ${KD:-} --time=00:30:00 --array=0-1 --dependency=afterok:$j1:$j2 cluster/1_maze/falsify.sbatch
}
maze_smoke() {
  smoke_env
  smoke_maze
  echo "maze smoke submitted; check: ls $RUNS/maze_falsify/*/*/*/summary.json" >&2
}
maze() {  # KD cancels later jobs instead of leaving them pending if the smoke test fails
  export KD=--kill-on-invalid-dep=yes
  local j; j=$(smoke_env; smoke_maze)
  j=$(GATE="$KD --dependency=afterok:$j" part1)
  sb --export=ALL --dependency=afterany:$j cluster/collect.sbatch >/dev/null
  echo "maze smoke + full run submitted; results -> ${OUT:-$ATLAS/maze_results.tgz}" >&2
}

smoke() {
  local R=$RUNS
  smoke_env
  smoke_maze
  local K="learning.num_train_envs=32 learning.num_minibatches=4 learning.update_epochs=2 eval.eval_num_attempts=2"
  local KS="learning.num_steps=64 ued.batch_size=64 ued.rollout_steps=64 ued.num_to_save=32"
  local k1 k2
  k1=$(ALGOS=accel STEPS=131072 EVAL_FREQ=4 CKPT_FREQ=8 KINETIX_ARGS="$K ued.level_buffer_capacity=64" sb --export=ALL --time=00:30:00 --array=0 cluster/2_kinetix/train.sbatch)
  k2=$(ALGOS=sfl STEPS=131072 EVAL_FREQ=4 CKPT_FREQ=8 KINETIX_ARGS="$K $KS" sb --export=ALL --time=00:30:00 --array=0 cluster/2_kinetix/train.sbatch)
  ALGOS="accel sfl" sb --export=ALL --time=00:30:00 --array=0-1 --dependency=afterok:$k1:$k2 cluster/2_kinetix/falsify.sbatch
  # full-size Kinetix speed probe; the log ends with full_run_h, the projected time of a real run
  ALGOS=dr RUNS=$R/probe STEPS=16777216 CKPT_FREQ=32 sb --export=ALL --time=01:00:00 --array=0 cluster/2_kinetix/train.sbatch >/dev/null
  ALGOS=sfl RUNS=$R/probe STEPS=25165824 CKPT_FREQ=48 sb --export=ALL --time=03:00:00 --array=0 cluster/2_kinetix/train.sbatch >/dev/null
  local j3; j3=$(ALGOS=sfl SFL_ARGS="$S" sb --export=ALL --time=00:30:00 --array=0 cluster/3_jaxnav/train.sbatch)
  ALGOS=sfl sb --export=ALL --time=00:30:00 --array=0 --dependency=afterok:$j3 cluster/3_jaxnav/falsify.sbatch
  ALGOS=minimax SFL_ARGS="learning.NUM_ENVS=16 learning.NUM_STEPS=32 learning.TOTAL_TIMESTEPS=4096 learning.EVAL_FREQ=2 learning.NUM_CHECKPOINTS=2" \
    sb --export=ALL --time=00:30:00 --array=0 cluster/3_jaxnav/train.sbatch >/dev/null
  echo "smoke jobs submitted; check runs/slurm/*.out (probe: full_run_h on the last line of runs/slurm/atlas-2-kinetix_*.out), then: python -m atlas.prelim $RUNS/maze_falsify" >&2
}

case ${1:-} in
  maze) maze;; maze-smoke) maze_smoke;; smoke) smoke;; 1) part1;; 2) part2;; 3) part3;;
  all) part1; part2; part3;;
  *) sed -n '2,7p' "$0"; exit 2;;
esac
