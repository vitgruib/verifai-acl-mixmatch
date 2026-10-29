#!/bin/bash
# Usage: bash cluster/submit.sh <target>   (DRY_RUN=1 prints the sbatch lines without submitting)
#   maze        Maze smoke test, then the full Maze suite and the CartPole bridge if it passes, then collect
#   maze-smoke  Maze smoke test only (~30 min)
#   smoke       smoke test of all three environments (~1 h; Kinetix speed probe up to 3 h)
#   exam-build  the 10 reference DR agents and the exam's VerifAI section only (CPU_JOBS at a time without a GPU)
#   bridge      the CartPole bridge only (BRIDGE_OUT in bridge.sbatch moves its scores)
#   1 | 2 | 3   full run of Maze | Kinetix | JaxNav;  all = 1-3
# Extra sbatch flags go in SBATCH_ARGS. LOCAL=1 runs the jobs on this machine instead of SLURM,
# JOBS_PER_GPU at a time on each GPU (for maze, chosen after the smoke test unless set); the GPUs are
# those in CUDA_VISIBLE_DEVICES if set, else every one nvidia-smi lists.
set -euo pipefail
cd "$(dirname "$0")/.."; source cluster/env.sh; mkdir -p runs/slurm
NSEEDS=${NSEEDS:-10}
# Once the exam's VerifAI section is committed (exam-build), the full run neither retrains the
# reference agents nor rebuilds the exam; smoke_report.py reads the same ALGOS.
exam_frozen && export ALGOS=${ALGOS:-"sfl dr plr rplr accel minimax"}
sb() {  # prints the job id
  echo "${LOCAL:+[local] }sbatch ${SBATCH_ARGS:-} $*" >&2
  if [ -n "${DRY_RUN:-}" ]; then echo DRYRUN
  elif [ -n "${LOCAL:-}" ]; then sb_local "$@"
  else sbatch --parsable ${SBATCH_ARGS:-} "$@"; fi
}
# Runs a job script here and returns when it is done, so dependencies hold by order. Array tasks
# go to the next free slot: JOBS_PER_GPU slots per GPU, each job seeing only its GPU and, when several
# share one, XLA_PYTHON_CLIENT_MEM_FRACTION 0.8 / JOBS_PER_GPU of its memory, leaving room for each
# process's CUDA context (one slot per
# cpus-per-task cores for CPU jobs); --time is enforced with timeout.
hms() { local IFS=:; set -- $1; echo $(( 10#$1 * 3600 + 10#$2 * 60 + 10#$3 )); }
clock() { date -d "@$1" '+%a %d %b %H:%M' 2>/dev/null || date -r "$1" '+%a %d %b %H:%M'; }
sb_local() {
  local a=0 t="" f="" x
  for x; do case $x in --array=*) a=${x#--array=};; --time=*) t=${x#--time=};; esac; f=$x; done
  local name c slots tmo="" tasks n st tmp pids=""
  name=$(sed -n 's/^#SBATCH --job-name=//p' "$f"); c=$(sed -n 's/^#SBATCH --cpus-per-task=//p' "$f")
  if grep -q '^#SBATCH --gres=gpu' "$f"; then
    slots=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr , ' ')
    [ -n "$slots" ] || slots=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' || true)
    slots=$(for s in $slots; do seq ${JOBS_PER_GPU:-1} | sed "s/.*/$s/"; done)
    [ -n "$slots" ] || slots=$(seq ${CPU_JOBS:-1} | sed 's/^/cpu/')   # no GPU: CPU_JOBS at a time
  else
    slots=$(seq 1 $(( $(getconf _NPROCESSORS_ONLN) / ${c:-1} > 0 ? $(getconf _NPROCESSORS_ONLN) / ${c:-1} : 1 )) | sed 's/^/cpu/')
  fi
  [ -n "$t" ] && command -v timeout >/dev/null && tmo="timeout $(hms "$t")"
  tasks=$(seq ${a%-*} ${a#*-}); n=$(echo $tasks | wc -w); st=$(date +%s); tmp=$(mktemp -d)
  for s in $slots; do
    ( for i in $tasks; do
        mkdir "$tmp/claim$i" 2>/dev/null || continue   # atomic: each task runs once, in order
        t0=$(date +%s); rc=0
        ( case $s in cpu*) ;; *) export CUDA_VISIBLE_DEVICES=$s
            [ "${JOBS_PER_GPU:-1}" -gt 1 ] && export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-$(
              awk "BEGIN {printf \"%.3f\", 0.8 / $JOBS_PER_GPU}")};; esac
          export SLURM_ARRAY_TASK_ID=$i SLURM_CPUS_PER_TASK=${c:-1} SLURM_SUBMIT_DIR=$PWD
          $tmo bash "$f" ) > "runs/slurm/${name}_${st}_$i.out" 2>&1 || rc=$?
        echo "$i $rc" >> "$tmp/done"; now=$(date +%s); d=$(wc -l < "$tmp/done")
        printf '%s  %s task %s %s in %d min (%d/%d done; this stage ends ~%s)\n' "$(date +%H:%M)" "$name" "$i" \
          "$([ $rc = 0 ] && echo ok || echo "FAILED rc=$rc")" $(( (now - t0) / 60 )) $d $n \
          "$(clock $(( now + (now - st) * (n - d) / d )))" >&2
      done ) &
    pids="$pids $!"
  done
  for x in $pids; do wait $x || true; done
  echo "local$st"
  ! awk '$2 != 0 {bad=1} END {exit !bad}' "$tmp/done"
}
count() { set -- $1; echo $(( $# * NSEEDS - 1 )); }

maze_time() { case $1 in sfl) echo 48:00:00;; minimax) echo 36:00:00;; *) echo 24:00:00;; esac; }
part1() {  # Maze training, exam and grading, and the CartPole bridge; prints the last jobs' ids
  local A=${ALGOS:-"sfl dr plr rplr accel minimax ref"} a t="" j e g b rc=0
  if [ -n "${LOCAL:-}" ]; then  # one queue over every slot, no time limit (jobs sharing a GPU run slower)
    t=$(ALGOS="$A" sb --export=ALL --array=0-$(( $(maze_tasks $A | wc -l) - 1 )) cluster/1_maze/train.sbatch) || rc=1
  else
    for a in $A; do
      j=$(ALGOS=$a sb --export=ALL ${GATE:-} --time=$(maze_time $a) --array=0-$(( $(maze_tasks $a | wc -l) - 1 )) \
        cluster/1_maze/train.sbatch); t=$t:$j
    done; t=${t#:}
  fi
  # the exam refuses unless all 10 reference agents finished; grading grades whatever finished
  if ! exam_frozen; then
    e=$(JOBS_PER_GPU=1 sb --export=ALL --dependency=afterany:$t cluster/1_maze/exam.sbatch) || rc=1; t=$t:$e
  fi
  g=$(JOBS_PER_GPU=1 sb --export=ALL --dependency=afterany:$t cluster/1_maze/grade.sbatch) || rc=1
  b=$(bridge) || rc=1
  echo "$g:$b"
  return $rc
}
exam_build() {  # the 10 reference agents, then the exam's VerifAI section; no smoke test
  local t; t=$(ALGOS=ref sb --export=ALL --array=0-9 cluster/1_maze/train.sbatch) || return 1
  JOBS_PER_GPU=1 sb --export=ALL --dependency=afterok:$t cluster/1_maze/exam.sbatch
}
bridge() {  # CartPole bridge: train the 3 arms, then grade and compare; prints the last job's id
  local b rc=0
  b=$(sb --export=ALL ${GATE:-} --array=0-$(( 3 * ${BRIDGE_SEEDS:-20} - 1 )) cluster/1_maze/bridge.sbatch) || rc=1
  BRIDGE_GRADE=1 sb --export=ALL --dependency=afterany:$b cluster/1_maze/bridge.sbatch || rc=1
  return $rc
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
# Maze SFL smoke run. On a GPU the full shape: 3 rounds of 20 updates, each scouting one batch of
# 5,000 levels, so a round is a fifth of a full one in training and scouting (smoke_report.py times
# it). On CPU a tiny shape. Rounds = NUM_CHECKPOINTS makes SFL save after every round.
SFL_SMOKE_GPU="learning.EVAL_FREQ=20 learning.NUM_CHECKPOINTS=3 NUM_BATCHES=1"
SFL_SMOKE_CPU="learning.NUM_ENVS=16 learning.NUM_ENVS_FROM_SAMPLED=8 learning.NUM_ENVS_TO_GENERATE=8 learning.NUM_STEPS=32 learning.EVAL_FREQ=2 learning.NUM_CHECKPOINTS=4 BATCH_SIZE=64 NUM_BATCHES=1 ROLLOUT_STEPS=50 NUM_TO_SAVE=32"
smoke_maze() {  # DR, RPLR, ACCEL, minimax and SFL, then an exam with 10 copies of the DR agent as reference, then grading
  local gpu="" p="" sa ss j t0 t1 t2
  nvidia-smi -L >/dev/null 2>&1 && gpu=1
  if [ -n "$gpu" ]; then sa=$SFL_SMOKE_GPU ss=3932160; else sa=$SFL_SMOKE_CPU ss=4096; fi
  if [ -n "${LOCAL:-}" ] && [ -n "$gpu" ]; then  # GPU memory while the smoke jobs run, one per GPU
    while :; do nvidia-smi ${CUDA_VISIBLE_DEVICES:+-i $CUDA_VISIBLE_DEVICES} --query-gpu=index,memory.used,memory.total \
      --format=csv,noheader,nounits; sleep 2; done \
      >> "$RUNS/gpu_mem.csv" 2>/dev/null &
    p=$!
  fi
  t0=$(date +%s)
  j=$(ALGOS="dr rplr accel minimax sfl" SFL_ARGS="$sa" SFL_STEPS=$ss XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JOBS_PER_GPU=1 sb --export=ALL --time=01:00:00 --array=0-4 cluster/1_maze/train.sbatch) || { [ -z "$p" ] || kill $p; return 1; }
  [ -z "$p" ] || kill $p
  t1=$(date +%s)
  j=$(REF_COPY=$RUNS/maze_train/checkpoints/maze_dr_n$NUM_UPDATES/0 REF_DIR=$RUNS/maze_ref EXAM_OUT=$RUNS/exam \
    POOL_OUT=$RUNS/exam_pool EXAM_ITERS=20 JOBS_PER_GPU=1 \
    sb --export=ALL ${KD:-} --time=01:00:00 --dependency=afterok:$j cluster/1_maze/exam.sbatch) || return 1
  t2=$(date +%s)
  j=$(EXAM_OUT=$RUNS/exam SCORES=$RUNS/scores.csv GRADE_ARGS="--arms DR RPLR ACCEL MINIMAX SFL" JOBS_PER_GPU=1 \
    sb --export=ALL ${KD:-} --time=01:00:00 --dependency=afterok:$j cluster/1_maze/grade.sbatch) || return 1
  # the first complete smoke test's times (a resumed one skips its work)
  [ -z "${LOCAL:-}" ] || [ -f "$RUNS/timing.txt" ] || printf 'train_sec %d\nexam_sec %d\nexam_iters 20\ngrade_sec %d\ngrade_runs 5\n' \
    $((t1 - t0)) $((t2 - t1)) $(($(date +%s) - t2)) > "$RUNS/timing.txt"
  echo "$j"
}
jobs_per_gpu() {  # local mode: prints the smoke test's report and sets JOBS_PER_GPU
  local k
  if k=$(PYTHONPATH=$ATLAS "$JAX_VENV/bin/python" cluster/1_maze/smoke_report.py "$RUNS/smoke"); then
    export JOBS_PER_GPU=$k
  else
    export JOBS_PER_GPU=${JOBS_PER_GPU:-1}; echo "smoke report failed; JOBS_PER_GPU=$JOBS_PER_GPU" >&2
  fi
}
maze_smoke() {
  (smoke_env; smoke_maze >/dev/null)
  [ -z "${LOCAL:-}" ] || jobs_per_gpu
  echo "maze smoke done or submitted; scores: $RUNS/smoke/scores.csv" >&2
}
maze() {  # KD cancels later jobs instead of leaving them pending if the smoke test fails
  export KD=--kill-on-invalid-dep=yes
  local j rc=0; j=$(smoke_env; smoke_maze)
  [ -z "${LOCAL:-}" ] || jobs_per_gpu
  j=$(GATE="$KD --dependency=afterok:$j" part1) || rc=1
  sb --export=ALL --dependency=afterany:$j cluster/collect.sbatch >/dev/null
  echo "maze smoke + full run $([ -n "${LOCAL:-}" ] && echo finished || echo submitted); results -> ${OUT:-$ATLAS/maze_results.tgz}" >&2
  return $rc
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
  maze) maze;; maze-smoke) maze_smoke;; smoke) smoke;; bridge) bridge;; exam-build) exam_build;; 1) part1;; 2) part2;; 3) part3;;
  all) part1; part2; part3;;
  *) sed -n '2,9p' "$0"; exit 2;;
esac
