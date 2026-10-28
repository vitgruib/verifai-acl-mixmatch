#!/bin/bash
# Kinetix training. One array task per (algo, seed): 5 algos x 10 seeds = 50 tasks.
#SBATCH --job-name=atlas-2-kinetix
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
ALGOS=${ALGOS:-"dr plr rplr accel sfl"}; NSEEDS=${NSEEDS:-10}
STEPS=${STEPS:-201326592}  # 384 updates of 2^19 steps
CKPT_FREQ=${CKPT_FREQ:-128}      # in updates; must divide the update count
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
# sfl evaluates at its learnability refresh, every 48 updates
[ $algo = sfl ] && d=48 || d=32; EVAL_FREQ=${EVAL_FREQ:-$d}
out="$RUNS/kinetix/$algo/s$seed"
# KINETIX_ARGS must not repeat a key set here
C="env=symbolic env_size=s eval=s seed=$seed learning.total_timesteps=$STEPS eval.eval_freq=$EVAL_FREQ
   misc.checkpoint_save_freq=$CKPT_FREQ misc.use_wandb=false misc.wandb_mode=disabled
   misc.save_path=$out/ckpt hydra.run.dir=$out"
case $algo in
  dr)    s=plr; x="ued.replay_prob=0";;
  plr)   s=plr; x="";;
  rplr)  s=plr; x="ued.exploratory_grad_updates=false";;
  # ued=accel crashes upstream; the plr config with use_accel is the same algorithm
  accel) s=plr; x="ued.use_accel=true ued.num_edits=5";;
  # without this, upstream recomputes the learnability set on every update
  sfl)   s=sfl; x="ued.buffer_update_frequency=$EVAL_FREQ";;
  *) echo "unknown algo $algo"; exit 2;;
esac
echo "task=$i algo=$algo seed=$seed out=$out host=$(hostname)"
if [ -n "${DRY_RUN:-}" ]; then echo "+ python experiments/$s.py" $C $x ${KINETIX_ARGS:-}; exit 0; fi
source "$KINETIX_VENV/bin/activate"; mkdir -p "$out" && cd "$KINETIX_DIR"
export MPLBACKEND=Agg SDL_VIDEODRIVER=dummy WANDB_MODE=disabled
python experiments/$s.py $C $x ${KINETIX_ARGS:-}
# full_run_h: this run's wall time scaled to the default 201M steps (used by the speed probe)
full=$(awk -v w=$SECONDS -v n=$STEPS 'BEGIN{printf "%.1f", w * 201326592 / n / 3600}')
echo "done algo=$algo seed=$seed steps=$STEPS wall=${SECONDS}s steps/s=$((STEPS / (SECONDS > 0 ? SECONDS : 1))) full_run_h=$full"
