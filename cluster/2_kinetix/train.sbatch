#!/bin/bash
# PART 2 - Kinetix (FLAIROx, ICLR 2025): train on the S-size symbolic entity env, random levels,
# evaluated on the 10 hand-designed S levels. Upstream scripts and configs, plus our bug-fix patch.
# DR, PLR, Robust PLR, ACCEL via experiments/plr.py; SFL via experiments/sfl.py (Kinetix has no minimax).
# One task = one (algo, seed). Default: 5 algos x 10 seeds = 50 tasks.
#   sbatch --array=0-49 cluster/2_kinetix/train.sbatch
# Budget: 384 PPO updates x 2048 envs x 256 steps = 201M env steps per run (paper: 1B+), a tight
# cut, not paper-faithful. GPU speed is unmeasured; `submit.sh smoke` prints it. The 6 h limit
# assumes >= ~20k steps/s. SFL keeps the paper's learnability sampling (16 batches x 16384 levels x
# 512 steps = 134M rollout steps per refresh, refresh every 48 updates; paper 50), i.e. ~1.07B
# rollout steps, ~5x the training steps as in the paper. submit.sh runs SFL as its own array with
# SFL_TIME (default 24 h); the smoke probe prints its projected wall time.
# DRY_RUN=1 prints the command instead of running it.
#SBATCH --job-name=atlas-2-kinetix
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
ALGOS=${ALGOS:-"dr plr rplr accel sfl"}; NSEEDS=${NSEEDS:-10}
STEPS=${STEPS:-201326592}                       # 384 updates of 2^19 steps (plr: 64 x 4 outer; sfl: 256)
CKPT_FREQ=${CKPT_FREQ:-128}                     # in updates; must divide the update count
i=${SLURM_ARRAY_TASK_ID:-0}
algo=$(pick $((i / NSEEDS)) $ALGOS); seed=$((i % NSEEDS))
# eval every 32 updates; sfl every 48 (its learnability refresh, closest divisor of 384 to the paper's 50)
[ $algo = sfl ] && d=48 || d=32; EVAL_FREQ=${EVAL_FREQ:-$d}
out="$RUNS/kinetix/$algo/s$seed"
# eval on the 10 S levels every EVAL_FREQ updates; KINETIX_ARGS must not repeat a key set here
C="env=symbolic env_size=s eval=s seed=$seed learning.total_timesteps=$STEPS eval.eval_freq=$EVAL_FREQ
   misc.checkpoint_save_freq=$CKPT_FREQ misc.use_wandb=false misc.wandb_mode=disabled
   misc.save_path=$out/ckpt hydra.run.dir=$out"
case $algo in
  dr)    s=plr; x="ued.replay_prob=0";;
  plr)   s=plr; x="";;                                    # configs/ued/plr.yaml: MaxMC, replay 0.5
  rplr)  s=plr; x="ued.exploratory_grad_updates=false";;
  # ued=accel lacks accel_start_from_empty and crashes; the plr config with use_accel is the same algo
  accel) s=plr; x="ued.use_accel=true ued.num_edits=5";;
  # buffer_update_frequency == eval_freq: upstream recomputes the learnability set on every update
  # otherwise. num_batches, batch_size, rollout_steps: configs/ued/sfl.yaml (paper values).
  sfl)   s=sfl; x="ued.buffer_update_frequency=$EVAL_FREQ";;
  *) echo "unknown algo $algo"; exit 2;;
esac
echo "task=$i algo=$algo seed=$seed out=$out host=$(hostname)"
if [ -n "${DRY_RUN:-}" ]; then echo "+ python experiments/$s.py" $C $x ${KINETIX_ARGS:-}; exit 0; fi
source "$KINETIX_VENV/bin/activate"; mkdir -p "$out" && cd "$KINETIX_DIR"
export MPLBACKEND=Agg SDL_VIDEODRIVER=dummy WANDB_MODE=disabled
python experiments/$s.py $C $x ${KINETIX_ARGS:-}
# wall time incl. JIT compile; steps/s counts training steps only (not SFL learnability rollouts).
# full = this run's wall time scaled to the default 201M steps (exact for the smoke probe's shapes).
full=$(awk -v w=$SECONDS -v n=$STEPS 'BEGIN{printf "%.1f", w * 201326592 / n / 3600}')
echo "done algo=$algo seed=$seed steps=$STEPS wall=${SECONDS}s steps/s=$((STEPS / (SECONDS > 0 ? SECONDS : 1))) full_run_h=$full"
