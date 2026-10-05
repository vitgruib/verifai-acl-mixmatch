#!/bin/bash
# PART 2 - CarRacing: train with DCD CarRacing (Robust PLR paper), 5.5M steps per run, rendered under Xvfb.
# NEVER run on a desktop: it opens pyglet windows without xvfb-run.
# DR ~11-13 h if 16 procs give ~120-140 steps/s (70/s measured on 8 local procs); cr_sfl ~2x (~25 h).
# Unmeasured on GPU, so submit.sh chains SEGMENTS=2 arrays of 24 h (afterany): a run that times out
# resumes from its model.tar (checkpoint every 100 updates; resume tested locally), a finished one
# exits after its final test. By hand, resubmit the same array index:
#   sbatch --array=<idx> cluster/2_carracing/train.sbatch runs/cr_cmds.txt
#   bash cluster/dcd_cmds.sh car_racing 1 cr_dr cr_minimax cr_plr cr_robust_plr cr_accel cr_sfl > runs/cr_cmds.txt
#   sbatch --array=0-$(( $(wc -l < runs/cr_cmds.txt) - 1 )) cluster/2_carracing/train.sbatch runs/cr_cmds.txt
#SBATCH --job-name=atlas-2-carracing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-.}"
grep -q '^xvfb-run' "$1" || { echo "refusing: commands lack xvfb-run (regenerate with xvfb=1)"; exit 1; }
exec bash cluster/dcd_task.sh "$1"
