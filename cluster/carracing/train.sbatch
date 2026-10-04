#!/bin/bash
# Part 2: DCD CarRacing (Robust PLR paper), 5.5M steps per run, rendered under Xvfb.
# NEVER run on a desktop: it opens pyglet windows without xvfb-run.
#   bash cluster/dcd_cmds.sh car_racing 1 cr_dr cr_minimax cr_plr cr_robust_plr cr_paired cr_repaired > runs/cr_cmds.txt
#   sbatch --array=0-$(( $(wc -l < runs/cr_cmds.txt) - 1 )) cluster/carracing/train.sbatch runs/cr_cmds.txt
#SBATCH --job-name=atlas-cr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-.}"
grep -q '^xvfb-run' "$1" || { echo "refusing: commands lack xvfb-run (regenerate with xvfb=1)"; exit 1; }
exec bash cluster/dcd_task.sh "$1"
