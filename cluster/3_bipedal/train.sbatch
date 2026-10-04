#!/bin/bash
# PART 3 - BipedalWalker: train with DCD BipedalWalker (ACCEL paper). Paper budget is 2B env steps per run
# (~63 h at the 8.8k steps/s we measured), so expect requeues; DCD resumes from its
# checkpoint because make_cmd sets --checkpoint=True.
#   bash cluster/dcd_cmds.sh bipedal 0 bipedal_dr bipedal_minimax bipedal_plr bipedal_robust_plr bipedal_accel > runs/bip_cmds.txt
#   sbatch --array=0-$(( $(wc -l < runs/bip_cmds.txt) - 1 )) cluster/3_bipedal/train.sbatch runs/bip_cmds.txt
#SBATCH --job-name=atlas-3-bipedal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --output=runs/slurm/%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-.}"
exec bash cluster/dcd_task.sh "$1"
