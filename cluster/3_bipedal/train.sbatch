#!/bin/bash
# PART 3 - BipedalWalker: train with DCD BipedalWalker (ACCEL paper config). The paper trains 2B env
# steps (~63 h at the 8.8k steps/s we measured); submit.sh cuts it to 300M via
# DCD_ARGS=--num_env_steps=300000000 (~10 h; bipedal_sfl ~13 h with its level scoring). No DCD schedule
# depends on the total, so this just stops training earlier. A run that times out resumes from its
# model.tar (saved every 100 updates; resume tested locally): resubmit the same index, or SEGMENTS=2.
#   sbatch --array=<idx> cluster/3_bipedal/train.sbatch runs/bip_cmds.txt
#   DCD_ARGS=--num_env_steps=300000000 bash cluster/dcd_cmds.sh bipedal 0 bipedal_dr bipedal_minimax bipedal_plr bipedal_robust_plr bipedal_accel bipedal_sfl > runs/bip_cmds.txt
#   sbatch --array=0-$(( $(wc -l < runs/bip_cmds.txt) - 1 )) cluster/3_bipedal/train.sbatch runs/bip_cmds.txt
#SBATCH --job-name=atlas-3-bipedal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --requeue
#SBATCH --output=runs/slurm/%x_%A_%a.out
cd "${SLURM_SUBMIT_DIR:-.}"
exec bash cluster/dcd_task.sh "$1"
