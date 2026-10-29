#!/bin/bash
# Maze training. One array task per line of `maze_tasks $ALGOS` (env.sh): the 10 reference DR agents
# (ref, seeds 100-109) and 6 methods x seeds 0-9 = 70 tasks. A run whose final checkpoint is already
# there is skipped, and an unfinished one is deleted and trained again, so rerunning resumes.
#SBATCH --job-name=atlas-1-maze-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm/%x_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source cluster/env.sh
ALGOS=${ALGOS:-"sfl ref dr plr rplr accel minimax"}; N=${NUM_UPDATES:-30000}
i=${SLURM_ARRAY_TASK_ID:-0}
read -r algo seed <<< "$(maze_tasks $ALGOS | sed -n "$((i + 1))p")"
echo "task=$i algo=$algo seed=$seed host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-}"
run() { if [ -n "${DRY_RUN:-}" ]; then echo "+ $*"; else "$@"; fi; }
stamp() { while IFS= read -r l; do printf '%s %s\n' "$(date +%s)" "$l"; done; }   # smoke_report.py times SFL's rounds

if [ "$algo" = sfl ]; then
  # SFL saves to SAVE_PATH/<wandb run name>, which is None offline unless WANDB_NAME is set.
  # config.json is written only once training has finished, so it marks a finished run.
  out="$RUNS/sfl/maze_sfl"; d=$out/s$seed; export WANDB_NAME=s$seed
  [ -f "$d/config.json" ] && { echo "already trained: $d"; exit 0; }
  # Recipe A of docs/maze_gpu_audit.md (4-6): the repo's Minigrid settings at 25 walls and the Maze budget
  args="SEED=$seed WANDB_MODE=offline SAVE_PATH=$out hydra.run.dir=$d/hydra env.N_WALLS=25"
  args="$args learning.TOTAL_TIMESTEPS=${SFL_STEPS:-245760000} ${SFL_ARGS:-}"
  if [ -n "${DRY_RUN:-}" ]; then echo "+ python sfl/train/minigrid_sfl.py $args"; exit 0; fi
  source "$SFL_VENV/bin/activate"; rm -rf "$d"; mkdir -p "$d"
  (cd "$SFL_DIR" && PYTHONUNBUFFERED=1 python sfl/train/minigrid_sfl.py $args) 2>&1 | stamp | tee "$d/train.log"
  # the settings Hydra composed, under the lower-cased names atlas.maze.grade and Policy read
  python - "$d" "$SFL_COMMIT" <<'EOF'
import json, os, sys, yaml
d, commit = sys.argv[1:]
c = yaml.safe_load(open(os.path.join(d, "hydra", ".hydra", "config.yaml")))
lr = c["learning"]
cfg = {"algo": "sfl", "seed": c["SEED"], "sfl_commit": commit, "n_walls": c["env"]["N_WALLS"],
       "agent_view_size": c["env"]["AGENT_VIEW_SIZE"], "total_timesteps": int(lr["TOTAL_TIMESTEPS"]),
       **{k.lower(): lr[k] for k in ("NUM_STEPS", "NUM_ENVS", "EVAL_FREQ")},
       **{k.lower(): c[k] for k in ("BATCH_SIZE", "NUM_BATCHES", "ROLLOUT_STEPS", "NUM_TO_SAVE")}}
assert os.path.exists(os.path.join(d, "model.safetensors")), "training wrote no model.safetensors"
json.dump(cfg, open(os.path.join(d, "config.json.part"), "w"), indent=1)
os.replace(os.path.join(d, "config.json.part"), os.path.join(d, "config.json"))
EOF
  exit 0
fi

case $algo in
  dr|ref)  f=maze_dr.py;     x="";;
  minimax) f=maze_paired.py; x="--minimax --adv_num_steps 28";;  # goal, dir, agent + 25 wall toggles: the 25-wall box
  rplr)    f=maze_plr.py;    x="";;  # Robust PLR
  plr)     f=maze_plr.py;    x="--exploratory_grad_updates";;
  accel)   f=maze_plr.py;    x="--use_accel";;
  *) echo "unknown algo $algo"; exit 2;;
esac
r=maze_${algo}_n$N; ck=checkpoints/$r/$seed
[ -n "${DRY_RUN:-}" ] || { source "$JAX_VENV/bin/activate"; mkdir -p "$RUNS/maze_train"; cd "$RUNS/maze_train"; }
# finished = a checkpoint at the final step its config.json implies (the step the exam and grader load)
finished() { local s; [ -f "$ck/config.json" ] || return 1
  s=$(PYTHONPATH=$ATLAS python -c 'import json, sys; from atlas.maze.exam import _final_step
print(_final_step(json.load(open(sys.argv[1]))))' "$ck/config.json") && [ -n "$s" ] || return 1
  [ -d "$ck/models/$s" ]; }
if [ -z "${DRY_RUN:-}" ] && finished; then
  echo "already trained: $RUNS/maze_train/$ck"
else
  run rm -rf "$ck" "results/$r/$seed"
  run python "$JAXUED_DIR/examples/$f" --run_name $r --seed $seed --num_updates $N $x ${JAXUED_ARGS:-}
fi
[ -f "results/$r/$seed/results.npz" ] || run python "$JAXUED_DIR/examples/$f" --mode eval --checkpoint_directory $ck
