#!/bin/bash
# Separate venv for the SFL repo (Part 1's Maze SFL cell, Part 3 JaxNav train and falsify). SFL pins
# jaxmarl + jaxued against 2024-era jax, so it cannot share Part 1's venv.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
PYBIN=${PYBIN:-python3.11}
$PYBIN -m venv "$SFL_VENV" && source "$SFL_VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -e "$SFL_DIR"
# Upstream builds on the nvcr.io jax container, which already has these; a bare venv does not.
pip install -q distrax wandb safetensors matplotlib tqdm pyyaml
# SFL code is 2024-era: newer jax drops jax.tree_map, newer flax/matplotlib break it.
# Pins verified by a local CPU smoke run of minigrid_sfl and jaxnav_sfl (2026-10-04).
pip install -q "jax[${JAX_EXTRA-cuda12}]==0.4.30" flax==0.8.5 chex==0.1.86 optax==0.2.3 \
  distrax==0.1.5 orbax-checkpoint==0.5.20 tensorflow-probability==0.24.0 "matplotlib<3.10"
# VerifAI for Part 3 falsify (atlas.jaxnav.falsify); local smoke run with these on jax 0.4.30 (2026-10-05).
pip install -q verifai==2.2.0 scenic==3.1.1 dotmap==1.3.30
cd "$ATLAS"; MPLBACKEND=Agg python -c "import atlas.jaxnav.falsify, jax, sfl, jaxmarl, jaxued, distrax, wandb, safetensors; import verifai, scenic; print('ok', jax.__version__, jax.devices())"
