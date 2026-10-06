#!/bin/bash
# Venv for Part 2 (Kinetix, FLAIROx, ICLR 2025): training and VerifAI falsify (atlas.kinetix).
# Pins are what the local headless CPU runs used (2026-10-06): all 5 algos train, every checkpoint
# loads and falsifies. On the cluster jax gets the CUDA 12 wheels (JAX_EXTRA= for CPU only).
set -euo pipefail
source "$(dirname "$0")/../env.sh"
PYBIN=${PYBIN:-python3.11}
$PYBIN -m venv "$KINETIX_VENV" && source "$KINETIX_VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -e "$KINETIX_DIR"
pip install -q "jax[${JAX_EXTRA-cuda12}]==0.9.0" flax==0.12.6 optax==0.2.8 chex==0.1.92 distrax==0.1.9 \
  jax2d==1.0.1 jaxgl==1.0.1 hydra-core==1.3.7 omegaconf==2.3.1 wandb==0.18.7
# VerifAI/Scenic pull antlr4 4.13; omegaconf (hydra) needs 4.9, and Scenic only uses antlr for Webots.
pip install -q verifai==2.2.0 scenic==3.1.1 dotmap==1.3.30
pip install -q antlr4-python3-runtime==4.9.3
cd "$ATLAS"
MPLBACKEND=Agg SDL_VIDEODRIVER=dummy python -c "import jax, kinetix, hydra, verifai, scenic, atlas.kinetix.falsify; print('ok', jax.__version__, jax.devices())"
