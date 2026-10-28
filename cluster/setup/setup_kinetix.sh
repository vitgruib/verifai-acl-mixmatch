#!/bin/bash
# Python env for Kinetix training and falsification. JAX_EXTRA="" installs CPU-only jax.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
PYBIN=${PYBIN:-python3.11}
$PYBIN -m venv "$KINETIX_VENV" && source "$KINETIX_VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -e "$KINETIX_DIR"
pip install -q "jax[${JAX_EXTRA-cuda12}]==0.9.0" flax==0.12.6 optax==0.2.8 chex==0.1.92 distrax==0.1.9 \
  jax2d==1.0.1 jaxgl==1.0.1 hydra-core==1.3.7 omegaconf==2.3.1 wandb==0.18.7
pip install -q verifai==2.2.0 scenic==3.1.1 dotmap==1.3.30
# Kinetix's PLR script imports jaxued; --no-deps keeps the pins above
pip install -q --no-deps -e "$JAXUED_DIR"
pip install -q antlr4-python3-runtime==4.9.3   # hydra needs 4.9; Scenic pulls 4.13
cd "$ATLAS"
MPLBACKEND=Agg SDL_VIDEODRIVER=dummy python -c "import jax, kinetix, jaxued, hydra, verifai, scenic, atlas.kinetix.falsify; print('ok', jax.__version__, jax.devices())"
