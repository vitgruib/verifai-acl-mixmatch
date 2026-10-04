#!/bin/bash
# Python 3.11 venv for Part 1 (JaxUED maze) and all falsification.
# These versions match the local runs. Set JAX_EXTRA=cuda12 (default) or "" for CPU.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
PYBIN=${PYBIN:-python3.11}
JAX_EXTRA=${JAX_EXTRA-cuda12}
$PYBIN -m venv "$JAX_VENV" && source "$JAX_VENV/bin/activate"
pip install -q --upgrade pip
pip install -q "jax[${JAX_EXTRA}]==0.10.2" jaxlib==0.10.2 flax==0.12.8 optax==0.2.8 chex==0.1.92 \
  distrax==0.1.9 gymnax==0.0.9 orbax-checkpoint==0.5.3 wandb==0.30.0 matplotlib numpy==2.4.6
pip install -q -e "$JAXUED_DIR"
pip install -q verifai==2.2.0 scenic==3.1.1 dotmap==1.3.30 pytest
python -c "import jax, jaxued, verifai, scenic; print('ok', jax.__version__, jax.devices())"
