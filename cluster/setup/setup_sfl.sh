#!/bin/bash
# Separate venv for Part 4 (SFL). SFL pins jaxmarl@latest + jaxued, which may need a
# different jax than Part 1, so it gets its own venv. XLand needs a third one (see xland/).
set -euo pipefail
source "$(dirname "$0")/../env.sh"
PYBIN=${PYBIN:-python3.11}
SFL_VENV=${SFL_VENV:-$ATLAS/.venv-sfl}
$PYBIN -m venv "$SFL_VENV" && source "$SFL_VENV/bin/activate"
pip install -q --upgrade pip
pip install -q "jax[${JAX_EXTRA-cuda12}]"
pip install -q -e "$SFL_DIR"
python -c "import jax, sfl, jaxmarl, jaxued; print('ok', jax.__version__, jax.devices())"
