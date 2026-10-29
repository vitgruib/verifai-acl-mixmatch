#!/bin/bash
# Separate Python env for the SFL repo (Maze SFL, JaxNav), which needs an older jax.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
PYBIN=${PYBIN:-python3.11}
$PYBIN -m venv "$SFL_VENV" && source "$SFL_VENV/bin/activate"
pip install -q --upgrade pip
# SFL's requirements.txt leaves jaxmarl and jaxued unpinned: jaxmarl at a fixed commit, jaxued at the
# one the JaxUED runs use (unpatched)
pip install -q --no-deps -e "$SFL_DIR"
pip install -q pandas hydra-core "jaxmarl @ git+$JAXMARL_REPO@$JAXMARL_COMMIT" "jaxued @ git+$JAXUED_REPO@$JAXUED_COMMIT"
pip install -q distrax wandb safetensors matplotlib tqdm pyyaml
pip install -q "jax[${JAX_EXTRA-cuda12}]==0.4.30" flax==0.8.5 chex==0.1.86 optax==0.2.3 \
  distrax==0.1.5 orbax-checkpoint==0.5.20 tensorflow-probability==0.24.0 "matplotlib<3.10" gymnax==0.0.9
pip install -q verifai==2.2.0 scenic==3.1.1 dotmap==1.3.30
pip install -q antlr4-python3-runtime==4.9.3   # hydra needs 4.9; Scenic pulls 4.13
cd "$ATLAS"; MPLBACKEND=Agg python -c "import atlas.jaxnav.falsify, jax, hydra, omegaconf, sfl, jaxmarl, jaxued, distrax, wandb, safetensors; import verifai, scenic; omegaconf.OmegaConf.create({'a': '\${b}', 'b': 1}).a; print('ok', jax.__version__, jax.devices())"
python -c "import importlib.metadata as m, sys
for p, c in zip(sys.argv[1::2], sys.argv[2::2]):
    u = m.distribution(p).read_text('direct_url.json'); assert c in u, p + ' is not at ' + c + ': ' + u" \
  jaxued "$JAXUED_COMMIT" jaxmarl "$JAXMARL_COMMIT"
