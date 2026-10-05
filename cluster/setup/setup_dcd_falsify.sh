#!/bin/bash
# py3.11 venv for falsifying DCD checkpoints (Parts 2-3 falsify.sbatch): DCD's torch/gym code plus
# VerifAI/Scenic, which need a newer python than the py3.8 training env. Pins are the exact freeze of
# the local venv the falsifiers were tested in. The DCD repo and openai baselines go on PYTHONPATH
# (baselines' setup.py pins break on 3.11; falsify.sbatch sets the path).
set -euo pipefail
source "$(dirname "$0")/../env.sh"
PYBIN=${PYBIN:-python3.11}
$PYBIN -m venv "$DCDF_VENV" && source "$DCDF_VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -r "$(dirname "$0")/dcd_falsify_requirements.txt"
PYTHONPATH="$ATLAS:$DCD_DIR:$TP/baselines" ACL27_SOFT_RENDER=1 python -c \
  "import verifai, scenic, atlas.carracing.policy, atlas.bipedal.policy; print('ok dcd falsify')"
