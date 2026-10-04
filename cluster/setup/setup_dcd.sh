#!/bin/bash
# Conda env for Parts 2-3 (DCD CarRacing / BipedalWalker), exactly as DCD's README says.
# Needs: conda, and xvfb (xvfb-run) on compute nodes for CarRacing.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
eval "$(conda shell.bash hook)"
conda create -y -q --name "$DCD_CONDA" python=3.8
conda activate "$DCD_CONDA"
pip install -q -r "$DCD_DIR/requirements.txt"
pip install -q -e "$TP/baselines"
pip install -q pyglet==1.5.11
python -c "import torch, gym; print('ok torch', torch.__version__, 'cuda', torch.cuda.is_available())"
command -v xvfb-run >/dev/null || echo "WARNING: xvfb-run not found; CarRacing (Part 2) needs it"
