#!/bin/bash
# Python env for acl_bench (PyTorch, CPU), which grades the CartPole bridge runs on our exam.
set -euo pipefail
source "$(dirname "$0")/../env.sh"
PYBIN=${PYBIN:-python3.11}
$PYBIN -m venv "$TORCH_VENV" && source "$TORCH_VENV/bin/activate"
pip install -q --upgrade pip
pip install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy==2.4.6 scipy==1.17.1 pandas==2.3.3 gymnasium==1.3.0 verifai==2.2.0 scenic==3.1.1 dotmap==1.3.30 pytest
cd "$ATLAS"; python -c "import atlas.bridge.grade_cartpole, atlas.bridge.compare, acl_bench.suite.run; print('ok')"
