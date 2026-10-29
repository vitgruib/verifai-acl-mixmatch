#!/bin/bash
# JaxUED and SFL at pinned commits (patched) into third_party/, and the three Python envs:
# .venv-jax (JaxUED, exam, grading), .venv-sfl (SFL needs an older jax), .venv-torch (CartPole check).
set -euo pipefail
cd "$(dirname "$0")/.."
bash cluster/setup/clone.sh
bash cluster/setup/setup_jax.sh
bash cluster/setup/setup_sfl.sh
bash cluster/setup/setup_torch.sh
