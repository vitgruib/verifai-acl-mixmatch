#!/bin/bash
# Print the DCD run dir (model.tar, meta.json) of one (algo, seed) cell: regenerates the xpid with
# the same make_cmd call train.sbatch used (trial <seed> -> "-tl_<seed>").
# Usage: dcd_ckpt.sh <car_racing|bipedal> <dr|minimax|plr|rplr|accel|sfl> <seed>
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); source "$here/env.sh"
dom=$1 algo=$2 seed=$3
case $dom in car_racing) p=cr;; bipedal) p=bipedal;; *) echo "unknown domain $dom" >&2; exit 2;; esac
case $algo in rplr) c=${p}_robust_plr;; dr|minimax|plr|accel|sfl) c=${p}_$algo;; *) echo "unknown algo $algo" >&2; exit 2;; esac
xp=$(NSEEDS=$((seed + 1)) bash "$here/dcd_cmds.sh" $dom 0 $c | sed -n "$((seed + 1))p" | grep -o -- '--xpid=[^ ]*' | cut -d= -f2 || true)
[ -n "$xp" ] || { echo "no xpid for $dom $c trial $seed" >&2; exit 1; }
echo "$RUNS/dcd/$dom/$xp"
