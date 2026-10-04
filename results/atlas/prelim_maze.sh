#!/bin/bash
# Maze falsifier prelim (2026-10-04): which VerifAI sampler finds failures best?
# 8 batch-19 checkpoints (dr/plr/rplr/accel x seeds 0,1; 3000 updates) x spaces dr,seg
# x samplers random,halton,ce,mab,sa x 3 sampler seeds x 1000 levels x 10 attempts.
# Usage: JAXUED_DIR=... PY=.../python bash results/atlas/prelim_maze.sh
cd "$(dirname "$0")/../.."
CK=$JAXUED_DIR/examples/checkpoints
for a in dr plr rplr accel; do for s in 0 1; do for sp in dr seg; do
  echo "$a $s $sp"
done; done; done | xargs -P 4 -L 1 sh -c '$PY -m atlas.maze.falsify --ckpt '"$CK"'/b19_$0_n3000/$1 --algo $0 --space $2 --sampler random halton ce mab sa --seed 0 1 2 --budget 1000 --out runs/maze_prelim 2>&1 | grep -v -i warn'
echo ALLDONE
