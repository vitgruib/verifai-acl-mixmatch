#!/bin/zsh
cd /Users/ethancai/Projects/Ongoing/ACL27
A=(fmodel sir_a1 sir_a4 sir_is sir_is05 replay_post mutate_post verifai_surr)
for e in cartpole acrobot; do
  python -m acl_bench.plr.screen --env $e --configs $A --seeds 1-8 --checks 10 --workers 5 \
    --allow-battery --resume --out results/$e/lib/batch1.csv
done
