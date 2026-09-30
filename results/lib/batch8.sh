for e in cartpole acrobot; do
  python -m acl_bench.plr.screen --env $e --configs sfl sfl_p25 --seeds 1-8 \
    --checks 10 --workers 5 --allow-battery --resume --out results/$e/lib/batch8.csv
done
