python -m acl_bench.plr.screen --env cartpole --configs sfl_tilt_soft sfl_tilt_carry --seeds 9-72 \
  --checks 10 --workers 6 --allow-battery --resume --out results/cartpole/lib/batch12.csv
python -m acl_bench.plr.screen --env mountaincar --configs sfl_tilt_soft sfl_tilt_carry --seeds 1-8 \
  --checks 10 --workers 6 --allow-battery --resume --out results/mountaincar/lib/batch12.csv
echo finished
