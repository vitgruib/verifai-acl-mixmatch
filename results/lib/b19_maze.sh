#!/bin/bash
# Batch 19: JaxUED maze reproduction at 10% budget (headless). Usage: b19_maze.sh [num_updates]
S=/private/tmp/claude-501/-Users-ethancai-Projects-Ongoing-ACL27/38c0405f-ead9-457e-a6da-6f51491463fc/scratchpad
N=${1:-3000}
cd $S/jaxued/examples && source $S/jvenv/bin/activate
export WANDB_MODE=offline WANDB_SILENT=true MPLBACKEND=Agg
one() {  # arm seed
  case $1 in
    dr) f=maze_dr.py; x="";; paired) f=maze_paired.py; x="";;
    rplr) f=maze_plr.py; x="";; plr) f=maze_plr.py; x="--exploratory_grad_updates";;
    accel) f=maze_plr.py; x="--use_accel";;
  esac
  r=b19_$1_n$N
  python $f --run_name $r --seed $2 --num_updates $N $x > logs/${r}_$2.log 2>&1 \
   && python $f --mode eval --checkpoint_directory checkpoints/$r/$2 >> logs/${r}_$2.log 2>&1
  echo "done $1 $2 rc=$?"
}
export -f one; export N; mkdir -p logs
for s in 0 1; do for a in rplr accel plr dr paired; do echo "$a $s"; done; done \
 | xargs -P 2 -L 1 bash -c 'one $0 $1'
echo finished-b19
