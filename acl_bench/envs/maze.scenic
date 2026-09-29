# Maze task space, sampled by VerifAI through Scenic's external-parameter
# mechanism (see cartpole.scenic); `acl_bench.envs.maze` builds the layout from these.

param n_walls = VerifaiRange(0, 60)
param layout_seed = VerifaiRange(0, 1000000)

ego = new Object at (0, 0)
