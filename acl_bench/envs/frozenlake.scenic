# FrozenLake task space, sampled by VerifAI through Scenic's external-parameter
# mechanism (see cartpole.scenic); `acl_bench.envs.frozenlake` builds the map from these.

param size = VerifaiRange(4, 10)
param hole_density = VerifaiRange(0.1, 0.3)
param slip = VerifaiRange(0.0, 0.67)
param layout_seed = VerifaiRange(0, 1000000)

ego = new Object at (0, 0)
