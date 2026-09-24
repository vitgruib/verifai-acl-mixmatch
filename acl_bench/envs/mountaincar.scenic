# MountainCar task space, sampled by VerifAI through Scenic's external-parameter
# mechanism (see cartpole.scenic); `acl_bench.envs.mountaincar` applies the values.

param force = VerifaiRange(0.0005, 0.002)
param gravity = VerifaiRange(0.00125, 0.005)
param init_range = VerifaiRange(0.05, 0.3)

ego = new Object at (0, 0)
