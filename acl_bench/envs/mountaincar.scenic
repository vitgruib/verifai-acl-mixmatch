# MountainCar task space, sampled by VerifAI through Scenic's external-parameter
# mechanism (see cartpole.scenic); `acl_bench.envs.mountaincar` applies the values.

param force = VerifaiRange(0.001, 0.004)
param gravity = VerifaiRange(0.0015, 0.0035)
param init_range = VerifaiRange(0.05, 0.3)

ego = new Object at (0, 0)
