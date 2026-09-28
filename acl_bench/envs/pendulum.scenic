# Pendulum task space, sampled by VerifAI through Scenic's external-parameter
# mechanism (see cartpole.scenic); `acl_bench.envs.pendulum` applies the values.

param mass = VerifaiRange(0.5, 2.0)
param length = VerifaiRange(0.5, 1.5)
param gravity = VerifaiRange(5.0, 15.0)
param max_torque = VerifaiRange(0.5, 3.0)
param init_range = VerifaiRange(0.3, 3.141592653589793)

ego = new Object at (0, 0)
