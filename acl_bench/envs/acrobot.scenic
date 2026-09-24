# Acrobot task space, sampled by VerifAI through Scenic's external-parameter
# mechanism (see cartpole.scenic); `acl_bench.envs.acrobot` applies the values.

param link_length_1 = VerifaiRange(0.5, 1.5)
param link_mass_1 = VerifaiRange(0.5, 2.0)
param link_mass_2 = VerifaiRange(0.5, 2.0)
param torque = VerifaiRange(0.5, 2.0)
param init_range = VerifaiRange(0.05, 0.3)

ego = new Object at (0, 0)
