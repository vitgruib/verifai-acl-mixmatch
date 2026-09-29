# Point-mass navigation task space, sampled by VerifAI through Scenic's external-parameter
# mechanism (see cartpole.scenic); `acl_bench.envs.pointnav` applies the values.

param gap_y = VerifaiRange(-0.8, 0.8)
param gap_width = VerifaiRange(0.08, 0.8)
param wind = VerifaiRange(-0.6, 0.6)
param force = VerifaiRange(0.4, 1.5)
param goal_y = VerifaiRange(-0.8, 0.8)

ego = new Object at (0, 0)
