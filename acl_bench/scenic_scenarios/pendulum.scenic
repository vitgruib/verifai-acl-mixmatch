# Pendulum task space (gravity, mass, length, actuator limits), sampled by
# VerifAI through Scenic. See cartpole.scenic for why there's no simulator
# model attached.

param g = VerifaiRange(5.0, 15.0)
param m = VerifaiRange(0.5, 2.0)
param l = VerifaiRange(0.5, 2.0)
param max_torque = VerifaiRange(1.0, 4.0)
param max_speed = VerifaiRange(4.0, 12.0)

ego = new Object at (0, 0)
