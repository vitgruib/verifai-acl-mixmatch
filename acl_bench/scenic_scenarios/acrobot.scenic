# Acrobot task space (link lengths/masses/inertia + torque noise), sampled by
# VerifAI through Scenic. See cartpole.scenic for why there's no simulator
# model attached.

param link_length_1 = VerifaiRange(0.5, 1.5)
param link_length_2 = VerifaiRange(0.5, 1.5)
param link_mass_1 = VerifaiRange(0.5, 1.5)
param link_mass_2 = VerifaiRange(0.5, 1.5)
param link_moi = VerifaiRange(0.5, 1.5)
param torque_noise_max = VerifaiRange(0.0, 0.3)

ego = new Object at (0, 0)
