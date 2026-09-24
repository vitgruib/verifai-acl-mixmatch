# CartPole task space, sampled by VerifAI through Scenic's external-parameter
# mechanism. No simulator model is attached -- Scenic only needs to sample
# these five scalars; `acl_bench.envs.cartpole` applies them to a real
# CartPoleEnv. verifaiSamplerType / verifaiSamplerParams are set at load time
# from Python (acl_bench/sampling.py) via Scenic's own
# `scenic.scenarioFromFile(..., params={"verifaiSamplerType": ...})` API.

param length = VerifaiRange(0.25, 1.5)
param masspole = VerifaiRange(0.05, 0.5)
param masscart = VerifaiRange(0.5, 2.0)
param force_mag = VerifaiRange(4.0, 16.0)
param init_range = VerifaiRange(0.05, 0.3)

ego = new Object at (0, 0)
