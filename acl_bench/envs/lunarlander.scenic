# LunarLander task space, sampled by VerifAI through Scenic's external-parameter
# mechanism. No simulator model is attached -- Scenic only needs to sample
# these three scalars; `acl_bench.envs.lunarlander` applies them to a real
# Gymnasium LunarLander. verifaiSamplerType / verifaiSamplerParams are set at load time
# from Python (acl_bench/sampling.py) via Scenic's own
# `scenic.scenarioFromFile(..., params={"verifaiSamplerType": ...})` API.

param gravity = VerifaiRange(-12.0, -2.0)
param wind_power = VerifaiRange(0.0, 20.0)
param turbulence_power = VerifaiRange(0.0, 2.0)

ego = new Object at (0, 0)
