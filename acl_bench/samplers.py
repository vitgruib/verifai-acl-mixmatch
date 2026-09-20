"""VerifAI-backed samplers over the CartPole task space.

VerifAI (github.com/BerkeleyLearnVerify/VerifAI, `pip install verifai`) was
built to drive falsification search over Scenic scenarios: a `FeatureSpace`
describes the parameters of a scene, a `FeatureSampler` proposes points in
that space, the scenario is run, and a scalar robustness value `rho` (STL
semantics: rho < 0 means a specification was violated) is fed back via
`sampler.update(sample, info, rho)` so the sampler can steer future proposals
toward interesting/failing regions.

We reuse that exact API for CartPole physics configurations instead of
Scenic scenes: `rho` is the (shifted) episode return, so "counterexample"
becomes "the policy still fails on this configuration" -- which is precisely
what an ACL curriculum wants to keep sampling. This file only wires up the
FeatureSpace and exposes the three sampler types under test; SIPACL's own
usage of VerifAI is via `params={"verifaiSamplerType": ...}` passed into
`scenic.scenarioFromFile`, which resolves to the same `FeatureSampler.*For`
factories used here.
"""
from __future__ import annotations

from verifai.features.features import Box, Feature, FeatureSpace
from verifai.samplers.feature_sampler import FeatureSampler

from acl_bench.envs.param_cartpole import CartPoleParams

SAMPLER_NAMES = ("random", "halton", "mab")


def make_feature_space() -> FeatureSpace:
    bounds = CartPoleParams.bounds()
    return FeatureSpace({
        name: Feature(Box(bound)) for name, bound in bounds.items()
    })


def make_sampler(name: str, space: FeatureSpace | None = None) -> FeatureSampler:
    space = space or make_feature_space()
    if name == "random":
        return FeatureSampler.randomSamplerFor(space)
    if name == "halton":
        return FeatureSampler.haltonSamplerFor(space)
    if name == "mab":
        # Discretize each continuous dim into 8 buckets and run UCB1 over them
        # (verifai.samplers.multi_armed_bandit.ContinuousMultiArmedBanditSampler).
        from dotmap import DotMap
        mab_params = DotMap(
            alpha=0.9,
            thres=0.0,  # rho < thres counts as a "counterexample" bucket hit
            cont=DotMap(buckets=8, dist=None),
            disc=DotMap(dist=None),
        )
        return FeatureSampler.multiArmedBanditSamplerFor(space, mab_params)
    raise ValueError(f"unknown sampler {name!r}; choose from {SAMPLER_NAMES}")


def point_to_params(point) -> CartPoleParams:
    return CartPoleParams(
        length=float(point.length[0]),
        masspole=float(point.masspole[0]),
        masscart=float(point.masscart[0]),
        force_mag=float(point.force_mag[0]),
        init_range=float(point.init_range[0]),
    )
