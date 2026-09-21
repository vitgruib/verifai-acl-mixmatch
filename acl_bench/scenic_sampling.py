"""Scenic-mediated VerifAI sampling: this is the actual "verifAI sampling"
half of the mix-and-match grid, and it goes through Scenic on purpose (not
a direct `verifai.samplers.FeatureSampler` call) to mirror how SIPACL itself
drives VerifAI -- `scenic.scenarioFromFile(..., params={"verifaiSamplerType":
...})` -- just without a MetaDrive/CARLA model attached (see the .scenic
files in acl_bench/scenic_scenarios/, and the module docstring of
`scenic.core.external_params` for how a plain `param x = VerifaiRange(...)`
gets resolved without any spatial simulator).

`verifai.server.choose_sampler` -- the function Scenic's `VerifaiSampler`
calls under the hood for every `verifaiSamplerType` string -- determines
which samplers are reachable by name. Checked directly against the
installed VerifAI release:

  - random, halton, ce (cross-entropy), mab (multi-armed bandit) work
    end-to-end through Scenic by name.
  - sa (simulated annealing) works when called directly through
    `FeatureSampler.simulatedAnnealingSamplerFor` but has no branch in
    `choose_sampler`, so `verifaiSamplerType='sa'` can't select it. It is
    registered here through Scenic's documented `externalSampler` global
    parameter instead (`SimulatedAnnealingSampler` below), so it still goes
    through the same `scenario.generate(feedback=...)` loop as the others.
  - bo (Bayesian optimization) also works, but was dropped: its GP refit
    grew with the run's task history and accounted for 83% of the first full
    grid's 131 minutes of compute (mean 227s per CartPole run vs ~5s for
    every other sampler) while adding two fragile dependencies (GPyOpt +
    GPy, and a `setuptools<81` pin for the removed `pkg_resources`). Replaced
    by sa, which costs about the same as random.
  - eg (epsilon-greedy) raises `NotImplementedError: tried to use abstract
    BoxSampler` even called directly through `FeatureSampler` -- broken in
    this VerifAI release. Excluded.
  - grid terminates after exhaustively covering its (default) resolution
    and took 58ms/sample even before that on a 5D box in a timing test --
    built for one-shot exhaustive coverage, not an open-ended training
    loop. Excluded as the wrong tool for this job, not because it's broken.
"""
from __future__ import annotations

from dataclasses import dataclass

import scenic
from dotmap import DotMap
from scenic.core.external_params import VerifaiSampler
from verifai.samplers.feature_sampler import FeatureSampler

SAMPLER_NAMES = ("random", "halton", "ce", "mab", "sa")

_ADAPTIVE_DEFAULTS = DotMap(alpha=0.9, thres=0.0, cont=DotMap(buckets=8, dist=None),
                             disc=DotMap(dist=None))

# num_epoch is how many temperature-reset rounds SA runs before raising
# TerminationException; set high enough that a training run never hits it.
_SA_PARAMS = DotMap(T=1.0, decay_rate=0.9, iterations=20, num_epoch=10**6)


class SimulatedAnnealingSampler(VerifaiSampler):
    """VerifAI's simulated-annealing sampler, injected via Scenic's
    `externalSampler` global parameter since `verifaiSamplerType` can't
    select it. The base class builds a throwaway random sampler over the same
    FeatureSpace (that's where the Scenic-declared VerifaiRange parameters get
    registered); we then swap in SA over that space."""

    def __init__(self, params, globalParams):
        super().__init__(params, globalParams)
        self.sampler = FeatureSampler.simulatedAnnealingSamplerFor(
            self.sampler.space, _SA_PARAMS.copy())


def _scenario_params(sampler_name: str) -> dict:
    if sampler_name == "sa":
        return {"verifaiSamplerType": "random", "externalSampler": SimulatedAnnealingSampler}
    params = {"verifaiSamplerType": sampler_name}
    if sampler_name in ("ce", "mab"):
        params["verifaiSamplerParams"] = _ADAPTIVE_DEFAULTS.copy()
    return params


@dataclass
class ScenicTaskSampler:
    """Wraps one Scenic Scenario + the running feedback loop VerifAI expects:
    `generate(feedback=rho_from_previous_draw)` on every call. `rho` is
    supplied by the caller (acl_bench.curriculum.plr) at report time and
    only takes effect on the *next* draw -- matching VerifAI's own
    falsification loop (propose, evaluate, feed back, propose again)."""

    scenario: object
    pending_feedback: float | None = None

    @classmethod
    def load(cls, scenic_file: str, sampler_name: str) -> "ScenicTaskSampler":
        if sampler_name not in SAMPLER_NAMES:
            raise ValueError(f"unknown sampler {sampler_name!r}; choose from {SAMPLER_NAMES}")
        scenario = scenic.scenarioFromFile(
            scenic_file, params=_scenario_params(sampler_name), mode2D=True)
        return cls(scenario=scenario)

    def draw(self, param_names: tuple[str, ...]) -> dict:
        scene, _info = self.scenario.generate(feedback=self.pending_feedback)
        self.pending_feedback = None
        return {name: scene.params[name] for name in param_names}

    def give_feedback(self, rho: float) -> None:
        """Queue rho to be delivered on the *next* draw() call."""
        self.pending_feedback = float(rho)
