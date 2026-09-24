"""Scenic-mediated VerifAI sampling: the task picker. It goes through Scenic on purpose (not
a direct `verifai.samplers.FeatureSampler` call), using Scenic's own documented
external-sampler API -- `scenic.scenarioFromFile(..., params={"verifaiSamplerType":
...})` and `Scenario.generate(feedback=...)` -- with no simulator model attached
(see acl_bench/envs/cartpole.scenic, and the module docstring of
`scenic.core.external_params` for how a plain `param x = VerifaiRange(...)`
gets resolved without any spatial simulator).

`verifai.server.choose_sampler` -- the function Scenic's `VerifaiSampler`
calls under the hood for every `verifaiSamplerType` string -- determines
which samplers are reachable by name. Checked directly against the
installed VerifAI release:

  - random, halton, ce (cross-entropy), mab (multi-armed bandit) work
    end-to-end through Scenic by name. halton is not used here: it ignores
    feedback like random, and the study needs only one non-adaptive baseline.
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

# random ignores the feedback it is given; ce/mab/sa steer by it.
ADAPTIVE_SAMPLERS = ("ce", "mab", "sa")
SAMPLER_NAMES = ("random",) + ADAPTIVE_SAMPLERS

# Hyperparameters per adaptive sampler (see docs/ablation.md for sources).
# ce: `buckets` per dimension; `alpha` = retention -- each counterexample (rho < thres)
#     moves a bucket distribution (1 - alpha) toward the sampled bucket; `thres` =
#     counterexample cutoff on rho. mab: UCB over buckets; a bucket is charged an
#     "error" when rho < thres (its `alpha` is stored but unused by VerifAI).
# sa: `T` initial temperature, `decay_rate` proposal-width decay per step within an
#     epoch, `iterations` steps per epoch before re-heating.
DEFAULT_SAMPLER_PARAMS = {
    "ce": {"buckets": 5, "alpha": 0.9, "thres": 0.0},      # VerifAI's own defaults
    "mab": {"buckets": 5, "thres": 0.0},                    # (alpha is unused by mab)
    "sa": {"T": 1.0, "decay_rate": 0.9, "iterations": 20},   # VerifAI has no defaults for SA: these are ours
}

# num_epoch is how many temperature-reset rounds SA runs before raising
# TerminationException; set high enough that a training run never hits it.
_SA_NUM_EPOCH = 10 ** 6


class SimulatedAnnealingSampler(VerifaiSampler):
    """VerifAI's simulated-annealing sampler, injected via Scenic's
    `externalSampler` global parameter since `verifaiSamplerType` can't
    select it. The base class builds a throwaway random sampler over the same
    FeatureSpace (that's where the Scenic-declared VerifaiRange parameters get
    registered); we then swap in SA over that space."""

    sa_params = dict(DEFAULT_SAMPLER_PARAMS["sa"])

    def __init__(self, params, globalParams):
        super().__init__(params, globalParams)
        self.sampler = FeatureSampler.simulatedAnnealingSamplerFor(
            self.sampler.space, DotMap(**self.sa_params, num_epoch=_SA_NUM_EPOCH))


def _scenario_params(sampler_name: str, sampler_params: dict) -> dict:
    if sampler_name == "sa":
        configured = type("ConfiguredSA", (SimulatedAnnealingSampler,), {"sa_params": sampler_params})
        return {"verifaiSamplerType": "random", "externalSampler": configured}
    params = {"verifaiSamplerType": sampler_name}
    if sampler_name in ("ce", "mab"):
        params["verifaiSamplerParams"] = DotMap(
            alpha=sampler_params.get("alpha", 0.9), thres=sampler_params["thres"],
            cont=DotMap(buckets=sampler_params["buckets"], dist=None), disc=DotMap(dist=None))
    return params


@dataclass
class ScenicTaskSampler:
    """Wraps one Scenic Scenario + the running feedback loop VerifAI expects:
    `generate(feedback=rho_from_previous_draw)` on every call. `rho` is
    supplied by the caller (acl_bench.acl) at report time and
    only takes effect on the *next* draw -- matching VerifAI's own
    falsification loop (propose, evaluate, feed back, propose again)."""

    scenario: object
    pending_feedback: float | None = None

    @classmethod
    def load(cls, sampler_name: str, scenic_file: str) -> "ScenicTaskSampler":
        if sampler_name not in SAMPLER_NAMES:
            raise ValueError(f"unknown sampler {sampler_name!r}; choose from {SAMPLER_NAMES}")
        scenario = scenic.scenarioFromFile(
            scenic_file, params=_scenario_params(sampler_name, DEFAULT_SAMPLER_PARAMS.get(sampler_name, {})),
            mode2D=True)
        return cls(scenario=scenario)

    def draw(self, param_names: tuple[str, ...]) -> dict:
        scene, _info = self.scenario.generate(feedback=self.pending_feedback)
        self.pending_feedback = None
        return {name: scene.params[name] for name in param_names}

    def give_feedback(self, rho: float) -> None:
        """Queue rho to be delivered on the *next* draw() call."""
        self.pending_feedback = float(rho)
