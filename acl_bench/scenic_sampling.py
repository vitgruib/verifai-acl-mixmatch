"""Scenic-mediated VerifAI sampling: this is the actual "verifAI sampling"
half of the mix-and-match grid, and it goes through Scenic on purpose (not
a direct `verifai.samplers.FeatureSampler` call) to mirror how SIPACL itself
drives VerifAI -- `scenic.scenarioFromFile(..., params={"verifaiSamplerType":
...})` -- just without a MetaDrive/CARLA model attached (see the .scenic
files in acl_bench/scenic_scenarios/, and the module docstring of
`verifai.core.external_params` for how a plain `param x = VerifaiRange(...)`
gets resolved without any spatial simulator).

`verifai.server.choose_sampler` -- the function Scenic's `VerifaiSampler`
calls under the hood for every `verifaiSamplerType` string -- was used to
determine which samplers are actually reachable this way. Checked directly
against the installed VerifAI release:

  - random, halton, ce (cross-entropy), mab (multi-armed bandit), bo
    (Bayesian optimization) all work end-to-end through Scenic.
  - eg (epsilon-greedy) raises `NotImplementedError: tried to use abstract
    BoxSampler` even called directly through `FeatureSampler` -- broken in
    this VerifAI release, not a Scenic issue. Excluded.
  - grid terminates after exhaustively covering its (default) resolution
    and took 58ms/sample even before that on a 5D box in a timing test --
    built for one-shot exhaustive coverage, not an open-ended training
    loop. Excluded as the wrong tool for this job, not because it's broken.
  - simulated annealing works directly via `FeatureSampler
    .simulatedAnnealingSamplerFor`, but has no branch in
    `verifai.server.choose_sampler` at all, so Scenic's `verifaiSamplerType`
    can never select it. Excluded so every sampler in the grid goes through
    the same Scenic-mediated code path.

Bayesian optimization additionally required installing `GPyOpt` + `GPy` and
pinning `setuptools<81` (GPyOpt still imports the now-removed
`pkg_resources`); see requirements.txt.
"""
from __future__ import annotations

from dataclasses import dataclass

import scenic
from dotmap import DotMap

SAMPLER_NAMES = ("random", "halton", "ce", "mab", "bo")

_ADAPTIVE_DEFAULTS = DotMap(alpha=0.9, thres=0.0, cont=DotMap(buckets=8, dist=None),
                             disc=DotMap(dist=None))


def _sampler_params(sampler_name: str):
    if sampler_name in ("ce", "mab"):
        return _ADAPTIVE_DEFAULTS.copy()
    if sampler_name == "bo":
        return DotMap(init_num=5)
    return None  # random / halton use their own defaults


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
        params = {"verifaiSamplerType": sampler_name}
        sp = _sampler_params(sampler_name)
        if sp is not None:
            params["verifaiSamplerParams"] = sp
        scenario = scenic.scenarioFromFile(scenic_file, params=params, mode2D=True)
        return cls(scenario=scenario)

    def draw(self, param_names: tuple[str, ...]) -> dict:
        scene, _info = self.scenario.generate(feedback=self.pending_feedback)
        self.pending_feedback = None
        return {name: scene.params[name] for name in param_names}

    def give_feedback(self, rho: float) -> None:
        """Queue rho to be delivered on the *next* draw() call."""
        self.pending_feedback = float(rho)
