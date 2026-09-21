"""Named experimental arms for the CartPole suite: the four primary arms (the 2x2
component ablation) and the hyperparameter ablations around them.

The plan is the eight "primary" arms only (`MAIN_ARMS`): does ACL help, does each
VerifAI sampler help, and do both help, with every hyperparameter fixed at a standard
value. The hyperparameter variants below (families "acl", "sampler", "coupling") are
PARKED: defined and tested, not part of the plan (docs/ablation.md). Comparisons are
between independent groups of runs (docs/cartpole_suite.md, "Comparing methods").
"""
from __future__ import annotations

from dataclasses import dataclass, field

from acl_bench.scenic_sampling import ADAPTIVE_SAMPLERS

# Provisional: Stage 0 of the suite sets the real budget from baseline convergence.
DEFAULT_BUDGET = 400_000
CHECKPOINT_EVERY = 20 * 1024          # a multiple of the 1024-step rollout, so checkpoints are exact
POTENTIAL_FN = "pvl_gae"              # SIPACL's own score; the scoring-function axis is a separate study


@dataclass(frozen=True)
class Arm:
    name: str
    sampler: str
    acl: bool
    potential_fn: str
    ppo: dict = field(default_factory=dict)             # PPOConfig overrides
    sampler_params: dict = field(default_factory=dict)  # scenic_sampling.DEFAULT_SAMPLER_PARAMS overrides
    family: str = "primary"
    reference: str | None = None                        # arm this variant is compared against
    changes: str = ""                                    # human-readable delta from the reference


def _primary(adaptive: str) -> list[Arm]:
    return [
        Arm(f"S_{adaptive}", adaptive, False, POTENTIAL_FN, family="primary"),
        Arm(f"B_{adaptive}", adaptive, True, POTENTIAL_FN, family="primary"),
    ]


def build_arms() -> dict[str, Arm]:
    arms: list[Arm] = [
        # N and A do not depend on the adaptive sampler.
        Arm("N", "random", False, "neg_return", family="primary"),   # score has no consumer here
        Arm("A", "random", True, POTENTIAL_FN, family="primary"),
    ]
    for adaptive in ADAPTIVE_SAMPLERS:
        arms += _primary(adaptive)

    # --- ACL hyperparameters, against A (ACL on, random sampler, pvl_gae) ---
    # replay_prob x rank_alpha jointly set selection pressure, so they are crossed.
    for rp in (0.25, 0.5, 0.75):
        for ra in (0.5, 1.0, 2.0):
            if (rp, ra) == (0.5, 1.0):
                continue
            arms.append(Arm(f"A_rp{rp}_ra{ra}", "random", True, POTENTIAL_FN,
                            ppo={"replay_prob": rp, "rank_alpha": ra}, family="acl", reference="A",
                            changes=f"replay_prob={rp}, rank_alpha={ra}"))
    for beta in (0.1, 0.5):
        arms.append(Arm(f"A_beta{beta}", "random", True, POTENTIAL_FN, ppo={"ema_beta": beta},
                        family="acl", reference="A", changes=f"ema_beta={beta}"))
    for buf in (100, 500):
        arms.append(Arm(f"A_buf{buf}", "random", True, POTENTIAL_FN, ppo={"buffer_max": buf},
                        family="acl", reference="A", changes=f"buffer_max={buf}"))

    # --- sampler hyperparameters, against S_<sampler> (adaptive sampler, ACL off) ---
    sampler_grid = {
        "ce": {"buckets": (3, 10), "alpha": (0.5, 0.98), "thres": (-0.33, 0.33)},
        "mab": {"buckets": (3, 10), "thres": (-0.33, 0.33)},
        "sa": {"T": (0.3, 3.0), "decay_rate": (0.8, 0.95), "iterations": (10, 40)},
    }
    for sampler, grid in sampler_grid.items():
        for key, values in grid.items():
            for v in values:
                arms.append(Arm(f"S_{sampler}_{key}{v}", sampler, False, POTENTIAL_FN,
                                sampler_params={key: v}, family="sampler", reference=f"S_{sampler}",
                                changes=f"{sampler}.{key}={v}"))

    # --- coupling: does the sampler steer by the same score that ranks replay? ---
    # Default (shared) is B_<sampler>. Decoupled: replay ranks by pvl_gae, the
    # sampler steers by neg_return -- closer to SIPACL, whose Scenic feedback is separate.
    for sampler in ADAPTIVE_SAMPLERS:
        arms.append(Arm(f"B_{sampler}_decoupled", sampler, True, POTENTIAL_FN,
                        ppo={"feedback_fn": "neg_return"}, family="coupling", reference=f"B_{sampler}",
                        changes="sampler feedback = neg_return (replay still pvl_gae)"))

    by_name = {a.name: a for a in arms}
    assert len(by_name) == len(arms), "duplicate arm names"
    for a in arms:
        assert a.reference is None or a.reference in by_name, f"{a.name}: unknown reference {a.reference}"
    return by_name


ARMS = build_arms()


def family(name: str) -> list[Arm]:
    return [a for a in ARMS.values() if a.family == name]


MAIN_ARMS: tuple[str, ...] = tuple(a.name for a in family("primary"))
RUNS_PER_ARM = 100


def comparisons() -> list[tuple[str, str]]:
    """(variant, reference) pairs to analyse, including the primary contrasts."""
    pairs = [(a.name, a.reference) for a in ARMS.values() if a.reference]
    for sampler in ADAPTIVE_SAMPLERS:
        pairs += [("A", "N"), (f"S_{sampler}", "N"), (f"B_{sampler}", "N"),
                  (f"B_{sampler}", f"S_{sampler}"), (f"B_{sampler}", "A")]
    return list(dict.fromkeys(pairs))
