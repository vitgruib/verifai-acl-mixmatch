import dataclasses

from acl_bench.ppo import PPOConfig
from acl_bench.scenic_sampling import resolve_sampler_params
from acl_bench.suite.ablation import ARMS, comparisons, family


def test_every_arm_override_is_valid():
    fields = {f.name for f in dataclasses.fields(PPOConfig)}
    for arm in ARMS.values():
        assert set(arm.ppo) <= fields, arm.name
        resolve_sampler_params(arm.sampler, arm.sampler_params)      # raises on unknown keys
        assert arm.reference is None or arm.reference in ARMS


def test_registry_shape_and_variants_actually_differ_from_their_reference():
    assert {a.family for a in ARMS.values()} == {"primary", "acl", "sampler", "coupling"}
    assert len(family("acl")) == 8 + 2 + 2
    for arm in ARMS.values():
        if arm.reference:
            ref = ARMS[arm.reference]
            assert (arm.ppo, arm.sampler_params, arm.sampler, arm.acl) != (ref.ppo, ref.sampler_params, ref.sampler, ref.acl), arm.name
    assert len(comparisons()) == len(set(comparisons()))
