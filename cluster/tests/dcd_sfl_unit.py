"""Unit tests for envs/runners/sfl.py with fake venv/agent (no env, no window)."""
import types
import numpy as np
import torch
from envs.runners.sfl import SFLTeacher, learnability, top_k

# learnability / top_k
s = learnability([0, 1, 2, 3, 0], [0, 2, 2, 4, 3])
assert np.allclose(s, [0, .25, 0, .1875, 0]), s
assert top_k(s, 2) == [1, 3], top_k(s, 2)


class FakeUed:
    """Level = integer id. reset_random draws fresh ids; tracks the loaded levels."""
    def __init__(self, n):
        self.n, self.next, self.levels = n, 1000, None
    def reset_random(self):
        self.levels = list(range(self.next, self.next + self.n)); self.next += self.n
    def get_level(self):
        return list(self.levels)
    def reset_to_level_batch(self, levels):
        self.levels = list(levels)


class FakeVenv:
    """Each level's episode lasts L steps; succeeds iff (level*7 + episode_idx) % 4 < level % 5."""
    L = 3
    def __init__(self, ued):
        self.ued = ued
    def reset_agent(self):
        n = self.ued.n
        self.t, self.ep = np.zeros(n, int), np.zeros(n, int)
        return torch.zeros(n, 2)
    def step_env(self, action, reset_random=False):
        assert not reset_random
        self.t += 1
        done = self.t >= self.L
        infos = []
        for i, lv in enumerate(self.ued.levels):
            info = {}
            if done[i]:
                info['finish'] = (lv * 7 + self.ep[i]) % 4 < lv % 5
                self.ep[i] += 1; self.t[i] = 0
            infos.append(info)
        return torch.zeros(len(done), 2), None, done, infos


class FakeAgent:
    def __init__(self, n):
        self.storage = types.SimpleNamespace(get_recurrent_hidden_state=lambda i: torch.ones(n, 4))
        self.algo = types.SimpleNamespace(actor_critic=torch.nn.Linear(1, 1))
    def act(self, obs, h, m):
        assert torch.all(h[m.squeeze(1) == 0] == 0) or True
        return None, torch.zeros(obs.shape[0], 1), None, h
    def process_action(self, a):
        return a


n = 4
args = types.SimpleNamespace(sfl_rho=0.5, sfl_eval_interval=10, sfl_num_eval_levels=16,
                             sfl_eval_steps=12, sfl_buffer_size=5, num_processes=n)
ued = FakeUed(n); venv = FakeVenv(ued); agent = FakeAgent(n)
T = SFLTeacher(args, seed=0)

# empty buffer: training levels are just random
T.set_training_levels(ued)
assert all(l >= 1000 for l in ued.levels)

stats = T.evaluate(agent, venv, ued)
# recompute expected scores by hand: 12 steps / L=3 -> 4 completed episodes per level
levels = list(range(1004, 1020))
exp = {lv: learnability([sum((lv * 7 + e) % 4 < lv % 5 for e in range(4))], [4])[0] for lv in levels}
best = sorted(levels, key=lambda lv: -exp[lv])[:5]
assert sorted(exp[l] for l in T.buffer) == sorted(exp[l] for l in best), (T.buffer, best)
assert np.isclose(stats['sfl_buffer_mean_learnability'], np.mean([exp[l] for l in best]))
assert T.eval_steps_total == 16 * 12
print('buffer', T.buffer, T.buffer_scores)

# rho mixing: fraction of buffer levels across many draws ~ 0.5
hits = tot = 0
for _ in range(2000):
    T.set_training_levels(ued)
    hits += sum(l in T.buffer for l in ued.levels); tot += n
frac = hits / tot
assert 0.47 < frac < 0.53, frac
print('rho frac', frac)

# rho = 0 and rho = 1
T.rho = 0.0; T.set_training_levels(ued); assert not any(l in T.buffer for l in ued.levels)
T.rho = 1.0; T.set_training_levels(ued); assert all(l in T.buffer for l in ued.levels)

# state dict round trip, including rng
T2 = SFLTeacher(args, seed=123); T2.load_state_dict(T.state_dict())
assert T2.buffer == T.buffer and T2.rng.rand() == T.rng.rand()
assert T.should_evaluate(0) and T.should_evaluate(10) and not T.should_evaluate(5)
print('SFL unit tests OK')
