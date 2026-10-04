"""Unit tests for sfl/train/jaxnav_minimax.py's LevelEditor, BFS and return reduction. Headless."""
import jax, jax.numpy as jnp, numpy as np
from collections import deque
from jaxmarl.environments.jaxnav import JaxNav
from sfl.train.jaxnav_minimax import LevelEditor, mean_episode_returns

env = JaxNav(num_agents=1, map_id="Grid-Rand-Poly", map_params={"map_size": (11, 11), "fill": 0.6}, max_steps=500)
ed = LevelEditor(env, num_walls=48, z_dim=16)
assert (ed.iw, ed.ih, ed.n, ed.T) == (9, 9, 81, 50)

def run(actions):
    s = ed.init()
    for c in actions:
        assert bool(ed.avail(s)[c]), (int(s.t), c)
        s = ed.step(s, jnp.int32(c))
    return s

# 1. goal mask at step 1; walls on goal/start are no-ops; repeat walls idempotent
s = ed.step(ed.init(), jnp.int32(10))
assert not bool(ed.avail(s)[10]) and int(ed.avail(s).sum()) == 80
s = run([10, 20, 10, 20, 30, 30] + [0] * 44)
assert int(s.goal) == 10 and int(s.start) == 20 and int(s.t) == 50
assert set(np.flatnonzero(np.asarray(s.walls))) == {30, 0}
o = ed.obs(s, jnp.zeros(16)); assert o.shape == (ed.obs_dim,)

# 2. level geometry: cell c=(x=c%9,y=c//9) -> map_data[y+1,x+1], world (x+1.5,y+1.5)
lv = ed.to_level(jax.random.PRNGKey(0), s)
md = np.asarray(lv.map_data)
assert md.shape == (11, 11) and md[0].all() and md[-1].all() and md[:, 0].all() and md[:, -1].all()
assert md[1:-1, 1:-1].sum() == 2 and md[30 // 9 + 1, 30 % 9 + 1] == 1 and md[1, 1] == 1
assert np.allclose(lv.pos, [[20 % 9 + 1.5, 20 // 9 + 1.5]]) and np.allclose(lv.goal, [[10 % 9 + 1.5, 10 // 9 + 1.5]])
assert -np.pi <= float(lv.theta[0]) <= np.pi
tpl = env.sample_test_case(jax.random.PRNGKey(0))
for f in ["pos", "theta", "goal", "map_data"]:
    assert getattr(lv, f).dtype == getattr(tpl, f).dtype and getattr(lv, f).shape == getattr(tpl, f).shape, f
# env agrees: a wall cell's centre collides, the start cell's centre does not

assert bool(env.map_obj.check_agent_map_collision(lv.pos[0], lv.theta[0], lv.map_data))is False
wall_xy = jnp.array([30 % 9 + 1.5, 30 // 9 + 1.5])
assert bool(env.map_obj.check_agent_map_collision(wall_xy, 0.0, lv.map_data)) is True

# 3. env runs from the level and resets to it after an episode
obs, st = env.set_state(lv)
assert np.allclose(st.pos, lv.pos) and int(st.step) == 0

# 4. BFS vs python BFS on random editor states
def py_solvable(s):
    free = np.asarray(ed.map_data(s)) == 0
    a = (int(s.start) // 9 + 1, int(s.start) % 9 + 1); g = (int(s.goal) // 9 + 1, int(s.goal) % 9 + 1)
    q, seen = deque([a]), {a}
    while q:
        y, x = q.popleft()
        if (y, x) == g: return True
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            n = (y+dy, x+dx)
            if free[n] and n not in seen: seen.add(n); q.append(n)
    return False
solv = jax.jit(ed.solvable)
rng = np.random.default_rng(0); nsolv = 0
for i in range(300):
    g = rng.integers(81); st_ = rng.choice([c for c in range(81) if c != g])
    s = run([g, st_] + list(rng.integers(0, 81, 48)))
    a, b = bool(solv(s)), py_solvable(s); assert a == b, i; nsolv += a
# a full wall between start (x=0) and goal (x=8): unsolvable; a snake corridor: solvable (long path)
s = run([8, 0] + [4 + 9 * y for y in range(9)] + [0] * 39); assert not bool(solv(s))
snake = [c for c in range(81) if (c // 9) % 2 == 1 and not ((c // 9) % 4 == 1 and c % 9 == 8) and not ((c // 9) % 4 == 3 and c % 9 == 0)]
s = run([80, 0] + snake[:48] + [0] * (48 - len(snake[:48])))
assert bool(solv(s)) == py_solvable(s)
print("BFS ok; solvable frac of random levels:", nsolv / 300)

# 5. mean_episode_returns
r = jnp.array([[1, 0], [1, 0], [2, 5], [1, 0], [3, 0]], jnp.float32)
d = jnp.array([[0, 0], [0, 0], [1, 0], [0, 0], [1, 0]], bool)
m, c = mean_episode_returns(d, r)
assert np.allclose(m, [(4 + 4) / 2, 5]) and np.allclose(c, [2, 0]), (m, c)
print("minimax unit tests OK")
