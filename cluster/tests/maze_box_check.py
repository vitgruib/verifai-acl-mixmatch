"""Every Maze method's training levels stay inside the 25-wall box (docs/maze_gpu_audit.md).

ACCEL: 30 generations of 5-edit mutations of 25-wall DR levels, child replacing parent (a buffer
lineage). Upstream's mutator drifts far past 25 walls; jaxued_accel_cap.patch's cap_walls does not.
Minimax: maze_paired.py's adversary (MazeEditor) from the empty level, toggling a fresh cell every
step: 50 steps (upstream default) reach 47 walls, the 28 that train.sbatch passes reach 25.
"""
import os
import sys

import jax
import jax.numpy as jnp

J = os.environ.get("JAXUED_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "jaxued"))
sys.path[:0] = [os.path.join(J, "src"), os.path.join(J, "examples")]
os.environ.setdefault("WANDB_MODE", "disabled")
import maze_plr
from jaxued.environments import Maze
from jaxued.environments.maze import (
    Level,
    make_level_generator,
    make_level_mutator_minimax,
)
from jaxued.environments.maze.env_editor import MazeEditor


def lineage_max_walls(mutate):
    m = jax.jit(jax.vmap(mutate, (0, 0, None)), static_argnums=2)
    lv = jax.vmap(make_level_generator(13, 13, 25))(jax.random.split(jax.random.PRNGKey(0), 256))
    worst = 0
    for g in range(30):
        lv = m(jax.random.split(jax.random.PRNGKey(g + 1), 256), lv, 5)
        worst = max(worst, int(lv.wall_map.sum((1, 2)).max()))
        i = jnp.arange(256)
        assert not bool(lv.wall_map[i, lv.agent_pos[:, 1], lv.agent_pos[:, 0]].any())
        assert not bool(lv.wall_map[i, lv.goal_pos[:, 1], lv.goal_pos[:, 0]].any())
    return worst


def adversary_max_walls(steps):
    ed = MazeEditor(Maze(max_height=13, max_width=13, agent_view_size=5), random_z_dimensions=16)
    empty = Level(wall_map=jnp.zeros((13, 13), bool), goal_pos=jnp.array([0, 0], jnp.uint32),
                  agent_pos=jnp.array([1, 0], jnp.uint32), agent_dir=jnp.array(0, jnp.uint8), width=13, height=13)
    _, st = ed.reset_to_level(jax.random.PRNGKey(0), empty, ed.default_params)
    for t in range(steps):   # goal at cell 7, agent at 8, then toggle cells 16, 17, ...
        _, st, *_ = ed.step(jax.random.PRNGKey(t), st, 13 + t if t >= 3 else (7, 0, 8)[t], ed.default_params)
    return int(st.level.wall_map.sum())


upstream = lineage_max_walls(make_level_mutator_minimax(100))
capped = lineage_max_walls(maze_plr.cap_walls(make_level_mutator_minimax(100), 25))
print(f"ACCEL lineage max walls: upstream {upstream}, cap_walls {capped}")
assert upstream > 25 and capped == 25
a50, a28 = adversary_max_walls(50), adversary_max_walls(28)
print(f"minimax adversary max walls: 50 steps {a50}, 28 steps {a28}")
assert a50 > 25 and a28 == 25
print("maze box check OK")
