# Tests for the ported cells

Headless (no windows). Run after `cluster/setup/clone.sh` has applied the patches.

| Test | Covers | Run from | Command |
| --- | --- | --- | --- |
| `dcd_sfl_unit.py` | SFL learnability, top-K buffer, rho mixing, checkpoint state (fake env) | `third_party/dcd` | `python ../../cluster/tests/dcd_sfl_unit.py` |
| `dcd_sfl_env.py` | `info['finish']` reaches SFL through DCD's CarRacing/Bipedal vec envs | `third_party/dcd` | `python ../../cluster/tests/dcd_sfl_env.py` |
| `jaxnav_minimax_unit.py` | JaxNav minimax level editor, BFS solvability, return reduction | `third_party/sfl` | `python ../../cluster/tests/jaxnav_minimax_unit.py` |

Each prints `... OK` on success. Use the DCD conda env for the first two and the SFL env for the third.
