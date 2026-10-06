# Tests for the ported cells

Headless (no windows). Run after `cluster/setup/clone.sh` has applied the patches.

| Test | Covers | Run from | Command |
| --- | --- | --- | --- |
| `jaxnav_minimax_unit.py` | JaxNav minimax level editor, BFS solvability, return reduction | `third_party/sfl` | `python ../../cluster/tests/jaxnav_minimax_unit.py` |

Each prints `... OK` on success. Use the SFL env (`.venv-sfl`).
