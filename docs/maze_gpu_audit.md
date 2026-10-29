# Maze GPU suite: audit of the reused code

Date: 2026-10-28. Gate for the Maze GPU suite (plan `docs/superpowers/plans/2026-10-28-maze-gpu-suite.md`,
Task 1), done before anything is built on this code. Checked: JaxUED @ 0f8f128 (`third_party/jaxued`,
with `cluster/patches/jaxued_{gif,minimax,accel_cap}.patch` applied), the SFL repo @ 8b14e2e
(`third_party/sfl`), `atlas/maze/{policy,oracle,space,falsify}.py`, `cluster/1_maze/train.sbatch`,
`cluster/submit.sh`, `cluster/run_maze.sh`. Line numbers below are for the checkout as
`cluster/setup/clone.sh` leaves it (all patches applied). Paper settings come from
`docs/ued_atlas.md` section 1 ("Paper-exact settings", lines 34-47); the suite settings from the
suite spec's method table (`docs/superpowers/specs/2026-10-28-ued-generalization-suite-design.md`,
lines 85-99, "JaxUED defaults").

Verdicts: **OK** matches; **defect** wrong for this suite and fixed here; **deviation** differs
from a paper or the spec, kept, with the reason; **open** needs a decision.

Everything was run on the Mac CPU (JAX CPU build). No GPU behaviour was tested.

## Summary

| # | check | verdict |
|---|---|---|
| 1 | JaxUED PLR/RPLR/ACCEL settings vs the papers and the suite spec | OK against the suite spec. Six deviations from the DCD paper configs (all JaxUED defaults the suite spec chose). |
| 2 | ACCEL's edits capped at 25 walls | **defect, fixed**: upstream edits drift to 55-66 walls. New patch `jaxued_accel_cap.patch`. |
| 3 | `--exploratory_grad_updates` is PLR, the default is Robust PLR | OK |
| 4 | SFL: 25 walls and 245,760,000 steps through its config | OK (Hydra overrides, below); `train.sbatch` does not pass them yet (Task 6) |
| 5 | SFL scouting N/K/T/rho vs the spec (5000/1000/50/0.5) | **open**: K and rho match. The repo's Minigrid config and the paper's own Minigrid row both use N = 25,000 and T = 100. The spec's 5000/50 are the paper's general defaults. |
| 6 | SFL scouting steps can be counted | OK: closed form from the config (below) |
| 7 | `oracle.solve`: JaxUED movement rule, 250-step limit | OK (0/3000 transition mismatches against the real env) |
| 8 | `policy.run`: stochastic policy, LSTM reset per episode, exactly the level given | OK |
| 9 | minimax patch: adversary reward = minus the student's return | OK |
| 10 | minimax: student network matches the other methods | deviation: tanh instead of ReLU in one 32-unit layer of the policy head (JaxUED's own PAIRED student). Not patched. |
| 11 | minimax adversary stays inside the 25-wall box | **defect, fixed**: with the default 50 steps the adversary reaches 47 walls. `train.sbatch` now passes `--adv_num_steps 28`. |
| 12 | `atlas/maze/space.py` `dr` space for the VerifAI exam | deviation for Task 3: it spans 0-60 walls, but the exam needs 0-25 |
| 13 | SFL venv's `jaxued` is unpinned | deviation: right now it is 0f8f128 with identical maze files, but a fresh install would pull JaxUED HEAD |

## 1. JaxUED PLR / Robust PLR / ACCEL settings

`examples/maze_plr.py` defaults, compared with the paper settings (DCD configs, `docs/ued_atlas.md:38-43`)
and the suite spec. `maze_dr.py` has the same PPO, generator and network defaults
(`maze_dr.py:405-407, 633-650`).

| setting | JaxUED (`maze_plr.py`) | Robust PLR paper (mg_25b) | ACCEL paper (mg_60b) | suite spec | verdict |
|---|---|---|---|---|---|
| train levels | 13x13 (`:469`), `n_walls` 25 (`:904`), cells drawn with replacement, so at most 25 walls (`util.py:26-28`) | at most 25 blocks | empty room + edits, 60 blocks | at most 25 walls | OK; ACCEL's start differs from its paper (spec's choice, "one box") |
| view / episode | 5x5 (`:902`), 250 steps (`env.py:77`) | same | same | 5x5, 250 | OK |
| budget | `--num_updates 30000` x 32 x 256 = 245,760,000 | 250M | 250M | 245.76M | deviation of 1.7% (spec) |
| rollout | 32 envs x 256 steps (`:878-879`) | 32 x 256 | same | | OK |
| PPO | 5 epochs, 1 minibatch, lr 1e-4, gamma 0.995, clip 0.2, grad norm 0.5, value coef 0.5 (`:873-886`) | same | same | | OK |
| entropy coef | 1e-3 (`:885`) | 0.01 (atlas table) | 0.01 | JaxUED default | deviation (JaxUED default) |
| GAE lambda | 0.98 (`:884`) | not in the atlas table | | | JaxUED default; not compared |
| lr schedule | linear decay per *gradient* step (`:491-496, 508`) | (not in table) | | | JaxUED behaviour, kept. With no gradient on new levels (Robust PLR, ACCEL), the rate decays only to about `1 - grad_updates/30000` (around 0.2 for Robust PLR at full buffer, more for ACCEL), never to 0. |
| replay prob | 0.8 (`:891`) | 0.5 | 0.8 | 0.8 | deviation from Robust PLR paper (spec) |
| buffer | 4000 (`:890`) | 4000 | 4000 | 4000 | OK |
| score | MaxMC (`:888`) | grounded signed value loss | PVL | MaxMC | deviation from both papers (spec) |
| prioritization | rank, temperature 0.3 (`:893, 896`) | rank, 0.1 | rank, 0.3 | 0.3 | deviation from Robust PLR paper (spec) |
| staleness | 0.3 (`:892`) | 0.3 | 0.3 | 0.3 | OK |
| ACCEL edits | 5 per child (`:900`, used at `:660`) | | 5 | "5 tile flips" | deviation in wording. Each edit draws one of NO_OP, FLIP_WALL, MOVE_GOAL uniformly (`util.py:126-200`, `:193`), so about 5/3 wall flips per child, not 5. This is JaxUED's minimax-derived mutator, kept. |
| ACCEL wall cap | none upstream | | | at most 25 walls | **defect, fixed (section 2)** |
| network | conv 16 filters 3x3, direction embedding 5, LSTM 256, 32-unit actor and critic heads (`:292-325`) | LSTM 256 with a conv encoder | same | | OK |

Every update counts all 32 x 256 environment steps, whether or not a gradient is taken. So
the budget covers Robust PLR's new-level rollouts and ACCEL's edit evaluations, as in the papers.
In the CPU run below, ACCEL split 600 updates into 100 DR, 250 replay and 250 mutation updates,
so only the replay updates took gradients.

## 2. ACCEL's edits were not capped at 25 walls (defect, fixed)

**Evidence.** `maze_plr.py:475` (upstream: `mutate_level = make_level_mutator_minimax(100)`) applies
`num_edits` mutations with no wall limit. FLIP_WALL picks a uniform non-agent, non-goal cell and
toggles it (`util.py:133-150`). On a 25-wall level about 85% of cells are free, so a flip adds a
wall far more often than it removes one. Children enter the buffer (`:660-674`) and become
parents again, so a lineage drifts away from the box the exam is built on.

**Reproduction before the fix:**
- Mutator lineage: 256 levels from the 25-wall generator, 30 generations of 5 edits, each child
  replacing its parent. Maximum 66 walls, mean 50.1 at generation 30.
- Real training (`maze_plr.py --use_accel --num_updates 600 --num_steps 16
  --level_buffer_capacity 128 --minimum_fill_ratio 0.1`, CPU, 75 s): **124 of the 128 buffer
  levels have more than 25 walls**, up to 55 (mean 37.8).

**Fix.** `cluster/patches/jaxued_accel_cap.patch` adds `cap_walls(mutate_level, max_walls)`
(`maze_plr.py:397-408`) and uses it at `:475` with `config["n_walls"]`. The edits are applied one
at a time by calling the upstream mutator with `n=1`. This is the same distribution as one call
with `n=5`, because the edit types are drawn i.i.d. and each edit uses its own key. An edit that
would leave more than `n_walls` walls is skipped. `clone.sh` applies the patch with `apply_once`,
which fails loudly if the patch neither applies nor is already applied.

**After the fix** (same commands, same seeds):
- Mutator lineage: maximum 25, mean 24.8. Agent and goal are never on a wall, and the agent is never on the goal.
- Real training run: 0 of 128 buffer levels over 25 walls, maximum 25. The update split is the same (100/250/250).
- `cluster/tests/maze_box_check.py` asserts both: upstream goes above 25 walls and `cap_walls` reaches exactly 25. It prints `maze box check OK`.

## 3. PLR vs Robust PLR switch

- `--exploratory_grad_updates` defaults to False (`maze_plr.py:889`). False means Robust PLR (the run is tagged "robust", `:412-413`).
- New levels (`on_new_levels`, `:532-590`) are rolled out, scored and inserted. The PPO update runs with `update_grad=config["exploratory_grad_updates"]` (`:576`), and `update_actor_critic_rnn` applies gradients only `if update_grad` (`:267`).
- Replay levels always train (`update_grad=True`, `:634`). ACCEL's mutated levels follow the same flag as new levels (`:696`).
- So `--exploratory_grad_updates` takes gradients on new levels (PLR), and the default does not (Robust PLR). The branch choice is `:717-721`. ACCEL runs the mutate branch right after every replay step, which is the ACCEL paper's loop.
- **Verdict: OK.** `train.sbatch` maps plr to `--exploratory_grad_updates`, rplr to no flag and accel to `--use_accel` (Robust PLR + edits). That is correct.

## 4-6. SFL: settings recipe and scouting count

The SFL run uses the same `Maze` env and the same `make_level_generator` (`minigrid_sfl.py:85-86`), and
JaxUED's network (`from sfl.train.minigrid_plr import ActorCritic`, `:16`; `minigrid_plr.py:303-337`
is identical to `maze_plr.py:292-325`). The `jaxued` installed in `.venv-sfl` is git 0f8f128
(its `direct_url.json`). Its `maze/env.py`, `util.py`, `level.py` and `wrappers/autoreplay.py` are
byte-identical to the pinned checkout. SFL's PPO is its paper's own setting, not JaxUED's:
256 envs x 256 steps, lr 2.5e-4, gamma 0.99, lambda 0.95, clip 0.04, entropy 0, 4 epochs x 4 minibatches
(`config/learning/ippo-minigrid.yaml`). This is a deviation by design: the spec says "papers' own code".

**Walls and steps (Hydra overrides; verified with `--cfg job` and a tiny CPU run):**

```
python sfl/train/minigrid_sfl.py SEED=<s> WANDB_MODE=offline SAVE_PATH=<dir> \
    env.N_WALLS=25 learning.TOTAL_TIMESTEPS=245760000 [scouting overrides below]
```

- `env.N_WALLS` (default 60, `config/env/minigrid.yaml:2`) feeds the generator (`minigrid_sfl.py:86`). Both training halves come from it or from the scouting buffer, so both stay inside the box. The fresh half is `NUM_ENVS_TO_GENERATE` = 128 (`:623-630`); the buffer half is `NUM_ENVS_FROM_SAMPLED` = 128 (`:633-638`), and the buffer itself is drawn from the generator (`:251`).
- `learning.TOTAL_TIMESTEPS` (default 3e8, `ippo-minigrid.yaml:6`) sets `NUM_UPDATES = TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS` (`:93-95`). For 245,760,000 that is 3,750.
- **Catch:** the outer loop runs `NUM_UPDATES // EVAL_FREQ` times, each round being one scouting pass followed by `EVAL_FREQ` updates (`:678-681, 727`). With the default `EVAL_FREQ=100`, only 37 x 100 = 3,700 updates run: **242,483,200 trained steps, 1.3% short**. The paper's "4,500 updates" is exactly this truncation of 3e8 (4,577 down to 4,500), so it is upstream behaviour.

**Scouting parameters (open: needs a decision).** Mapping: N = `BATCH_SIZE` x `NUM_BATCHES` (the top K is taken over all batches together,
`:284-289`), L = `ROLLOUT_STEPS`, K = `NUM_TO_SAVE`, T = `learning.EVAL_FREQ`, rho = `NUM_ENVS_FROM_SAMPLED / NUM_ENVS`.

| | N | L | K | T | rho | source |
|---|---|---|---|---|---|---|
| repo Minigrid config | 5 x 5000 = 25,000 (`minigrid-sfl.yaml:18-19`) | 2000 (`:20`) | 1000 (`:21`) | 100 (`ippo-minigrid.yaml:5`) | 128/256 = 0.5 (`minigrid-sfl.yaml:14-15`) | code |
| SFL paper, Table 4, Minigrid column (arXiv 2408.15099, HTML) | 25,000 | 2000 | 1000 | 100 | 0.5 | paper's own maze row |
| SFL paper section 4.3, "default values ... work well across domains" | 5000 | 2000 | 1000 | 50 | 0.5 | general defaults |
| suite spec / `ued_atlas.md:45-47` | 5000 | (2000) | 1000 | 50 | 0.5 | took the general defaults |

K and rho match. N and T in the spec are the paper's cross-domain defaults, not the paper's or
the repo's Minigrid setting. (The extracted Table 4 merges cells. Its T and K rows read
"50 50 100 4" and "100 100 1000 8192" for JaxNav single, JaxNav multi, Minigrid, XLand, and its
N row reads "5000 25000 40000". The Minigrid reading agrees with the repo config, which supports
it.) Two consistent recipes:

| recipe | overrides beyond walls/steps | rounds | trained steps | scouting env steps | scouting / training |
|---|---|---|---|---|---|
| A. paper's Minigrid row (repo defaults) | none | 37 | 242,483,200 | 37 x 25,000 x 2,000 = **1,850,000,000** | 7.6x |
| B. spec values | `BATCH_SIZE=5000 NUM_BATCHES=1 learning.EVAL_FREQ=50` | 75 | 245,760,000 (exact) | 75 x 5,000 x 2,000 = **750,000,000** | 3.1x |

Recipe A follows "the papers' own code, unchanged", but trains 1.3% short. Setting
`TOTAL_TIMESTEPS=249036800` gives 38 rounds and 249,036,800 steps, 1.3% over. Recipe B matches
the approved spec and the exact budget. Either way scouting is not charged (user decision), but it is several times
the training steps in GPU time, so it should go into the README's time estimate. **The
controller or user should pick A or B; this audit does not.**

**Counting scouting steps for `scores.csv` (`scouted_steps`).** Scouting runs once at the start of every round,
including before the first update (`:678`). Each run steps `NUM_BATCHES` x `BATCH_SIZE` levels for
`ROLLOUT_STEPS` steps under `AutoReplayWrapper` (`:89, 266-267`), and no step is skipped. So:

```
NUM_UPDATES   = TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS            # learning.*
rounds        = NUM_UPDATES // EVAL_FREQ
trained_steps = rounds * EVAL_FREQ * NUM_STEPS * NUM_ENVS
scouted_steps = rounds * NUM_BATCHES * BATCH_SIZE * ROLLOUT_STEPS
```

All of these are the Hydra overrides `train.sbatch` passes. Hydra also writes the composed config to
`third_party/sfl/outputs/<date>/<time>/.hydra/config.yaml`. Task 6 should compute both numbers
from the values it passes and write them into the run's `config.json`, along with `n_walls: 25`.
Today `config.json` lacks `n_walls`, so `Policy` falls back to 60 (`policy.py:37`). That only
affects the old falsifier's context, not grading. Per-round evaluation (5 attempts x 8 named
mazes x 250 steps = 10,000 steps, `:692`) is neither training nor scouting, and JaxUED's
evaluation isn't charged either.

**Tiny CPU run** (`env.N_WALLS=25` plus `submit.sh`'s smoke overrides): 48 s. It wrote
`<SAVE_PATH>/s0/model.safetensors`, which `atlas.maze.policy.Policy` loads and plays (section 8).

## 7. `oracle.solve` vs JaxUED's movement rule

- JaxUED (`env.py:218-244`): forward moves along `DIR_TO_VEC` (right, down, left, up, `:53-58`), clamped to the grid. A wall or the goal blocks the move. Entering the goal is a forward action into it, and that action ends the episode with reward. Left and right turn by -1/+1 mod 4. The other four actions are no-ops. `time` increments after every action, and the episode is done at `time >= 250` (`:120-130, 168-170`).
- `oracle._step` (`oracle.py:33-44`) implements the same rule. `solve` (`:47-73`) is a BFS over (x, y, dir) with actions F/L/R, so the first time it hits the goal gives the minimal action count. It returns SOLVABLE iff that count is at most `max_steps` (250, `:28`).
- **Reproduction:** 3,000 random (25-wall level, free start cell, direction, action) transitions compared against `Maze.step_env` (agent position, direction, goal reached): **0 mismatches**. A certificate of length 18 started at `time = 232` reaches the goal on the 250th action and pays reward 0.1 (the time penalty), and the episode is done. Started one step later, the episode ends before the goal. So "at most 250 actions, the last one may enter the goal" is exactly right.
- **Verdict: OK.** `tests/test_atlas.py` (8 tests, pass) also replays certificates in the real env.

## 8. `policy.run`

- `Policy.run` (`policy.py:82-89`) builds one `Level` from exactly the walls, start, direction and goal given (13x13). It repeats that level `attempts` times and calls `_rollout` (`:69-79`), which resets each copy with `reset_to_level` (no randomisation, `env.py:135-158`). It then calls JaxUED's own `evaluate_rnn` (`maze_plr.py:147-206`): actions are `pi.sample` (`:180`), so the policy is stochastic. The LSTM starts from `initialize_carry` (zeros) for every attempt. Rewards and lengths count the first episode only (mask, `policy.py:78`; the unwrapped env does not auto-reset).
- Minimax checkpoints load the protagonist with `maze_paired.ActorCritic` (`:53-56`). SFL checkpoints load into `maze_plr.ActorCritic` (`:33-44`), the same architecture as SFL's network.
- **Reproduction** on three tiny CPU checkpoints (ACCEL, minimax, SFL): all 10 attempts start at the given agent cell; the same seed reproduces the result exactly; and an unrelated call in between leaves it unchanged, so no state carries over. The 10 attempts give 10 distinct trajectories for minimax and SFL. For the 600-update ACCEL net they give 1, because that net puts 0.991 on "forward" and stands against a wall; the sampling code is the same, so this comes from the net, not the code.
- **Verdict: OK.**

## 9-11. Minimax (`maze_paired.py --minimax`)

- **Adversary reward** (`jaxued_minimax.patch`): with `--minimax`, `est_regret = -pro_mean_returns` (`maze_paired.py:559-561`). It is the adversary's only reward, placed on its last step (`:563`). So the adversary minimises the protagonist's mean return, the Dennis et al. 2020 minimax baseline. **OK.** The antagonist is still rolled out and trained (`:554-556, 567`) but no longer affects anything, which wastes compute but changes no result. Our grader plays the protagonist (`pro_train_state`, `policy.py:53-54`).
- **Student network:** `maze_paired.ActorCritic` matches `maze_plr.ActorCritic` (conv 16, LSTM 256, 32-unit heads), except that `actor0` is followed by **tanh** (`maze_paired.py:301`), where maze_plr uses ReLU (`maze_plr.py:313`). Student PPO defaults match maze_plr (`maze_paired.py:715-726`). **Deviation, not patched.** It is JaxUED's own PAIRED student, so changing it would change the paper code for a non-bug, and it affects one 32-unit layer of the policy head. It is listed so the minimax result is read with it in mind.
- **Adversary stays inside the box: defect, fixed.** The adversary (`MazeEditor`, `env_editor.py:119-163`) runs `adv_num_steps` steps (default 50, `maze_paired.py:730`). Step 0 places the goal, step 1 sets the agent's direction, step 2 places the agent, and every later step toggles a wall. Fifty steps allow 47 walls.
  - Reproduction: an untrained (uniform) adversary averages 35.8 walls (maximum 47), and 100% of its levels exceed 25. An adversary that toggles a fresh cell each step reaches 47.
  - With `--adv_num_steps 28` (3 placement steps + 25 toggles, the analogue of DCD's 25-block adversary): uniform adversary mean 21.6, maximum 25; fresh-cell adversary 25. A toggle can also remove a wall, so 25 is a hard cap.
  - Fix: `cluster/1_maze/train.sbatch` passes `--adv_num_steps 28` for minimax. This is a command-line setting, not a code change, so it needs no patch. A 20-update CPU run with the flag trains and checkpoints, and `Policy` loads and plays it. `cluster/tests/maze_box_check.py` asserts 47 walls at 50 steps and 25 at 28.
- Not verified: whether JaxUED's adversary network and PPO settings (conv 128, LSTM 256, 16-dimensional random z, lr 1e-4, entropy 1e-3, 5 epochs, `:728-739`) equal DCD's. The suite spec says "DCD's adversary network and PPO settings"; what runs is JaxUED's port.

## 12-13. Other findings

- **`space.py` `dr` space spans 0-60 walls** (`space.py:27`, `n_walls: (0, 61)`). The spec's VerifAI section is "cross-entropy search over the 25-wall `dr` space". Task 3 must search `n_walls` in (0, 26) (at most 25 walls; agent and goal cells are cleared afterwards, `:54`), not the existing bound. The space places exactly n distinct cells, while the generator draws with replacement; both stay inside the box. This file is left unchanged because the old atlas uses it and Task 3 owns the exam.
- **SFL's `jaxued` is unpinned:** `third_party/sfl/requirements.txt:4` says `jaxued @ git+https://github.com/DramaCow/jaxued` with no commit, and `setup_sfl.sh` installs it. The local venv happens to hold 0f8f128 with identical maze files. A fresh setup on the GPU machine would take JaxUED HEAD. Suggested fix for Task 6: install `jaxued @ git+...@$JAXUED_COMMIT` in `setup_sfl.sh` after `pip install -e "$SFL_DIR"`.
- **`clone.sh`'s loop for `jaxued_gif`/`jaxued_minimax`** prints "already applied" whenever `git apply --check` fails, including when the patch doesn't apply at all. The new patch uses the stricter `apply_once`. Moving the two old ones to `apply_once` would catch a silently missing minimax flag.
- **`train.sbatch` (Task 6):** the SFL line passes neither `env.N_WALLS=25` nor `learning.TOTAL_TIMESTEPS=245760000`, so as written it trains at 60 walls for 294,912,000 steps. `submit.sh`'s smoke `$S` overrides also lack `env.N_WALLS`. `learning.WARMUP_UPDATES` in the smoke overrides is unused by `minigrid_sfl.py`, which is harmless.
- Out of scope, as known: JaxUED's gymnax ACCEL is broken upstream, and the bridge excludes it.

## Reproductions

All of these ran on the Mac CPU with `.venv-jax` (JAX 0.10.2) or `.venv-sfl` (JAX 0.4.30). The scratch scripts and outputs are in the session
scratchpad; the durable check is `cluster/tests/maze_box_check.py`.

| what | command (from the repo root unless noted) | result |
|---|---|---|
| oracle vs env | scratch `check_oracle.py` (3,000 random transitions + 250-step boundary) | 0/3000 mismatches; boundary as in section 7 |
| ACCEL lineage before/after | scratch `check_accel_cap.py` | upstream max 66 walls, capped max 25 |
| ACCEL training before | `maze_plr.py --use_accel --num_updates 600 --eval_freq 200 --num_steps 16 --level_buffer_capacity 128 --minimum_fill_ratio 0.1` (75 s) | buffer: 124/128 levels over 25 walls, maximum 55 |
| ACCEL training after | same, patched (77 s) | buffer: 0/128 over 25, maximum 25 |
| minimax walls | scratch `check_minimax_walls.py` | 50 steps: mean 35.8, max 47; 28 steps: mean 21.6, max 25 |
| minimax run | `maze_paired.py --minimax --adv_num_steps 28 --num_updates 20 --eval_freq 10 --student_num_steps 16` (3 min 42 s) | trains and checkpoints; `Policy` plays it |
| SFL config | `minigrid_sfl.py env.N_WALLS=25 learning.TOTAL_TIMESTEPS=245760000 --cfg job` (in `third_party/sfl`) | `N_WALLS: 25`, `TOTAL_TIMESTEPS: 245760000`, `EVAL_FREQ: 100`, `BATCH_SIZE: 5000`, `NUM_BATCHES: 5`, `ROLLOUT_STEPS: 2000`, `NUM_TO_SAVE: 1000` |
| SFL tiny run | smoke overrides + `env.N_WALLS=25` (48 s) | `model.safetensors` written; `Policy` plays it |
| policy | scratch `check_policy.py` on the ACCEL, minimax and SFL checkpoints | section 8 |
| box check | `.venv-jax/bin/python cluster/tests/maze_box_check.py` | `maze box check OK` |
| atlas tests | `.venv-jax/bin/python -m pytest tests/test_atlas.py -q` | 8 passed |

## End-to-end CPU run (2026-10-28)

The whole pipeline (`cluster/run_maze.sh` -> `submit.sh maze`, LOCAL mode) ran on the Mac CPU at a
tiny size, with all six methods and REF, from a fresh `$RUNS`. The CPU was shared with an 8-worker
LunarLander run (load average 12-39 throughout), so every wall time below is inflated.
No code changes were needed. CPU only; no GPU behaviour tested.

**Command** (from the repo root; `RUNS` was a fresh scratch directory):

```
ALLOW_CPU=1 RUNS=<scratch>/e2e NUM_UPDATES=50 NSEEDS=1 \
  JAXUED_ARGS="--eval_freq 10 --checkpoint_save_interval 1" SFL_STEPS=4096 \
  SFL_ARGS="learning.NUM_ENVS=16 learning.NUM_ENVS_FROM_SAMPLED=8 learning.NUM_ENVS_TO_GENERATE=8 learning.NUM_STEPS=32 learning.EVAL_FREQ=2 learning.NUM_CHECKPOINTS=4 BATCH_SIZE=64 NUM_BATCHES=1 ROLLOUT_STEPS=50 NUM_TO_SAVE=32" \
  EXAM_ITERS=20 BRIDGE_SEEDS=1 bash cluster/run_maze.sh
```

`BRIDGE_STEPS` stayed at its default of 614,400. `compare.py` only reads rows at the budget of
batch 21, and 614,400 steps is well past the point where replay starts. All outputs went under
`$RUNS/outputs`, and `git status` showed no change under `results/` or `frozen_sets/`.

| stage | wall time (18:07:21 - 18:53:30, 46 min in all) | result |
|---|---|---|
| setup (clone, patches, `.venv-jax`, `.venv-sfl`, new `.venv-torch`) | ~3 min | all import checks `ok` |
| smoke (DR, RPLR, SFL; exam from 10 copies of DR; grade) | 4 min | passed; `JOBS_PER_GPU=1` |
| training, 16 tasks: SFL, REF 100-109, DR, PLR, RPLR, ACCEL, minimax | 35 min, one at a time (a `--gres=gpu` job on a host without a GPU gets a single `cpu` slot) | 16/16 ok. About 1.5 min each, except minimax at 11 min |
| exam (`EXAM_ITERS=20`, 10 REF agents) | 1 min | `froze verifai: 196 of 196 levels`; manifest holds random, custom, verifai |
| grade | 1 min | `scores.csv`: 18 rows, 6 methods x 3 sections |
| bridge: 3 runs + grade + compare | 1.5 min | DR 24 s, PLR 31 s, RPLR 29 s of training; 9 rows; `bridge_cartpole.md` |
| collect | <1 min | `maze_results.tgz`, 266 MB (most of it is `runs/slurm` and checkpoints) |

**Carried checks.**
1. *Every method, real checkpoints.* SFL wrote a real `sfl/maze_sfl/s0/model.safetensors` (1.7 MB)
   and its `config.json`. Minimax `maze_minimax_n50/0` has `config.json` with `num_updates 50,
   eval_freq 10, checkpoint_save_interval 1, minimax true, adv_num_steps 28` and `models/0-4`.
   `exam._final_step` gives 4, so train.sbatch's `finished` check found it on the rerun and the
   grader loaded it (`MINIMAX 0` graded on all three sections). Every method finished training,
   wrote its final checkpoint and was graded.
2. *Bridge with replay.* At 614,400 steps PLR and RPLR each did 45 DR and 105 replay updates
   (buffer 360/1000). DR did 150 DR and 0 replay updates (buffer 1000). The runs were exported
   (`params.npz`) and graded by the real `grade_cartpole` and `compare`. The verdict is FAIL: with
   n=1 the JaxUED-trained agents pass far less than batch 21 (DR random 0.07 vs 0.94), the same
   numbers as Task 6's bridge run. This run predates 7ec1fd9. At that point the bridge used 4
   minibatches on its 4,096-sample rollout, which meant 2,400 Adam steps per run against our
   9,600, so the agents were undertrained (root cause in
   `.superpowers/sdd/2026-10-28-maze-gpu-suite/bridge-debug-report.md`). With the 16-minibatch
   fix, one full-budget DR seed passes random 0.97, custom 0.96 and verifai 0.447. These bridge
   numbers are stale, and the bridge stage needs a rerun before its verdict means anything.
3. *solve_rates on CPU.* `exam.solve_rates(policy, random section, plays=10, batch=256)` on a
   tiny DR checkpoint covers 300 levels (2 jitted batches of 256 levels x 10 plays) and took
   5.80 s (including compile), then 4.54 s and 3.45 s. That is about 1.7-2.3 s per warm batch of
   256 x 10, under load 13-21.
4. *Fresh-$RUNS output dirs.* `grade_cartpole` (`out.parent.mkdir`) and `compare`
   (`Path(out).parent.mkdir`) created `outputs/results/suite/cartpole_jaxued/` and
   `.../analysis/` themselves. `bridge.sbatch` also runs `mkdir -p` on `analysis`.
5. *Rerun* (18:54:12 - 18:57:42, 3.5 min, almost all of it setup's pip checks). Every training
   task printed `already trained`, the exam printed `exam already frozen; nothing to do`, Maze
   grading graded nothing, the bridge runs printed `already trained`, and `grade_cartpole` found
   `3 runs found, 0 to grade`. The sha1 of all 76 csv/npz/json/safetensors/md outputs under
   `$RUNS` was identical before and after. The tarball was produced again.
6. *Tarball analysis.* Unpacked in the scratchpad, the run files sit under the runs directory's
   own name (`e2e/outputs/...` here; `runs/outputs/...` with the default `RUNS`).
   `acl_bench.suite.analyze` has no results option: it reads `results/suite` next to its own
   package. I copied `acl_bench/` and the tarball's `outputs/results/suite/maze` into a scratch
   tree and ran `python -m acl_bench.suite.analyze --envs maze --allow-partial` there. It printed
   the screen and the per-arm table (6 arms, n=1, all near 0.05 random / 0.07 verifai at this
   size) and wrote `summary.md`, `screen.csv`, `table_maze.csv` and `custom_maze.csv`.
   `outputs/results/suite/analysis/bridge_cartpole.md` is in the tarball.

**Notes.** In the progress lines, two bridge tasks that finish at the same moment can both print
`3/3 done`: the counter reads `done` after both have appended to it. This is cosmetic. On a CPU-only
host the GPU-typed jobs (train, exam, grade) run one at a time.
