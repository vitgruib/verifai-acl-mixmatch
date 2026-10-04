# UED failure atlas: technique registry, standard suite, test pipeline

Goal (2026-10-03): document published UED algorithms, show where they fail, with and without
VerifAI falsification, on the exact environments their papers used, and explain why. Fixes
(falsifier + PLR) come later. This file is the registry and the plan; results are appended as
they come in.

Sources checked 2026-10-03: the official DCD configs (`facebookresearch/dcd`,
`train_scripts/grid_configs/`, read directly), the SFL paper (arXiv 2408.15099, HTML), the JaxUED
paper (arXiv 2403.13091). The PAIRED, PLR and ACCEL environment lists come from the papers'
abstracts and the DCD README. Anything marked *(unverified)* still needs checking against the
paper's appendix.

## 1. Technique registry: published UED algorithms

| Algorithm | Paper | Objective / level score | Generator | Envs in the paper | Official code | In our repo |
|---|---|---|---|---|---|---|
| DR | (baseline in all) | none: uniform levels | random | all below | DCD, JaxUED | `DR` |
| Minimax adversary | Dennis+ NeurIPS'20 (baseline) | minimise protagonist return | RL adversary that places walls | MiniGrid maze, CarRacing, Bipedal | DCD | **missing** (needs an RL generator) |
| PAIRED | Dennis+ NeurIPS'20 | regret = antagonist return - protagonist return | RL adversary + antagonist agent | MiniGrid maze (25 blocks; Labyrinth, Maze, ...) | `ucl-dark/paired`, DCD, JaxUED | **missing** |
| PLR | Jiang+ ICML'21 | replay levels with high TD-error score (PVL / L1 / GAE), staleness mix | random + replay buffer | Procgen (16 games), MiniGrid | `facebookresearch/level-replay`, DCD | `paper`, `paper_l1`, `paper_maxmc`, ... |
| Robust PLR (PLR-perp) | Jiang+ NeurIPS'21 | as PLR, but **no gradient on new random levels** (train only on replay) | random + replay | MiniGrid maze 25 blocks; CarRacing (Bezier train, F1 test) | DCD | `paper_robust` |
| REPAIRED | Jiang+ NeurIPS'21 | PAIRED adversary + PLR replay (grounded regret) | RL adversary + replay | maze, CarRacing | DCD | **missing** |
| ACCEL | Parker-Holder+ ICML'22 | Robust PLR + **edit** replayed levels (small mutations), keep high-regret edits | edits from empty rooms + replay | MiniGrid maze 60 blocks; MiniHack lava; BipedalWalker | DCD, JaxUED | `accel`, `accel_maxmc` (edits parameters, not tiles; see section 2) |
| PAIRED+HiEnt+BC+Evo | Mediratta+ CoLLAs'23 | stabilised PAIRED | adversary + edits | maze, CarRacing, Bipedal | DCD | missing (low priority) |
| SFL | Rutherford+ NeurIPS'24 | learnability p(1-p): sample N levels, roll out, replay the top K | random + scouting rollouts | MiniGrid maze (60 walls), XLand-MiniGrid (high-3m, 13x13, 4 rooms), JaxNav single-agent and multi-agent | `amacrutherford/sampling-for-learnability` *(unverified URL)* | `sfl`, `sfl_tilt`, ... |
| VerifAI-picker variants (ours) | this project | falsifier samplers (CE, MAB) choose levels | VerifAI | classic control | n/a | `vlearn_ce`, `vlearn_mab`, `oracle*` |

Our own non-paper arms (around 80 in `acl_bench/plr/screen.py`: score variants, SFL variants
from batches 1-18, `var_low`, `fmodel`, `lp`, ...) are recorded with their verdicts in
`docs/library_log.md` and `docs/library.md`. They belong to the old goal and are **not** part of
the atlas unless reused.

### Paper-exact settings (from the DCD configs)

| Setting | Maze, Robust PLR / PAIRED (`mg_25b_*`) | Maze, ACCEL (`mg_60b_uni_*`) | CarRacing (`cr_*`) | BipedalWalker (`bipedal_*`) |
|---|---|---|---|---|
| Train env | `GoalLastFewerBlocksAdversarial` (<= 25 blocks) | DR / Robust PLR / PAIRED: `GoalLastVariableBlocks` (0-60); ACCEL: empty room + edits | `CarRacing-Bezier-Adversarial` | `BipedalWalker-Adversarial(-Easy)` |
| Test envs | named mazes (Labyrinth, Maze, SixteenRooms, ...) | SixteenRooms, Maze, Labyrinth (+ others in the paper) | Vanilla + 20 F1 tracks | v3, Hardcore, Stairs, PitGap, StumpHeight, Roughness |
| Steps | 250M | 250M | 5.5M | **2B** |
| Network | LSTM 256 (conv obs encoder) | LSTM 256 | CNN (pixels) | MLP |
| PPO | 32 procs x 256 steps, 5 epochs, 1 minibatch, lr 1e-4, gamma 0.995, ent 0.01 | same | 16 x 125, 8 epochs, 4 mb, lr 3e-4, gamma 0.99 | 16 x 2048, 5 epochs, 32 mb, lr 3e-4, gamma 0.99 |
| PLR | replay 0.5, buffer 4000, score `grounded_signed_value_loss`, rank temperature 0.1, staleness 0.3 | replay 0.5 (ACCEL 0.8), buffer 4000, PVL, temperature 0.3; ACCEL 5 edits | replay 0.5, buffer 8000, PVL, temperature 1.0, staleness 0.7 | replay 0.5 (ACCEL 0.9), buffer 1000, PVL, staleness 0.5; ACCEL 3 edits |

SFL (paper, maze): 4,500 updates x 256 envs (JAX), fully connected network ([16, 256]); SFL
defaults T=50, rho=0.5, N=5000, L=2000, K=1000. Evaluation is **CVaR over 10,000 random solvable
levels x 10 episodes** (the worst alpha%). That is a random-sampling adversarial evaluation,
which is exactly what a VerifAI falsifier generalises.

## 2. How our current suite deviates (it is not yet standardized)

| Item | Ours now | Paper | Consequence |
|---|---|---|---|
| Network | MLP, no memory | LSTM 256 (DCD) / FC (SFL, JAX) | Our held-out maze success (DR 0-0.13, sfl_tilt 0-0.18) **cannot** be read as the algorithms failing; a memoryless agent can't do the named mazes |
| Budget | 5.12M steps | 250M (DCD maze), ~295M (SFL maze) *(4,500 x 256 x 256, to verify)* | 50x short |
| PPO | 8 envs, lr 3e-4, gamma 0.99, 4 epochs, 4 mb | 32 procs, lr 1e-4, gamma 0.995, 5 epochs, 1 mb | different |
| PLR buffer | 1000, temperature 0.1, staleness 0.1 | 4000, temperature 0.1-0.3, staleness 0.3 | different |
| Level encoding | 2 scalars `(n_walls, layout_seed)` | tile grid (adversary places walls one at a time; ACCEL edits tiles) | ACCEL "edits" change a seed, which changes the whole maze, so it is not ACCEL. A VerifAI falsifier over `layout_seed` **degenerates to random search** (neighbouring seeds share nothing) |
| Train distribution | 0-60 walls uniform | 25 blocks (Robust PLR) vs 0-60 (ACCEL, SFL) | matches ACCEL/SFL only |
| Algorithms | no minimax, PAIRED, REPAIRED | all present in DCD / JaxUED | coverage gap |

**Standardization proposal**: train with the reference implementations, unchanged: JaxUED
(DR, PLR, Robust PLR, ACCEL, PAIRED on the maze), SFL's official code (SFL, plus its own
baselines on maze / XLand / JaxNav), and DCD (CarRacing F1, BipedalWalker). Our repo does
evaluation, falsification and diagnosis only. Then a reviewer can't say the failures are our
reimplementation bugs. The torch harness stays for cheap probes.

## 2b. Decision: run the papers' own code; environment x method matrix

Decided 2026-10-03 (user): train with each paper's own code, at a scoped budget, rather than our
re-implementations. Our suite (section 2) stays as the falsification/diagnosis harness.

| Environment (paper protocol) | Introduced by | Methods the papers ran there | Code |
|---|---|---|---|
| MiniGrid maze, 25 walls, 8 named held-out mazes | PAIRED | DR, minimax, PAIRED, PLR, Robust PLR, REPAIRED, ACCEL, SFL | JaxUED (DR/PLR/RPLR/ACCEL/PAIRED), SFL repo, DCD |
| CarRacing, Bezier train, Vanilla + 20 F1 tracks | Robust PLR | Robust PLR, with DR, PLR, PAIRED, REPAIRED as its baselines | DCD only |
| BipedalWalker, Adversarial-Easy train, v3/Hardcore/4 Med tests | ACCEL | ACCEL, with DR, PLR, Robust PLR, PAIRED as baselines | DCD |
| XLand-MiniGrid, JaxNav | SFL | SFL, DR, PLR, Robust PLR, ACCEL | SFL repo (jaxmarl) |

CarRacing is a single-paper benchmark (Robust PLR). The maze is the only environment shared by all.

**Uniform matrix (decided 2026-10-04, user):** every method is tested on every environment
(Maze, CarRacing, BipedalWalker, JaxNav), not only where its paper ran it. Paper code still
trains each cell where any paper's code can. Where none can, the cell is a port.

| | Maze | CarRacing | BipedalWalker | JaxNav |
|---|---|---|---|---|
| DR | JaxUED | DCD | DCD | SFL repo |
| minimax | JaxUED + `--minimax` patch | DCD, derived config | DCD | **port** |
| PAIRED | JaxUED | DCD | DCD | **port** |
| PLR | JaxUED | DCD | DCD, derived config | SFL repo |
| Robust PLR | JaxUED | DCD | DCD | SFL repo |
| ACCEL | JaxUED | **port** | DCD | SFL repo |
| SFL | SFL repo (`minigrid_sfl`, JaxUED Maze env) | **port** | **port** | SFL repo |

Covered cells: 23 of 28. Their cluster jobs are in `cluster/README.md`. The five port cells
would need our own code: SFL's sampler inside DCD's PyTorch runner (CarRacing, Bipedal); a
Bezier-track mutator for ACCEL; an adversary generator for JaxNav. They are deferred until
decided. Maze caveat: SFL's own maze settings (60 walls, its PPO) differ from JaxUED's (25 walls).

**Small-scale reproduction feasibility (measured on this Mac):**
- *JaxUED maze:* runs unmodified (one wandb gif patch), headless. About 1.9 s/update contended:
  10% budget (3000 updates) is 1-2 h/run. Feasible locally (batch 19).
- *DCD BipedalWalker:* runs only with py3.11 compatibility patches (numpy aliases, cloudpickle).
  About 8.8k steps/s, so the paper budget (2B steps) is ~63 h of contended local time per run (1% is ~40 min). Full budget: cluster.
- *DCD CarRacing:* not faithful on macOS. It renders through real OpenGL windows (there is no Xvfb),
  and needs patches to gym/Box2D that change track generation. Run unmodified on the Linux cluster
  with the original pins (py3.8, numpy 1.x, gym 0.15.7) under Xvfb.
- *SFL, Procgen, MiniHack:* not yet tested.

## 3. Pipeline to test everything

1. **Train** each algorithm x env x seed (>= 8 seeds; 10 is standard in DCD/SFL) with the
   paper config. Save checkpoints (at least the final one and a few intermediate) and the
   **curriculum logs**: every level sampled and its score, replay buffer snapshots. The
   diagnosis step needs these logs.
2. **Export** policies to a common inference interface (JAX/torch policy -> `act(obs, state)`),
   and levels to a common **structured encoding**: wall bitmap, start, goal (maze); track
   control points (CarRacing); terrain parameters (Bipedal).
3. **Evaluate without VerifAI** (the papers' own protocol, reproduced): named held-out mazes /
   F1 tracks / Bipedal test envs; random-level success; SFL's CVaR over 10k random solvable
   levels. *Failure without VerifAI* = low held-out or CVaR performance. Expected first finding:
   we should reproduce the paper numbers. If we can't, that's a standardization bug, not a
   failure.
4. **Evaluate with VerifAI falsification**: one Scenic spec per env over the structured
   encoding, with **validity as a hard constraint** (maze: BFS-solvable, i.e. `certify_exact`;
   CarRacing: track self-intersection free; Bipedal: within the paper's parameter bounds).
   Property: "reach the goal within 250 steps", scored as a robustness value (e.g. return, or
   STL robustness). Samplers: cross-entropy, Bayesian optimisation, simulated annealing, and
   **Halton/random as the control**. Use the same budget per policy (e.g. 2,000 levels x 10
   episodes) for every arm, with evaluation seeds disjoint from anything used in training.
   Report: falsification rate, time to first counterexample, and the counterexamples themselves.
   *Failure with VerifAI* = the falsifier finds valid levels the policy fails even though its
   paper metrics look fine. The difference between steps 3 and 4 is the point of the atlas.
5. **Diagnose** each failure cluster: did the curriculum ever sample that region? what score
   did it give it? regret-estimate error vs true regret (from an oracle / best-of-N policy);
   share of unsolvable levels in the buffer; learnability at the time. Each failure gets one
   mechanism plus the diagnostic behind it.
6. **(Later) Fix**: falsifier + PLR, compared at equal steps and an equal evaluation falsifier.

Order: maze (all algorithms) -> CarRacing F1 -> BipedalWalker -> XLand / JaxNav.

## 4. Compute: what needs the supercomputer

Measured here (10-core Mac, torch CPU, MLP maze): about 18k steps/s per worker, i.e. DR 5.12M
in 280 s; our SFL-style scouting costs about 9x DR wall-clock.

| Job | Per seed | x runs (8-10 seeds) | Where |
|---|---|---|---|
| Maze, paper-exact (LSTM), DR/PLR/Robust PLR/ACCEL, 250M steps, DCD on CPU | roughly 1-3 days CPU *(estimate; LSTM is slower than our MLP's 3.8 h)* | 4 algorithms x 10 seeds = 40 runs | **supercomputer**, or a GPU with JaxUED (reported about 100x faster than the CPU DCD code) |
| Maze PAIRED / minimax / REPAIRED (3 networks) | about 2-3x the above | 30 runs | **supercomputer / GPU** |
| SFL maze, official JAX code | needs a GPU; the scouting (N=5000 x L=2000 each T) dominates | 10 runs | **GPU** |
| XLand-MiniGrid, JaxNav (SFL paper) | GPU-only (JAX, 256-8192 envs) | 10 x algorithms | **supercomputer GPU** |
| CarRacing F1 (DCD, pixels, 5.5M steps) | several hours with a GPU | 7 algorithms x 10 seeds | GPU node |
| BipedalWalker (DCD, **2B steps**) | days per seed on CPU | 6 algorithms x 10 seeds | **supercomputer** (largest job) |
| VerifAI evaluation (inference only, 2k levels x 10 episodes per policy x 4 samplers) | minutes per policy (maze); about 1 h (Bipedal/CarRacing with rendering) | hundreds of policies | local Mac is fine for the maze; cluster CPU for the others |
| Diagnosis | offline on logs | | local |

Ask for: GPU nodes with JAX (JaxUED, SFL, XLand, JaxNav), and CPU-many-core jobs for DCD
BipedalWalker. Everything after training (steps 3-5) can run locally.

## 4b. Falsifier prelim (maze, 2026-10-04)

Purpose: choose the VerifAI samplers for the cluster falsification (`cluster/maze/falsify.sbatch`).
Setup: batch-19 JaxUED checkpoints (DR, PLR, Robust PLR, ACCEL; 3000 updates = 10% of the paper
budget; train seeds 0-1). Two spaces (`dr`: JaxUED's own generator parameters; `seg`: wall
segments), five samplers (random, halton, ce, mab, sa), falsifier seeds 0-2, 1000 levels each,
10 attempts per level, spec `solve_rate >= 0.5`. Every level is oracle-certified solvable first.
240 runs, no crashes. Driver `results/atlas/prelim_maze.sh`, log `results/atlas/prelim_maze.log`,
table `results/atlas/prelim_maze.csv`, reproduce with `python -m atlas.prelim runs/maze_prelim`.

Ranking (geometric mean over the 4 algorithms, relative to random at the same budget; headline
is distinct failure modes, i.e. coarse `desc` cells, because the atlas wants kinds of failure,
not one failure resampled):

| space | sampler | modes x | cex rate x | invalid |
|---|---|---|---|---|
| dr | **ce** | **1.13** | **4.51** | 0.211 |
| dr | **mab** | 1.06 | 2.46 | 0.123 |
| dr | random | 1.00 | 1.00 | 0.066 |
| dr | halton | 0.99 | 1.01 | 0.060 |
| dr | sa | 0.92 | 0.93 | 0.062 |
| seg | **ce** | **1.07** | **5.89** | 0.173 |
| seg | random | 1.00 | 1.00 | 0.094 |
| seg | sa | 0.96 | 1.07 | 0.084 |
| seg | halton | 0.89 | 1.14 | 0.086 |
| seg | **mab** | 0.87 | 5.22 | 0.145 |

**Chosen: `random ce mab`.** ce is best on both metrics in both spaces. mab is the runner-up on
counterexample rate everywhere (and finds the first one quickly), but concentrates on fewer
modes in `seg`. Random stays as the control every ratio is taken against, and it gives the
unbiased failure rate. Halton is indistinguishable from random and sa is below it, so both are
dropped (this cuts the falsify cost by 40%). The adaptive samplers waste 2-3x more budget on
invalid levels: they steer toward wall-heavy regions where more draws are unsolvable.

Failure rate under random search (share of solvable levels with solve rate < 0.5; 10% budget,
so these are not the paper agents): `dr` space DR 0.053, PLR 0.056, Robust PLR 0.058, ACCEL 0.083;
`seg` space DR 0.058, PLR 0.040, Robust PLR 0.063, ACCEL 0.098.

**What kind of level fails** (random sampler, pooled over algorithms; fail / pass ratio):
walls 1.72x (`dr`) / 1.60x (`seg`), shortest path 1.78x / 1.83x, dead ends 2.08x / 1.62x,
detour (path / Manhattan) 1.55x / 1.58x, but Manhattan distance only 1.25x / 1.30x. Failures are
levels that force a detour around walls, not simply levels with a distant goal.

**Why (train context; H2 = the curriculum saw the region but scored it low):**
- PLR and Robust PLR: failing levels are further from the DR generator's levels (`dr_knn` 15.6
  vs 10.3 for PLR; 12.3 vs 10.1 for Robust PLR) and from the replay buffer (`buf_knn` 15.7 vs
  10.3; 12.3 vs 10.1). Their buffer neighbours have *higher* replay scores (percentile 0.58 vs
  0.46; 0.53 vs 0.47), with buffer age unchanged. The failures are mostly out of distribution,
  and where the buffer does reach them it already ranks them high. H2 is not supported at
  this budget; the gap is generation, not scoring.
- ACCEL: failing levels are slightly *closer* to its buffer (5.4 vs 6.2), whose neighbours are
  fresher (age 15.0k vs 17.6k) and higher-scored (0.63 vs 0.56). ACCEL's edits are already
  working on these regions; at 10% budget it has not learned them yet.
- DR: failures are further from its own generator distribution (`dr_knn` 12.9 vs 10.3), i.e. the
  tail of what DR samples.

Caveats: 10% budget, two training seeds, 1000-level runs. Repeat on the full-budget cluster
checkpoints before writing any of this into section 5.

## 5. Failure atlas (to be filled)

Two columns per cell: **without VerifAI** (the paper's own metrics) and **with VerifAI**
(adaptive falsification, random sampler as the control). "Lit" = reported in prior work, not
yet reproduced by us.

| Algorithm | Without VerifAI | With VerifAI |
|---|---|---|
| DR | Lit: weak on the named held-out mazes (PAIRED, Robust PLR, ACCEL papers). Ours (non-standard harness): held-out 0-0.13 | not run |
| Minimax | Lit: generates unsolvable levels, so the agent learns nothing (PAIRED paper) | not run |
| PAIRED | Lit: protagonist/antagonist/adversary training instability, curriculum collapse (Mediratta+ 2023) | not run |
| PLR / Robust PLR | Lit: score proxies (PVL, MaxMC) track success rate, not regret (SFL paper) | not run |
| ACCEL | Lit: same proxy issue (SFL paper: ACCEL below SFL on maze / JaxNav CVaR) | not run |
| SFL | Lit: scouting compute; CVaR is evaluated on random levels only | not run. Our harness: held-out 0-0.18 (non-standard) |

Hypotheses to test (pre-register before running): (H1) adaptive falsifiers find valid
counterexamples at a higher rate than SFL's random CVaR sampling at equal budget; (H2) the failure
regions found are ones the curriculum's score ranked low (regret proxy ~ success), linking the
failure to the SFL-paper mechanism; (H3) for minimax / PAIRED the failures are concentrated in
structures the adversary never produced.
