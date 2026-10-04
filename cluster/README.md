# Running the UED failure atlas on a SLURM cluster

Everything heavy runs here, never on a laptop: paper-length training, PAIRED, and
anything that renders (CarRacing opens pyglet windows unless run under `xvfb-run`).
Results come back as one tarball (`collect.sh`) and are analysed locally.

The suite has four independent parts. Each is one `sbatch` array, and each array task is one
(algorithm, seed). Paper code is used at pinned commits. The only changes are two small JaxUED
patches and two derived DCD configs:

- `patches/jaxued_gif.patch` (cosmetic): logs videos as GIFs, so no ffmpeg is needed.
- `patches/jaxued_minimax.patch`: adds `maze_paired.py --minimax`, which sets the adversary's
  score to the protagonist's negative return. The antagonist still trains; it is ignored.
- `dcd_configs/`: `cr_minimax.json` is `cr_paired.json` with `ued_algo=minimax` and
  `adv_normalize_returns=false`, the same two keys that separate DCD's own bipedal minimax and
  PAIRED configs. `bipedal_plr.json` is `bipedal_robust_plr.json` with exploratory gradient
  updates on.

| part | paper code | algorithms | budget per run | array |
|---|---|---|---|---|
| 1 Maze train + falsify | JaxUED | DR, minimax, PAIRED, PLR, Robust PLR, ACCEL | 30k updates (JaxUED default) | 60 + 60 |
| 2 CarRacing | DCD | DR, minimax, PAIRED, REPAIRED, PLR, Robust PLR | 5.5M steps | 60 |
| 3 BipedalWalker | DCD | DR, minimax, PAIRED, PLR, Robust PLR, ACCEL | 2B steps | 60 |
| 4 SFL repo | sampling-for-learnability | SFL on Maze; DR, PLR, Robust PLR, ACCEL, SFL on JaxNav | paper configs | 60 |

### The algorithm × environment matrix

Every algorithm should be run on every environment. Each cell names the code that trains it.
`port` marks a cell no paper ships code for.

| | Maze | CarRacing | BipedalWalker | JaxNav |
|---|---|---|---|---|
| DR | JaxUED | DCD | DCD | SFL repo |
| minimax | JaxUED + patch | DCD + derived config | DCD | **port** |
| PAIRED | JaxUED | DCD | DCD | **port** |
| PLR | JaxUED | DCD | DCD + derived config | SFL repo |
| Robust PLR | JaxUED | DCD | DCD | SFL repo |
| ACCEL | JaxUED | **port** (DCD has no car-track mutator) | DCD | SFL repo |
| SFL | SFL repo (part 4) | **port** | **port** | SFL repo |

There are five port cells. SFL on CarRacing and Bipedal needs SFL's learnability sampler written
into DCD's PyTorch runner. ACCEL on CarRacing needs a Bezier control-point mutator. Minimax and
PAIRED on JaxNav need an adversary level generator for JaxNav. None of these exist upstream, and
each would be our own implementation rather than paper code, so they are left out until decided.

Two caveats for the Maze row:

- SFL's maze cell is the SFL repo's `minigrid_sfl`, which uses JaxUED's Maze env and the same
  network. Its defaults differ: 60 walls instead of 25, and its own PPO settings.
  `atlas.maze.falsify` loads its `model.safetensors` directly.
- JaxNav cells use the paper's single-agent sweep settings (`env.num_agents=1 env.test_set=single`).

## 0. One-time setup (login node)

```bash
git clone https://github.com/vitgruib/verifai-acl-mixmatch.git atlas && cd atlas
bash cluster/setup/clone.sh        # upstream repos at pinned commits -> third_party/
bash cluster/setup/setup_jax.sh    # .venv-jax: py3.11, jax[cuda12], jaxued, verifai, scenic
bash cluster/setup/setup_dcd.sh    # conda env "dcd": py3.8 + DCD's original requirements
bash cluster/setup/setup_sfl.sh    # .venv-sfl (separate: SFL pins its own jaxmarl)
mkdir -p runs/slurm                # sbatch writes logs here and fails if it is missing
```

Paths and pins live in `cluster/env.sh`. Override any variable (e.g. `RUNS=/scratch/$USER/atlas`)
in your shell before submitting. If your cluster needs `module load cuda` or a partition or account
flag, add it to the `#SBATCH` lines or pass it on the command line (`sbatch -p gpu -A myacct ...`).
wandb runs offline. `wandb sync` afterwards if you want the dashboards.

Smoke-test before the full arrays (minutes, not hours):

```bash
NUM_UPDATES=50 NSEEDS=1 ALGOS=rplr sbatch --array=0 cluster/maze/train.sbatch
# when it finishes:
NUM_UPDATES=50 NSEEDS=1 ALGOS=rplr BUDGET=50 FSEEDS=0 sbatch --array=0 cluster/maze/falsify.sbatch
```

## 1. Maze (JaxUED): train, then falsify

```bash
jid=$(sbatch --parsable --array=0-59 cluster/maze/train.sbatch)
sbatch --array=0-59 --dependency=afterok:$jid cluster/maze/falsify.sbatch
```

- Task `i` is algorithm `ALGOS[i / NSEEDS]` and seed `i % NSEEDS`. The default ALGOS is
  `dr plr rplr accel paired minimax`, with 10 seeds each. Checkpoints go to
  `runs/maze_train/checkpoints/maze_<algo>_n30000/<seed>/`, and JaxUED's own eval on its named
  mazes goes to `runs/maze_train/results/`.
- Falsification runs every `SPACES` × `SAMPLERS` × `FSEEDS` combination on each checkpoint:
  `BUDGET` levels, 10 stochastic attempts each, spec `solve_rate >= 0.5`. Records go to
  `runs/maze_falsify/`. See [docs/failure_records.md](../docs/failure_records.md).
- The sampler defaults come from the local prelim ([docs/ued_atlas.md](../docs/ued_atlas.md),
  section 4b, "Falsifier prelim"): `random ce mab`. Override with `SAMPLERS="random ce"` and similar.
- Cost: about 1,000 falsifier levels take ~45 s on a laptop CPU. One falsify task (2 spaces ×
  3 samplers × 3 seeds × 5000 levels = 90k levels) is about 1 h of laptop-CPU time, well under
  the 8 h limit. Training time has not been
  measured at 30k updates; the 24 h limit is generous. PAIRED and minimax are the slow ones.

## 2. CarRacing (DCD, Robust PLR paper) — Xvfb only

```bash
bash cluster/dcd_cmds.sh car_racing 1 cr_dr cr_minimax cr_plr cr_robust_plr cr_paired cr_repaired > runs/cr_cmds.txt
sbatch --array=0-$(( $(wc -l < runs/cr_cmds.txt) - 1 )) cluster/carracing/train.sbatch runs/cr_cmds.txt
```

`dcd_cmds.sh` expands DCD's own `train_scripts/make_cmd.py` (paper configs; seeds 88–97) into one
command per line and redirects logs to `runs/dcd/<domain>/`. The sbatch refuses commands that
lack `xvfb-run`. Afterwards, DCD's evaluation on the 20 F1 tracks is:
`python -m eval --base_path runs/dcd/car_racing --xpid <xpid> --model_tar model --benchmark f1`.

## 3. BipedalWalker (DCD, ACCEL paper)

```bash
bash cluster/dcd_cmds.sh bipedal 0 bipedal_dr bipedal_minimax bipedal_paired bipedal_plr bipedal_robust_plr bipedal_accel > runs/bip_cmds.txt
sbatch --array=0-$(( $(wc -l < runs/bip_cmds.txt) - 1 )) cluster/bipedal/train.sbatch runs/bip_cmds.txt
```

2B steps take about 63 h at the 8.8k steps/s we measured. The job requeues and DCD resumes from its
checkpoint. Evaluate with `--benchmark bipedal`, which covers v3, Hardcore, Stairs, PitGap,
Stump and Roughness.

## 4. SFL repo (Maze SFL cell, JaxNav row)

```bash
sbatch --array=0-59 cluster/sfl/train.sbatch
```

The default CELLS are `maze:sfl jaxnav:dr jaxnav:plr jaxnav:rplr jaxnav:accel jaxnav:sfl`,
with 10 seeds each. JaxNav DR, PLR, Robust PLR and ACCEL are all `jaxnav_plr` with the
overrides from the paper's `sweep_configs/jaxnav-sa` (the upstream ACCEL sweep has a typo,
`use.USE_ACCEL`; the sbatch uses `ued.USE_ACCEL=True`). Outputs go to
`runs/sfl/<env>_<algo>/s<seed>/model.safetensors`. XLand-MiniGrid lives in
`third_party/sfl/xland` with its own Dockerfile and jax pin, so it is not scripted here. The
hydra overrides were checked against the real configs, but no run has finished end to end, so
run a smoke test first.

## Bring results home

```bash
bash cluster/collect.sh   # -> runs/atlas_collect_<date>.tgz: falsifier records, eval CSVs,
                          #    configs, slurm logs, final maze checkpoints only
```

Then, on the laptop, run `tar xzf` into `runs/` and `python -m atlas.prelim runs/maze_falsify`.

## Status and gaps

- Part 1 is complete. The falsify sbatch was smoke-tested locally against a real checkpoint.
  The train sbatch is a thin wrapper over JaxUED's own script and has not been run as a SLURM job.
- Parts 2–3 train only. Falsification harnesses for CarRacing (a space over Bezier
  control points) and Bipedal (the 8-D terrain parameter vector) are not written yet. Ship the
  training first; checkpoints are what the falsifier needs.
- Part 4 setup is untested (SFL pins `jaxmarl@latest`). Falsifying the Maze SFL cell works
  today. A JaxNav falsifier is not written yet.
- The five port cells in the matrix above are not implemented.
