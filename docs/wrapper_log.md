# Wrapper search: log

Results of the pipeline in `docs/wrapper_methodology.md`, in order. Earlier PLR work is in
`docs/plr.md`. All runs on the fast harness (`acl_bench/plr`), graded on each environment's
exam; "hard" = the VerifAI suite.

## Oracle test (section 3)

Hard-suite final success vs DR (16-24 seeds): half of training (or 20% / 80%) on the hard
suite's own tasks, or half on frontier tasks (search-pool tasks 30-70% of the reference
agents fail; MountainCar has none).

| environment | 20% | 50% | 80% | frontier 50% | set |
|---|---|---|---|---|---|
| CartPole (DR 0.304) | +0.01 | **+0.20** | **+0.37** | **+0.20** | discovery |
| Acrobot (0.084) | -0.02 | -0.02 | -0.05 | -0.03 | no-harm |
| MountainCar (0.000) | 0 | 0 | 0 | n/a | no-harm |
| Pendulum (0.152) | +0.01 | -0.10 | -0.15 | -0.05 | no-harm |

On the no-harm environments the oracles also cost the random suite (up to -0.30 Acrobot,
-0.37 MountainCar, -0.64 Pendulum at 80%). On CartPole, **frontier tasks found by VerifAI
search (not by looking at the exam) work as well as the exam's own tasks**: a wrapper
needs to find the frontier, not the test. Maze calibrated (budget 5.12M: 6 DR runs to 10.2M, random mazes flatten near 0.70, named
mazes stay near 0.05); PointNav calibration running (6 DR runs to 4.9M).

## Offline signal screen (section 4)

See `docs/plr.md`: a pass rate from 2-4 training episodes (overlap@10% 0.25-0.62) beats
PVL everywhere (PVL 0.00-0.02 on Acrobot and MountainCar, worse than random).

## Stage S on CartPole, 24 seeds (advance rule: hard final > DR at p < 0.1, random >= -0.03)

| candidate | hard final | p | random final | advances |
|---|---|---|---|---|
| 1. VerifAI ce steered by bucket learnability, 50% uniform | +0.076 | 0.12 | 0.00 | no |
| 1. VerifAI mab, same | +0.058 | 0.28 | +0.03 | no |
| 1. VerifAI ce, 25% uniform | +0.040 | 0.48 | +0.11 | no |
| 2. PLR buffer ranked by bucket learnability | -0.020 | 0.70 | +0.07 | no |
| 2. PLR buffer ranked by per-level learnability | -0.005 | 0.92 | +0.07 | no |
| *reference: SFL (uncharged scouting)* | *+0.110* | *0.03* | *+0.08* | *(confirmed at 100 seeds)* |

## What SFL costs

Counting its scouting rollouts: a default SFL run on CartPole simulates **~136M scouting
steps against a 614k-step training budget (about 220x)**. Charged to the budget
(`charge_scouting`), the default spends nearly all of it scouting; even 100 tasks x 4
rollouts every 20 updates takes 536k of the 614k steps. SFL's lesson is its signal, not a
cheap method.

## Next

The frontier is where the gains are (oracle), a pass-rate signal finds it (offline screen),
but per-task estimates need many rollouts and per-parameter buckets (VerifAI's samplers)
are too coarse to pin a frontier that depends on combinations of parameters. Candidate 5:
a **pass-rate model** over the task space fit to ordinary training episodes (one per task,
pooled; GoalGAN-style, Florensa et al. 2018), sampling where it predicts p near 0.5. Offline,
it is the best signal in the screen (`model_1ep`, overlap@10% 0.65-0.86 outside late
CartPole; `docs/plr.md`); next is Stage S on CartPole.
