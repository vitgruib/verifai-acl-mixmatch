---
description: Restate the ACL27 goal (UED failure atlas via VerifAI falsification, then falsifier + PLR) and keep iterating toward it
argument-hint: "[optional focus or extra constraint]"
---

# Goal: how UED algorithms fail, and how falsifiers fix it

Set 2026-10-03, replacing the earlier "environment-agnostic curriculum library" goal (its
batches 1-18 and protocol remain the record in `docs/library_log.md`).

1. **Document UED algorithms and their failures.** For each published UED algorithm (DR,
   minimax adversary, PAIRED, PLR, Robust PLR / PLR-perp, ACCEL, SFL; others when warranted),
   write down what it optimizes and its assumptions, then **find concrete failure cases with
   VerifAI falsification**: policies trained by the algorithm, attacked by a falsifier over the
   level space (and start states) inside a Scenic / temporal-logic specification of valid levels.
2. **Explain why they fail.** Each failure gets a mechanism, supported by a diagnostic (e.g. the
   score the algorithm assigned the failing levels during training, how often it ever sampled
   that region, regret-estimate error, unsolvable-level share), not just a story.
3. **Use the exact environments they tested on**, in the papers' conventions, verified against
   each paper before claiming. Start with what exists in the repo:
   - Grid mazes with the literature's held-out mazes (`acl_bench/envs/maze.py`,
     `maze_layouts.HELD_OUT`, from DCD): PAIRED, PLR, Robust PLR, ACCEL, SFL all use them.
   - Then, in rough order of cost: CarRacing with the F1 held-out tracks (Robust PLR / DCD),
     BipedalWalker incl. hardcore (ACCEL), then JaxNav / XLand-MiniGrid (SFL) if feasible.
   Note any deviation from a paper's setup (budget, network, observation) explicitly.
4. **Succeed by combining falsifiers with existing PLR techniques**: e.g. falsifier-found levels
   fed into the PLR/SFL replay buffer, falsifier as the level generator with learnability or
   regret as its objective, spec-constrained validity instead of regret to exclude unsolvable
   levels. Keep the method simple enough to state in a paragraph and implement in
   `acl_bench/curriculum/` / `acl_bench/plr/`.

## Success criterion

- **Failure atlas:** per algorithm x environment, reproducible counterexamples (seed, level,
  spec, falsifier, budget) with a falsification rate over enough training seeds (>= 8 to report,
  more for any comparative claim) and a diagnosed mechanism.
- **The fix:** on the same environments, the combined method reduces the falsification rate
  (equal falsifier budget against every arm) and does not lose on the papers' own held-out
  test sets (e.g. the named mazes), versus the best baseline. Compare at equal training steps;
  report falsifier / scouting compute as a cost. Pre-register the comparison and use fresh seeds
  with multiple-comparison correction for the final claim, as in `docs/protocol.md`.
- Falsification-based evaluation must be fair: same spec, same falsifier and budget for every
  arm, and the falsifier used for evaluation is separate (seeds, or algorithm) from any falsifier
  used inside training.

## Working style

- Clean protocol first. **Register every batch in `docs/library_log.md` before running it.**
- **Self-evaluate before each experiment:** hypothesis, what would falsify it, whether the seed
  count can detect the effect, and whether it is genuinely new rather than a knob toggle.
- **Prod before long runs:** 1-2 seeds, short budgets, unit tests or offline checks first.
- Abandon dead directions early. Record every result and verdict; keep `docs/handoff.md`
  current. The write-up of algorithms, failures and mechanisms lives in `docs/` (one document,
  growing as evidence comes in).

## Operations

- Start by reading `docs/handoff.md`, then the tail of `docs/library_log.md`, and check for
  running jobs with `pgrep -f "name[.]sh"` (plain `name.sh` also matches waiting shells).
- Launch runs with `--allow-battery`; use `--resume` to continue interrupted runs.
- Activate the venv first: `source .venv/bin/activate`.
- Commit and push `wip` after each tested change, then `git push origin wip:main`.

## Current focus

$ARGUMENTS
