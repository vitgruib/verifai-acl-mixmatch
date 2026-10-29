# CartPole bridge: JaxUED PLR training vs our PyTorch training

Both graded by our grader on frozen_sets/suite/cartpole, final check-in at 614400 steps.
Ours (PyTorch): `results/suite/cartpole/scores.csv`. JaxUED: `/Users/ethancai/Projects/Ongoing/ACL27/results/suite/cartpole_jaxued_mac/scores.csv`.

**Verdict: PASS**

| rule | result | detail |
|---|---|---|
| DR verifai within 0.10 | PASS | difference -0.003 |
| PLR verifai within 0.10 | PASS | difference -0.002 |
| RPLR verifai within 0.10 | PASS | difference -0.068 |
| DR random within 0.10 | PASS | difference +0.036 |
| PLR random within 0.10 | PASS | difference +0.085 |
| RPLR random within 0.10 | PASS | difference +0.078 |
| DR > PLR on verifai | PASS | DR 0.568, PLR 0.468 |
| DR > RPLR on verifai | PASS | DR 0.568, RPLR 0.401 |

Mean pass rate per arm and section (n = seeds):

| arm | section | ours | n | JaxUED | n | difference |
|---|---|---|---|---|---|---|
| DR | custom | 0.910 | 20 | 0.952 | 20 | +0.042 |
| DR | random | 0.940 | 20 | 0.976 | 20 | +0.036 |
| DR | verifai | 0.570 | 20 | 0.568 | 20 | -0.003 |
| PLR | custom | 0.872 | 20 | 0.940 | 20 | +0.068 |
| PLR | random | 0.876 | 20 | 0.961 | 20 | +0.085 |
| PLR | verifai | 0.470 | 20 | 0.468 | 20 | -0.002 |
| RPLR | custom | 0.854 | 20 | 0.922 | 20 | +0.068 |
| RPLR | random | 0.826 | 20 | 0.904 | 20 | +0.078 |
| RPLR | verifai | 0.469 | 20 | 0.401 | 20 | -0.068 |
