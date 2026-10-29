# Batch 21 analysis

Per-case custom scores are the mean share of plays passed, averaged over seeds.

## Screen (one-sided Mann-Whitney, Holm over 12 tests, alpha 0.05)

| method | task | diff | p | Holm p | improved |
|---|---|---|---|---|---|
| PLR | cartpole | -0.100 | 0.9571 | 1.0000 | False |
| RPLR | cartpole | -0.101 | 0.9261 | 1.0000 | False |
| ACCEL | cartpole | -0.036 | 0.9030 | 1.0000 | False |
| SFL | cartpole | -0.239 | 0.9991 | 1.0000 | False |
| PLR | frozenlake | -0.084 | 0.9961 | 1.0000 | False |
| RPLR | frozenlake | -0.097 | 0.9987 | 1.0000 | False |
| ACCEL | frozenlake | -0.109 | 1.0000 | 1.0000 | False |
| SFL | frozenlake | -0.087 | 0.9943 | 1.0000 | False |
| PLR | lunarlander | -0.055 | 0.9904 | 1.0000 | False |
| RPLR | lunarlander | -0.113 | 0.9998 | 1.0000 | False |
| ACCEL | lunarlander | -0.113 | 0.9998 | 1.0000 | False |
| SFL | lunarlander | -0.148 | 0.9995 | 1.0000 | False |

## cartpole: pass rate, mean +- std over 20 seeds

| arm | random | verifai | custom |
|---|---|---|---|
| ACCEL | 0.884 +- 0.251 | 0.535 +- 0.198 | 0.844 +- 0.219 |
| DR | 0.940 +- 0.135 | 0.570 +- 0.188 | 0.910 +- 0.104 |
| PLR | 0.876 +- 0.265 | 0.470 +- 0.219 | 0.872 +- 0.182 |
| RPLR | 0.826 +- 0.344 | 0.469 +- 0.250 | 0.854 +- 0.232 |
| SFL | 0.698 +- 0.421 | 0.332 +- 0.254 | 0.698 +- 0.379 |

## frozenlake: pass rate, mean +- std over 20 seeds

| arm | random | verifai | custom |
|---|---|---|---|
| ACCEL | 0.411 +- 0.087 | 0.032 +- 0.025 | 0.229 +- 0.138 |
| DR | 0.589 +- 0.058 | 0.141 +- 0.094 | 0.286 +- 0.135 |
| PLR | 0.568 +- 0.037 | 0.056 +- 0.033 | 0.424 +- 0.109 |
| RPLR | 0.543 +- 0.030 | 0.043 +- 0.025 | 0.386 +- 0.101 |
| SFL | 0.541 +- 0.035 | 0.054 +- 0.041 | 0.351 +- 0.107 |

## lunarlander: pass rate, mean +- std over 20 seeds

| arm | random | verifai | custom |
|---|---|---|---|
| ACCEL | 0.424 +- 0.251 | 0.141 +- 0.087 | 0.217 +- 0.127 |
| DR | 0.658 +- 0.294 | 0.254 +- 0.133 | 0.406 +- 0.208 |
| PLR | 0.649 +- 0.203 | 0.199 +- 0.071 | 0.350 +- 0.150 |
| RPLR | 0.449 +- 0.292 | 0.141 +- 0.094 | 0.211 +- 0.153 |
| SFL | 0.338 +- 0.357 | 0.106 +- 0.116 | 0.167 +- 0.202 |

## cartpole: custom cases no agent passes

init_range:max+50%

## frozenlake: custom cases no agent passes

default/map1, default/map3, hole_density:max/map2, hole_density:max+beyond(0.32)/map2, slip:max+beyond(0.73)/map1, slip:max+beyond(0.73)/map3, slip:max+beyond(0.9)/map1, slip:max+beyond(0.9)/map2, slip:max+beyond(0.9)/map3, held_out:8x8, held_out:single_path_8x8

## lunarlander: custom cases no agent passes

wind_power:max+50%, turbulence_power:max+50%
