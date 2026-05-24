# Dual-System Experiment Summary

Viability formula:

`no_fallback_viability_score = food_score + energy_score + entropy_score - collision_penalty`

- `food_score = min(eval_no_fallback_total_food_eaten / 10, 1)`
- `energy_score = clamp(eval_no_fallback_mean_energy / 1000, 0, 1)`
- `entropy_score = clamp(eval_no_fallback_action_entropy / ln(5), 0, 1)`
- `collision_penalty = min(eval_no_fallback_collisions_per_100_steps / 100, 1)`

| Experiment | Seed | Status | Train fallback | Train err EMA | World100 | Reservoir100 | Goal H | Eval fallback food | Eval no-fallback food | No-fallback energy | Survival | Action H | Policy/fallback agreement | Viability |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `faster_slow_loop` | 1 | ok | 0.040000 | 2.659859 | 2.071560 | 0.084725 | 1.202448 | 91 | 767 | 826.506783 | 1403 | 1.208647 | 0.185000 | 2.162481 |
| `faster_slow_aux` | 1 | ok | 0.020000 | 2.613974 | 1.976407 | 0.089227 | 1.270989 | 63 | 703 | 780.192033 | 1660 | 0.808354 | 0.155000 | 1.782451 |
| `faster_slow_aux_stronger_collision` | 1 | ok | 0.055000 | 2.485840 | 2.054759 | 0.096262 | 1.247870 | 91 | 961 | 864.370800 | 3000 | 0.695218 | 0.305000 | 2.081334 |

The existing analyzer is also run over successful phase directories. Its outputs are under `runs/dual_system_experiments/analysis/` by default.