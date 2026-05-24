# Dual-System NLRI Runtime

`play_train_dual_system.py` adds a runnable dual-system agent on top of the existing pygame maze simulation.

- `SharedEncoder` creates the shared latent state `z_t`.
- `FastSystem` runs every environment step, selects motor logits, and predicts immediate transition signals.
- `SlowSystem` runs every `--slow-interval` steps and holds modulation between ticks.
- `DualSystemAgent` combines them as `final_logits = fast_logits * slow_energy_scale + slow_action_bias`.

The old `play_train_nlri.py` entrypoint is preserved.

## Run

```bash
python3 play_train_dual_system.py
python3 play_train_dual_system.py --resume
python3 play_train_dual_system.py --eval --resume --force-no-fallback
```

Checkpoints are written to `checkpoints/dual_system/latest.pt`. Run artifacts are written to `runs/dual_system/<timestamp>/`.

## Experiment Commands

Training:

```bash
python3 play_train_dual_system.py --steps 20000 --slow-interval 25 --checkpoint-every 2000
```

Evaluation:

```bash
python3 play_train_dual_system.py --eval --resume --force-no-fallback --steps 3000
```

Analysis:

```bash
python3 analyze_dual_system_runs.py --run-dir runs/dual_system --latest 5
```

Controlled experiment harness:

```bash
python3 run_dual_system_experiments.py --experiments all --seeds 1 --dummy-sdl
python3 run_dual_system_experiments.py --experiments longer_dual,faster_slow_loop --seeds 1,2,3 --dummy-sdl
```

The harness trains each selected experiment, evaluates with fallback enabled, evaluates with fallback disabled, then writes `dual_system_experiment_summary.md` and `dual_system_experiment_summary.csv`.

## Smoke Tests

```bash
python3 -m compileall play_train_dual_system.py nlri
SDL_VIDEODRIVER=dummy python3 play_train_dual_system.py --steps 300 --warmup-steps 20 --train-every 4 --batch-size 16 --checkpoint-dir /tmp/selfsim-dual-smoke
python3 play_train_dual_system.py
```

The dummy SDL command should complete without opening pygame.

## Zero-Sum Energy Duel

`play_train_zero_sum.py` runs a separate competitive stress test. It uses the same visual maze and dual-system agents, but food capture transfers energy from the opponent and rewards are centered so each step is zero-sum.

Run:

```bash
python3 play_train_zero_sum.py
python3 play_train_zero_sum.py --stress-preset scarce
python3 play_train_zero_sum.py --stress-preset collapse --steps 20000
python3 play_train_zero_sum.py --eval --resume --force-no-fallback
```

Smoke:

```bash
python3 -m compileall play_train_zero_sum.py nlri
SDL_VIDEODRIVER=dummy python3 play_train_zero_sum.py --steps 300 --warmup-steps 20 --train-every 4 --batch-size 16 --checkpoint-dir /tmp/selfsim-zero-sum-smoke
```

Stress metrics include energy gap, capture share, contact steals, lead changes, reward-sum error, fallback rate, and a compact stress score.
