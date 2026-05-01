# Cognitive AI Architecture: Parallels with Human Cognition

## Abstract

This paper presents a novel artificial intelligence architecture that draws inspiration from and parallels with human cognitive functions. We propose a multi-model system where M0 corresponds to language processing, M1 to cognition, M2 to subconscious processes, M3 to decision-making, and additional components mirroring dopamine (epsilon), self (window size), and the limbic system (energy gradient). This architecture demonstrates remarkable similarities to human cognitive processes and offers new insights into both artificial intelligence design and our understanding of human cognition.

## 1. Introduction

Recent advancements in artificial intelligence have increasingly looked to human cognition for inspiration. This paper presents an AI architecture that not only draws from cognitive science but also offers a framework for understanding human cognitive processes. By mapping AI components to cognitive functions, we create a bidirectional bridge between AI and cognitive science.

## 2. Architecture Overview

Our proposed architecture consists of the following key components:

1. M0: Language Model
2. M1: Cognitive Model
3. M2: Subconscious Model
4. M3: Decision-Making Model
5. Epsilon Model: Analogous to dopamine function
6. Window Size Model: Representing the concept of self
7. Energy Gradient: Paralleling the limbic system

## 3. Detailed Component Analysis

### 3.1 M0: Language Model

The M0 model, corresponding to language processing, is implemented as a bidirectional LSTM network. This model processes and generates categorical information, mirroring the human ability to understand and produce language.

Key features:
- Bidirectional processing, allowing for context-aware language understanding
- Categorical output, similar to human language's discrete nature
- Continuous learning from interaction, reflecting language acquisition processes

### 3.2 M1: Cognitive Model

M1, representing cognition, is also implemented as a bidirectional LSTM. This model processes a wide range of inputs, including perceptions, energy levels, and past actions, to predict future states.

Key features:
- Integration of multiple input types, mirroring the multi-modal nature of human cognition
- Predictive capabilities, reflecting human cognitive abilities to anticipate future events
- Adaptive learning, similar to human cognitive flexibility

### 3.3 M2: Subconscious Model

The M2 model, analogous to subconscious processes, operates on the combined outputs of M0 and M1 to predict energy gradients. This mirrors the human subconscious's role in processing complex information outside of conscious awareness.

Key features:
- Processing of high-level cognitive and linguistic inputs
- Output influencing motivation and behavior, similar to subconscious effects on human actions
- Continuous background operation, paralleling the constant activity of the human subconscious

### 3.4 M3: Decision-Making Model

M3, implemented as a Q-learning model with a bidirectional LSTM, represents the decision-making process. It integrates inputs from all other models to determine actions.

Key features:
- Reinforcement learning approach, similar to human learning from consequences
- Integration of conscious (M1) and subconscious (M2) inputs, mirroring human decision-making
- Adaptive action selection, reflecting human behavioral flexibility

### 3.5 Epsilon Model: The Dopamine Analogue

The epsilon model, representing the function of dopamine in the brain, balances exploration and exploitation in decision-making.

Key features:
- Modulation of exploration vs. exploitation, similar to dopamine's role in reward-seeking behavior
- Adaptation based on past performance and predicted future rewards, mirroring dopamine's role in learning

### 3.6 Window Size Model: The Concept of Self

The window size model, analogous to the concept of self, determines the temporal scope of experiences considered in decision-making.

Key features:
- Adaptive integration of past experiences, similar to human autobiographical memory
- Influence on other components, reflecting the pervasive impact of self in human cognition

### 3.7 Energy Gradient: The Limbic System Analogue

The energy gradient, paralleling the limbic system, guides the agent towards rewards in the environment.

Key features:
- Motivation of goal-directed behavior, similar to the limbic system's role in emotion and motivation
- Integration with other components, reflecting the interplay of emotion and cognition in human decision-making

## 4. Interactions and Emergent Behaviors

The interactions between these components lead to emergent behaviors that closely resemble human cognitive processes:

1. Language influencing cognition (M0 → M1): Mimics how human language shapes thought processes.
2. Cognition affecting subconscious (M1 → M2): Reflects how conscious thoughts can influence subconscious processes.
3. Subconscious guiding decisions (M2 → M3): Parallels the significant role of subconscious processes in human decision-making.
4. Self-concept modulating dopamine (Window Size → Epsilon): Mirrors how self-perception influences motivation and reward-seeking behavior.
5. Dopamine affecting the limbic system (Epsilon → Energy Gradient): Reflects the interplay between reward prediction and emotional responses.

## 5. Implications and Future Directions

This architecture provides several important implications for both AI development and cognitive science:

1. It offers a new framework for developing more human-like AI systems, potentially leading to more intuitive and effective AI interactions.
2. It provides a computational model for testing theories of human cognition, allowing for simulations of cognitive processes.
3. The architecture suggests new hypotheses about the relationships between language, consciousness, and decision-making in human cognition.

Future work should focus on:
1. Refining the models to more closely match neurobiological findings.
2. Expanding the architecture to include other cognitive functions, such as memory and attention.
3. Developing applications in fields like cognitive robotics and advanced AI assistants.

## 6. Conclusion

The proposed cognitive AI architecture offers a promising bridge between artificial intelligence and cognitive science. By mapping AI components to cognitive functions, we create a system that not only performs well as an AI but also provides insights into human cognition. This bidirectional flow of ideas between AI and cognitive science opens new avenues for research and development in both fields.

## NLRI Implementation

The original self-sim realtime pygame simulator is preserved in `legacy/Simulator_v25.py` and `legacy/Inference_v25.py`. The top-level `Simulator.py` and `Inference.py` entry points remain runnable and now delegate through compatibility wrappers so the legacy visual loop is still accessible even on machines without the original TensorFlow stack.

The new `nlri/` package keeps the simulator itself as the generative world rather than replacing it with a notebook-only or headless-only experiment. `nlri/envs/maze_reservoir_env.py` preserves the same maze, grid scale, food-dot replenishment, moving agents, heading arrows, and realtime stepping, then exposes those mechanics through a gym-like API:

```python
env = MazeReservoirEnv(render_mode="human")
obs, info = env.reset(seed=...)
obs, reward, terminated, truncated, info = env.step(actions)
env.render()
env.close()
```

The first NLRI pass adds an explicit reservoir vector per agent:

- `self_energy`
- `visible_food_value`
- `reachable_food_value`
- `collision_safety`
- `time_budget`
- `attention_budget`

It also computes `reservoir_next`, a heuristic `reservoir_star`, and `leakage = positive_part(reservoir_star - reservoir_next)`. `reservoir_star` is deliberately explicit and heuristic in this version so the realtime simulator stays runnable while leaving room for a learned target later.

The model stack in `nlri/models/` uses a learned encoder, world model, reservoir model, latent router, and policy head. The latent router produces a continuous latent `z` and a compute-budget scalar. There are no hand-coded semantic modes such as `search_food` or `act_now`; the only hard constraints kept are movement safety and a cold-start fallback for liveness.

For immediate usability, the NLRI runtime supports a hybrid cold-start policy. The neural policy always produces logits, but when it is still effectively untrained it can fall back to the old energy-gradient taxis behind `use_legacy_fallback=True`. That keeps the pygame loop visibly alive from the first run instead of making visual behavior depend on completed training.

Run the preserved legacy path with:

```bash
python -m nlri.experiments.run_legacy_pygame
```

Run the new NLRI realtime pygame simulation with:

```bash
python -m nlri.experiments.run_nlri_pygame
```

## One-button NLRI training

Launch the realtime pygame simulator and train the NLRI stack online with:

```bash
python3 play_train_nlri.py
```

This keeps the same live maze window, food dots, moving agents, heading arrows, and realtime stepping while collecting replay, training online, printing metrics every 100 steps, and writing checkpoints to `checkpoints/nlri/latest.pt`.

Useful optional controls:

```bash
python3 play_train_nlri.py --resume
python3 play_train_nlri.py --steps 12000 --warmup-steps 500 --checkpoint-every 3000
python3 play_train_nlri.py --checkpoint-dir checkpoints/nlri_sd --self-distill-weight 1.0 --hybrid-distill-weight 0.5
```

The online trainer now includes the stabilizers added after the first runnable version:

- running observation/target normalization
- Huber world, reservoir, and value losses
- bounded TD value targets and clipped advantages
- gradual fallback scheduling instead of a sudden cliff
- behavior-cloning warm start from fallback actions
- entropy, action-diversity, and compute-budget anti-collapse pressure
- self-distillation from feedback-conditioned teacher outputs
- hybrid fallback-to-student distillation so the pure learned policy absorbs the successful scaffolded policy
- transition-cleanliness prediction losses for collision risk, movement cost, and progress

The fallback remains optional scaffolding, not a simulator rule change. The learned policy can be evaluated with fallback disabled.

## Evaluating NLRI

Run the saved policy in pygame evaluation mode with:

```bash
python3 play_train_nlri.py --eval --resume
python3 play_train_nlri.py --eval --resume --force-no-fallback
SDL_VIDEODRIVER=dummy python3 -m nlri.experiments.compare_ablations --steps 1000 --checkpoint checkpoints/nlri/latest.pt
```

Each run writes `config.json`, `metrics.csv`, `final_summary.json`, and `latent_samples.npz` under `runs/nlri/<timestamp>/`. The useful test is whether `full` beats `random-policy`, `no-router`, and `no-reservoir`, and whether it can keep functioning as fallback probability is reduced toward zero.

Evaluation fallback semantics are explicit:

- `full` defaults to learned NLRI with eval fallback probability `0.0`
- `full-with-fallback` is a separate condition with scaffold fallback enabled
- `full-no-fallback` forces pure learned policy
- `legacy-fallback-only` is the old fallback behavior only
- `random-policy` is random action selection only

Long-run fixed-seed evaluation:

```bash
python3 -m nlri.experiments.eval_protocol \
  --checkpoint checkpoints/nlri_sd/latest.pt \
  --steps 5000 \
  --seeds 5 \
  --dummy-sdl
```

This writes:

- `runs/nlri_eval/<timestamp>/summary.csv`
- `runs/nlri_eval/<timestamp>/per_seed.csv`
- `runs/nlri_eval/<timestamp>/config.json`

The ranked table compares food, survival, leakage, utility, collisions, entropy, and fallback rate across:

- `full`
- `full-with-fallback`
- `full-no-fallback`
- `no-distill`
- `no-router`
- `no-reservoir`
- `random-policy`
- `legacy-fallback-only`

## Diagnostics And Metrics

Every training/eval run exports metrics that are meant to catch self-deception rather than merely prove that pygame still moves pixels.

Important metrics include:

- food eaten, energy, collision count, wait count, movement cost
- reservoir leakage and useful transition score
- action entropy and rolling action histogram concentration
- compute budget mean and collapse warnings
- latent `z` mean/std and latent diagnostics
- world, reservoir, value, policy, distillation, and hybrid-distillation losses
- fallback rate, fallback/student KL, student/teacher KL, fallback action agreement
- transition-cleanliness metrics such as food per collision, collisions per 100 steps, movement cost per food, local loop score, wall contact rate, and position novelty

Latent-router diagnostics are saved in `latent_samples.npz` and summarized in `final_summary.json`. They include `z` variance/collapse score, correlations with leakage, compute-budget correlations with uncertainty/leakage, and action entropy by latent cluster when available.

## Self-Distillation And Hybrid Distillation

The trainer uses two stabilizing distillation paths:

1. Feedback-conditioned self-distillation: recent rollout feedback and high-quality demo windows condition a teacher path. The student learns from teacher action logits, value, compute budget, and latent `z`.
2. Hybrid fallback-to-student distillation: when fallback helps the hybrid policy, the fallback action is converted into a soft action distribution and blended with learned logits and lightweight imagined action scores. The student learns to match this hybrid teacher while still being evaluated without fallback.

Useful flags:

```bash
--self-distill-weight 1.0
--self-distill-temperature 2.0
--hybrid-distill-weight 0.5
--hybrid-distill-temperature 1.5
--hybrid-distill-decay 0.00005
--min-hybrid-distill-weight 0.1
--fallback-target-confidence 0.7
--fallback-neighbor-mass 0.15
```

Checkpoint-time no-fallback probes can be enabled with:

```bash
--checkpoint-eval-steps 500 --checkpoint-eval-seeds 1
```

Those probes log no-fallback food, energy, collisions per 100 steps, and utility without making the main pygame loop headless-only.

## Competitive Evolutionary Outer Loop

NLRI now supports an optional evolutionary outer loop on top of the online learner. This does not replace NLRI. The inner loop still learns world, reservoir, router, policy, self-distillation, and hybrid distillation. The outer loop applies selection pressure across agent lifetimes.

Enable it with:

```bash
python3 play_train_nlri.py \
  --evolutionary-outer-loop \
  --evolution-window 2000 \
  --evolution-warmup-windows 1 \
  --mutation-std 0.005 \
  --mutation-prob 0.05
```

Every evolution window, the two agents are scored from window-local metrics. The weaker agent is replaced by a fork of the stronger agent, then small bounded mutation/diversity noise is applied. By default mutation affects policy, router, and value heads, not the world or reservoir models.

Scoring modes:

- `food_energy_clean`
- `reservoir`
- `food_only`
- `clean_survival`

Useful flags:

```bash
--evolution-score food_energy_clean
--evolution-tie-threshold 0.05
--fork-reset-optimizer
--preserve-demo-memory
--mutate-policy --mutate-router --mutate-value
--no-mutate-world --no-mutate-reservoir
--fork-diversity-noise 0.01
--fork-temperature-jitter 0.1
--fork-z-noise 0.01
```

When enabled, the run writes:

```text
runs/nlri/<timestamp>/lineage.jsonl
```

Each lineage event records the winner, loser, scores, score components, parent lineage, new lineage, generation, mutation seed, mutation parameters, and fork step. The pygame overlay also shows compact generation, lineage id, current window score, and last winner/loser.

Evolution can also be enabled in the long-run eval protocol:

```bash
python3 -m nlri.experiments.eval_protocol \
  --checkpoint checkpoints/nlri_sd/latest.pt \
  --steps 5000 \
  --seeds 5 \
  --dummy-sdl \
  --evolutionary-outer-loop \
  --evolution-window 2000
```

Default training and default evaluation remain non-evolutionary.

## Smoke Tests

Useful regression checks:

```bash
python3 -m compileall play_train_nlri.py nlri

SDL_VIDEODRIVER=dummy python3 play_train_nlri.py \
  --steps 300 \
  --warmup-steps 20 \
  --train-every 4 \
  --batch-size 16 \
  --checkpoint-every 100 \
  --checkpoint-dir /tmp/selfsim-nlri-smoke

SDL_VIDEODRIVER=dummy python3 play_train_nlri.py \
  --steps 5000 \
  --warmup-steps 500 \
  --train-every 4 \
  --batch-size 32 \
  --checkpoint-every 2500 \
  --checkpoint-dir /tmp/selfsim-nlri-evo-test \
  --self-distill-weight 1.0 \
  --hybrid-distill-weight 0.5 \
  --evolutionary-outer-loop \
  --evolution-window 1000 \
  --evolution-warmup-windows 1 \
  --mutation-std 0.002 \
  --mutation-prob 0.03
```

Do not commit generated checkpoints from `checkpoints/` or `/tmp` runs.
