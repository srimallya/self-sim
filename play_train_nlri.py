import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nlri.envs.maze_reservoir_env import MazeReservoirEnv
from nlri.models.nlri_agent import NLRIAgent
from nlri.training import LatentDiagnosticsCollector, OnlineNLRITrainer, ReplayBuffer


ABLATIONS = {
    "full",
    "no-router",
    "no-reservoir",
    "no-world",
    "random-policy",
    "legacy-fallback-only",
}

CSV_COLUMNS = [
    "step",
    "agent_id",
    "energy",
    "food_eaten",
    "movement_cost",
    "collision_count",
    "wait_count",
    "leakage",
    "leakage_mean_100",
    "compute_budget",
    "z_mean",
    "z_std",
    "action_entropy",
    "fallback_rate",
    "world_loss",
    "world_loss_ema",
    "reservoir_loss",
    "policy_loss",
    "value_loss",
    "raw_value_loss",
    "clipped_value_loss",
    "value_mean",
    "value_std",
    "value_target_mean",
    "value_target_std",
    "advantage_mean",
    "advantage_std",
    "advantage_max_abs",
    "raw_policy_loss",
    "clipped_policy_loss",
    "entropy_target_loss",
    "action_diversity_loss",
    "action_histogram_max_fraction",
    "position_novelty",
    "steps_since_food",
    "compute_target",
    "distill_loss",
    "teacher_entropy",
    "student_entropy",
    "teacher_student_kl",
    "demo_memory_size",
    "feedback_score",
    "demo_score_mean",
    "food_per_collision",
    "food_per_100_steps",
    "collisions_per_100_steps",
    "movement_cost_per_food",
    "useful_score_per_100_steps",
    "energy_slope",
    "local_loop_score",
    "wall_contact_rate",
    "clean_utility",
    "collision_bce_loss",
    "movement_cost_huber_loss",
    "progress_huber_loss",
    "clean_utility_aux_loss",
    "skipped_updates",
    "nan_recoveries",
    "useful_transition_score",
    "raw_total_loss",
    "total_loss",
    "ablation",
    "eval_mode",
    "warnings",
]


def build_parser():
    parser = argparse.ArgumentParser(description="One-button realtime NLRI training and evaluation.")
    parser.add_argument("--steps", type=int, default=0, help="0 means run until the window closes.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/nlri")
    parser.add_argument("--disable-fallback-after", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--buffer-capacity", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--model-lr", type=float, default=1e-4)
    parser.add_argument("--policy-lr", type=float, default=3e-5)
    parser.add_argument("--value-lr", type=float, default=3e-5)
    parser.add_argument("--fallback-prob", type=float, default=1.0)
    parser.add_argument("--fallback-decay", type=float, default=0.0002)
    parser.add_argument("--min-fallback-prob", type=float, default=0.25)
    parser.add_argument("--force-no-fallback", action="store_true")
    parser.add_argument("--eval-fallback-prob", type=float, default=0.0)
    parser.add_argument("--eval-use-training-fallback-schedule", action="store_true")
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="full")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render-mode", choices=["human", "none"], default="human")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--loss-ema-beta", type=float, default=0.98)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument("--bc-decay", type=float, default=0.0005)
    parser.add_argument("--min-bc-weight", type=float, default=0.05)
    parser.add_argument("--compute-floor", type=float, default=0.05)
    parser.add_argument("--compute-cost-weight", type=float, default=0.01)
    parser.add_argument("--compute-uncertainty-weight", type=float, default=0.05)
    parser.add_argument("--compute-leakage-weight", type=float, default=0.05)
    parser.add_argument("--world-loss-weight", type=float, default=0.2)
    parser.add_argument("--reservoir-loss-weight", type=float, default=1.0)
    parser.add_argument("--policy-loss-weight", type=float, default=0.5)
    parser.add_argument("--latent-loss-weight", type=float, default=0.01)
    parser.add_argument("--compute-loss-weight", type=float, default=0.05)
    parser.add_argument("--utility-aux-weight", type=float, default=0.1)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--entropy-decay", type=float, default=0.0001)
    parser.add_argument("--min-entropy-weight", type=float, default=0.005)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-target", type=float, default=1.0)
    parser.add_argument("--entropy-target-weight", type=float, default=0.08)
    parser.add_argument("--action-diversity-weight", type=float, default=0.04)
    parser.add_argument("--policy-temperature", type=float, default=1.15)
    parser.add_argument("--eval-temperature", type=float, default=0.5)
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--imagined-candidates", type=int, default=4)
    parser.add_argument("--imagined-horizon", type=int, default=3)
    parser.add_argument("--imagined-weight", type=float, default=0.2)
    parser.add_argument("--compute-target-weight", type=float, default=0.05)
    parser.add_argument("--compute-target-floor", type=float, default=0.05)
    parser.add_argument("--compute-target-ceil", type=float, default=0.8)
    parser.add_argument("--utility-clip", type=float, default=5.0)
    parser.add_argument("--return-clip", type=float, default=10.0)
    parser.add_argument("--advantage-clip", type=float, default=5.0)
    parser.add_argument("--value-target-clip", type=float, default=10.0)
    parser.add_argument("--value-huber-delta", type=float, default=1.0)
    parser.add_argument("--policy-loss-clip", type=float, default=10.0)
    parser.add_argument("--value-loss-clip", type=float, default=10.0)
    parser.add_argument("--total-loss-clip", type=float, default=100.0)
    parser.add_argument("--max-entropy-bonus", type=float, default=0.2)
    parser.add_argument("--log-prob-clip", type=float, default=20.0)
    parser.add_argument("--min-action-prob", type=float, default=1e-6)
    parser.add_argument("--total-loss-abs-guard", type=float, default=1000.0)
    parser.add_argument("--grad-abs-guard", type=float, default=100.0)
    parser.add_argument("--nan-recovery-checkpoint", type=str, default="checkpoints/nlri/last_safe.pt")
    parser.add_argument("--self-distill-weight", type=float, default=1.0)
    parser.add_argument("--self-distill-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-ema-rate", type=float, default=0.01)
    parser.add_argument("--feedback-window", type=int, default=100)
    parser.add_argument("--demo-memory-size", type=int, default=128)
    parser.add_argument("--demo-min-score", type=float, default=0.0)
    parser.add_argument("--actor-critic-weight", type=float, default=0.1)
    parser.add_argument("--collision-loss-weight", type=float, default=0.05)
    parser.add_argument("--movement-cost-loss-weight", type=float, default=0.03)
    parser.add_argument("--progress-loss-weight", type=float, default=0.03)
    parser.add_argument("--clean-utility-weight", type=float, default=0.05)
    parser.add_argument("--clean-utility-clip", type=float, default=5.0)
    parser.add_argument("--demo-collision-weight", type=float, default=0.25)
    parser.add_argument("--demo-movement-weight", type=float, default=0.05)
    parser.add_argument("--demo-novelty-weight", type=float, default=0.1)
    return parser


def parse_args():
    return build_parser().parse_args()


def make_run_dir(args):
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        mode = "eval" if args.eval else "train"
        run_dir = Path("runs/nlri") / f"{stamp}_{args.ablation}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def fallback_probability(step: int, args):
    if args.force_no_fallback and args.ablation != "legacy-fallback-only":
        return 0.0
    if args.ablation == "legacy-fallback-only":
        return 1.0
    if args.eval and not args.eval_use_training_fallback_schedule:
        return max(0.0, min(1.0, float(args.eval_fallback_prob)))
    if step < args.warmup_steps:
        return 1.0
    decay_steps = max(0, step - args.warmup_steps)
    fallback_prob = max(args.min_fallback_prob, args.fallback_prob - args.fallback_decay * decay_steps)
    if args.disable_fallback_after > 0 and step >= args.disable_fallback_after:
        fallback_prob = max(0.05, args.min_fallback_prob * 0.5) if not args.eval else 0.0
    return max(0.0, min(1.0, fallback_prob))


def build_agents(observations, args):
    agents = [NLRIAgent(use_legacy_fallback=True, ablation_mode=args.ablation) for _ in observations]
    for agent in agents:
        agent.reset_state()
        agent.compute_floor = args.compute_floor
        if args.eval:
            agent.eval()
        else:
            agent.train()
    return agents


def build_runtime_args(overrides):
    parser = build_parser()
    defaults = vars(parser.parse_args([]))
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def run_session(args):
    run_dir = make_run_dir(args)
    metrics_path = run_dir / "metrics.csv"
    summary_path = run_dir / "final_summary.json"
    latent_path = run_dir / "latent_samples.npz"
    config_path = run_dir / "config.json"
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    config_dict = {key: _jsonable(value) for key, value in vars(args).items()}
    config_dict["run_dir"] = str(run_dir)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config_dict, handle, indent=2, sort_keys=True)

    max_steps = args.steps if args.steps > 0 else 1_000_000_000
    env = MazeReservoirEnv(render_mode=None if args.render_mode == "none" else "human", max_steps=max_steps)
    if env.viewer is not None:
        env.viewer.fps = args.fps

    observations, _info = env.reset(seed=args.seed)
    agents = build_agents(observations, args)
    replay_buffer = ReplayBuffer(capacity=args.buffer_capacity)
    trainer = OnlineNLRITrainer(
        agents=agents,
        replay_buffer=replay_buffer,
        config={
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "model_lr": args.model_lr,
            "policy_lr": args.policy_lr,
            "value_lr": args.value_lr,
            "warmup_steps": args.warmup_steps,
            "checkpoint_dir": args.checkpoint_dir,
            "disable_fallback_after": args.disable_fallback_after,
            "eval_mode": args.eval,
            "ablation": args.ablation,
            "grad_clip": args.grad_clip,
            "loss_ema_beta": args.loss_ema_beta,
            "bc_weight": args.bc_weight,
            "bc_decay": args.bc_decay,
            "min_bc_weight": args.min_bc_weight,
            "compute_floor": args.compute_floor,
            "compute_cost_weight": args.compute_cost_weight,
            "compute_uncertainty_weight": args.compute_uncertainty_weight,
            "compute_leakage_weight": args.compute_leakage_weight,
            "world_loss_weight": args.world_loss_weight,
            "reservoir_loss_weight": args.reservoir_loss_weight,
            "policy_loss_weight": args.policy_loss_weight,
            "latent_loss_weight": args.latent_loss_weight,
            "compute_loss_weight": args.compute_loss_weight,
            "utility_aux_weight": args.utility_aux_weight,
            "entropy_weight": args.entropy_weight,
            "entropy_decay": args.entropy_decay,
            "min_entropy_weight": args.min_entropy_weight,
            "min_fallback_prob": args.min_fallback_prob,
            "value_loss_weight": args.value_loss_weight,
            "gamma": args.gamma,
            "entropy_target": args.entropy_target,
            "entropy_target_weight": args.entropy_target_weight,
            "action_diversity_weight": args.action_diversity_weight,
            "imagined_candidates": args.imagined_candidates,
            "imagined_horizon": args.imagined_horizon,
            "imagined_weight": args.imagined_weight,
            "compute_target_weight": args.compute_target_weight,
            "compute_target_floor": args.compute_target_floor,
            "compute_target_ceil": args.compute_target_ceil,
            "utility_clip": args.utility_clip,
            "return_clip": args.return_clip,
            "advantage_clip": args.advantage_clip,
            "value_target_clip": args.value_target_clip,
            "value_huber_delta": args.value_huber_delta,
            "policy_loss_clip": args.policy_loss_clip,
            "value_loss_clip": args.value_loss_clip,
            "total_loss_clip": args.total_loss_clip,
            "max_entropy_bonus": args.max_entropy_bonus,
            "log_prob_clip": args.log_prob_clip,
            "min_action_prob": args.min_action_prob,
            "total_loss_abs_guard": args.total_loss_abs_guard,
            "grad_abs_guard": args.grad_abs_guard,
            "nan_recovery_checkpoint": args.nan_recovery_checkpoint,
            "self_distill_weight": args.self_distill_weight,
            "self_distill_temperature": args.self_distill_temperature,
            "teacher_ema_rate": args.teacher_ema_rate,
            "feedback_window": args.feedback_window,
            "demo_memory_size": args.demo_memory_size,
            "demo_min_score": args.demo_min_score,
            "actor_critic_weight": args.actor_critic_weight,
            "collision_loss_weight": args.collision_loss_weight,
            "movement_cost_loss_weight": args.movement_cost_loss_weight,
            "progress_loss_weight": args.progress_loss_weight,
            "clean_utility_weight": args.clean_utility_weight,
            "clean_utility_clip": args.clean_utility_clip,
            "demo_collision_weight": args.demo_collision_weight,
            "demo_movement_weight": args.demo_movement_weight,
            "demo_novelty_weight": args.demo_novelty_weight,
        },
    )
    latent_collector = LatentDiagnosticsCollector()
    metrics_rows = []
    last_saved_checkpoint = ""
    checkpoint_step = 0

    loaded_checkpoint = None
    checkpoint_source = args.checkpoint_path or args.checkpoint_dir
    if args.resume:
        loaded_checkpoint = trainer.load_checkpoint(checkpoint_source)
        trainer.reset_live_tracking()
        if loaded_checkpoint:
            checkpoint_step = int(getattr(trainer, "env_step", 0))
        if not args.quiet:
            print(
                f"resume={'loaded' if loaded_checkpoint else 'not-found'} "
                f"path={args.checkpoint_path or (Path(args.checkpoint_dir) / 'latest.pt')}"
            )

    step = 0
    last_recorded_step = 0
    try:
        while True:
            if env.viewer is not None and env.viewer.closed:
                break
            if args.steps > 0 and step >= args.steps:
                break

            actions = []
            debug_rows = []
            current_obs = observations
            fallback_prob = fallback_probability(step, args)

            for agent in agents:
                agent.use_legacy_fallback = args.ablation not in {"random-policy"} and not args.force_no_fallback
            for agent, obs in zip(agents, current_obs):
                action, debug = agent.act(
                    obs,
                    deterministic=args.eval and args.deterministic_eval,
                    fallback_probability=fallback_prob,
                    force_no_fallback=args.force_no_fallback and args.ablation != "legacy-fallback-only",
                    temperature=args.eval_temperature if args.eval else args.policy_temperature,
                )
                actions.append(action)
                debug_rows.append(debug)

            next_observations, rewards, terminated, truncated, info = env.step(actions)

            for agent_id, (obs, next_obs, reward, debug, agent_info) in enumerate(
                zip(current_obs, next_observations, rewards, debug_rows, info["agents"])
            ):
                trainer.observe_transition(
                    agent_id=agent_id,
                    obs=obs,
                    action=actions[agent_id],
                    next_obs=next_obs,
                    reward=float(reward),
                    terminated=bool(terminated or truncated),
                    agent_info=agent_info,
                    debug=debug,
                    collision=bool(info["collisions"][agent_id]),
                    movement_cost=float(info["movement_costs"][agent_id]),
                )
                reservoir_next = agent_info.get("reservoir_next") or {}
                latent_collector.add(
                    z=debug.get("z", []),
                    compute_budget=float(_first_scalar(debug.get("compute_budget", [0.0]))),
                    energy=float(agent_info.get("energy", 0.0)),
                    leakage=float(np_mean_dict(agent_info.get("leakage") or {})),
                    visible_food_value=float(reservoir_next.get("visible_food_value", 0.0)),
                    reachable_food_value=float(reservoir_next.get("reachable_food_value", 0.0)),
                    collision_safety=float(reservoir_next.get("collision_safety", 0.0)),
                    selected_action=int(debug.get("selected_action", 0)),
                    fallback_used=bool(debug.get("fallback_used", False)),
                    uncertainty=float(_first_scalar(debug.get("uncertainty", [0.0]))),
                    action_entropy=float(entropy(debug.get("action_probs"))),
                )
            trainer.env_step = step + 1

            if not args.eval and step >= args.warmup_steps and step % args.train_every == 0:
                trainer.train_step(batch_size=args.batch_size)

            trainer_metrics = trainer.metrics()
            env.overlay_stats = []
            for agent_id, (agent_info, agent_metrics) in enumerate(zip(info["agents"], trainer_metrics["per_agent"])):
                env.overlay_stats.append(
                    {
                        "color": agent_info["color"],
                        "energy": agent_info["energy"],
                        "leakage": agent_metrics["leakage"],
                        "compute_budget": agent_metrics["compute_budget"],
                        "selected_action": agent_metrics["selected_action"],
                        "fallback_used": agent_metrics["fallback_used"],
                        "loss": agent_metrics.get("loss"),
                    }
                )

            step += 1
            observations = next_observations

            if step % 100 == 0:
                rows = snapshot_metrics_rows(step, trainer_metrics, args)
                metrics_rows.extend(rows)
                last_recorded_step = step
                if not args.quiet:
                    for row in rows:
                        print(format_agent_metrics(row))

            if not args.eval and step % args.checkpoint_every == 0:
                saved = trainer.save_checkpoint(args.checkpoint_dir, step, periodic=True)
                last_saved_checkpoint = saved[0]
                checkpoint_step = step

            if terminated or truncated:
                observations, _info = env.reset(seed=args.seed + step)
                for agent in agents:
                    agent.reset_state()

        if step > 0 and last_recorded_step != step:
            rows = snapshot_metrics_rows(step, trainer.metrics(), args)
            metrics_rows.extend(rows)
            if not args.quiet:
                for row in rows:
                    print(format_agent_metrics(row))

        if not args.eval:
            saved = trainer.save_checkpoint(args.checkpoint_dir, step, periodic=False)
            last_saved_checkpoint = saved[0]
            checkpoint_step = step
    finally:
        env.close()

    write_metrics_csv(metrics_path, metrics_rows)
    latent_collector.save(latent_path)
    latent_summary = latent_collector.summary()
    final_summary = build_final_summary(
        trainer.metrics(),
        latent_summary,
        last_saved_checkpoint,
        loaded_checkpoint,
        checkpoint_step,
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2, sort_keys=True)
    if not args.quiet:
        print(json.dumps(final_summary, indent=2, sort_keys=True))

    return {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "summary_path": str(summary_path),
        "latent_path": str(latent_path),
        "summary": final_summary,
        "rows": metrics_rows,
    }


def snapshot_metrics_rows(step, trainer_metrics, args):
    rows = []
    for agent_id, metrics_row in enumerate(trainer_metrics["per_agent"]):
        rows.append(
            {
                "step": step,
                "agent_id": agent_id,
                "energy": float(metrics_row["energy"]),
                "food_eaten": int(metrics_row["food_eaten"]),
                "movement_cost": float(metrics_row["movement_cost"]),
                "collision_count": int(metrics_row["collision_count"]),
                "wait_count": int(metrics_row["wait_count"]),
                "leakage": float(metrics_row["leakage"]),
                "leakage_mean_100": float(metrics_row["leakage_mean_100"]),
                "compute_budget": float(metrics_row["compute_budget"]),
                "z_mean": float(metrics_row["z_mean"]),
                "z_std": float(metrics_row["z_std"]),
                "action_entropy": float(metrics_row["action_entropy"]),
                "fallback_rate": float(metrics_row["fallback_used_rate"]),
                "world_loss": metrics_row.get("world_prediction_loss"),
                "world_loss_ema": metrics_row.get("world_loss_ema"),
                "reservoir_loss": metrics_row.get("reservoir_loss"),
                "policy_loss": metrics_row.get("policy_loss"),
                "value_loss": metrics_row.get("value_loss"),
                "raw_value_loss": metrics_row.get("raw_value_loss"),
                "clipped_value_loss": metrics_row.get("clipped_value_loss"),
                "value_mean": metrics_row.get("value_mean"),
                "value_std": metrics_row.get("value_std"),
                "value_target_mean": metrics_row.get("value_target_mean"),
                "value_target_std": metrics_row.get("value_target_std"),
                "advantage_mean": metrics_row.get("advantage_mean"),
                "advantage_std": metrics_row.get("advantage_std"),
                "advantage_max_abs": metrics_row.get("advantage_max_abs"),
                "raw_policy_loss": metrics_row.get("raw_policy_loss"),
                "clipped_policy_loss": metrics_row.get("clipped_policy_loss"),
                "entropy_target_loss": metrics_row.get("entropy_target_loss"),
                "action_diversity_loss": metrics_row.get("action_diversity_loss"),
                "action_histogram_max_fraction": float(metrics_row["action_histogram_max_fraction"]),
                "position_novelty": float(metrics_row["position_novelty"]),
                "steps_since_food": int(metrics_row["steps_since_food"]),
                "compute_target": float(metrics_row.get("compute_target") or 0.0),
                "distill_loss": metrics_row.get("distill_loss"),
                "teacher_entropy": float(metrics_row.get("teacher_entropy") or 0.0),
                "student_entropy": float(metrics_row.get("student_entropy") or 0.0),
                "teacher_student_kl": float(metrics_row.get("teacher_student_kl") or 0.0),
                "demo_memory_size": int(metrics_row.get("demo_memory_size") or 0),
                "feedback_score": float(metrics_row.get("feedback_score") or 0.0),
                "demo_score_mean": float(metrics_row.get("demo_score_mean") or 0.0),
                "food_per_collision": float(metrics_row.get("food_per_collision") or 0.0),
                "food_per_100_steps": float(metrics_row.get("food_per_100_steps") or 0.0),
                "collisions_per_100_steps": float(metrics_row.get("collisions_per_100_steps") or 0.0),
                "movement_cost_per_food": float(metrics_row.get("movement_cost_per_food") or 0.0),
                "useful_score_per_100_steps": float(metrics_row.get("useful_score_per_100_steps") or 0.0),
                "energy_slope": float(metrics_row.get("energy_slope") or 0.0),
                "local_loop_score": float(metrics_row.get("local_loop_score") or 0.0),
                "wall_contact_rate": float(metrics_row.get("wall_contact_rate") or 0.0),
                "clean_utility": float(metrics_row.get("clean_utility") or 0.0),
                "collision_bce_loss": metrics_row.get("collision_bce_loss"),
                "movement_cost_huber_loss": metrics_row.get("movement_cost_huber_loss"),
                "progress_huber_loss": metrics_row.get("progress_huber_loss"),
                "clean_utility_aux_loss": metrics_row.get("clean_utility_aux_loss"),
                "skipped_updates": int(metrics_row.get("skipped_updates") or 0),
                "nan_recoveries": int(metrics_row.get("nan_recoveries") or 0),
                "useful_transition_score": float(metrics_row["useful_transition_score"]),
                "raw_total_loss": metrics_row.get("raw_total_loss"),
                "total_loss": metrics_row.get("loss"),
                "ablation": args.ablation,
                "eval_mode": bool(args.eval),
                "warnings": "|".join(metrics_row.get("warning_flags", [])),
            }
        )
    return rows


def format_agent_metrics(row):
    warning_suffix = f" warn={row['warnings']}" if row["warnings"] else ""
    return (
        f"step={row['step']} agent={row['agent_id']} energy={row['energy']:.1f} "
        f"food={row['food_eaten']} leakage={row['leakage']:.3f} "
        f"budget={row['compute_budget']:.3f} z_mean={row['z_mean']:.3f} "
        f"z_std={row['z_std']:.3f} entropy={row['action_entropy']:.3f} "
        f"fallback_rate={row['fallback_rate']:.2f} util={row['useful_transition_score']:.2f} "
        f"world_loss={fmt_loss(row['world_loss'])} "
        f"world_ema={fmt_loss(row['world_loss_ema'])} "
        f"reservoir_loss={fmt_loss(row['reservoir_loss'])} "
        f"policy_loss={fmt_loss(row['policy_loss'])} "
        f"value_loss={fmt_loss(row['value_loss'])} "
        f"distill={fmt_loss(row['distill_loss'])} "
        f"adv_max={fmt_loss(row['advantage_max_abs'])} "
        f"hist_max={row['action_histogram_max_fraction']:.2f} "
        f"novelty={row['position_novelty']:.2f} "
        f"no_food={row['steps_since_food']} "
        f"skips={row['skipped_updates']} nan_rec={row['nan_recoveries']} "
        f"total_loss={fmt_loss(row['total_loss'])}{warning_suffix}"
    )


def build_final_summary(trainer_metrics, latent_summary, checkpoint_path, loaded_checkpoint, checkpoint_step):
    per_agent = trainer_metrics["per_agent"]
    mean_energy = safe_mean([row["energy_sum"] / max(1, row["energy_count"]) for row in per_agent])
    total_food_eaten = int(sum(row["food_eaten"] for row in per_agent))
    total_collision_count = int(sum(row["collision_count"] for row in per_agent))
    mean_leakage = safe_mean([row["leakage_sum"] / max(1, row["energy_count"]) for row in per_agent])
    mean_compute_budget = safe_mean([row["compute_budget_sum"] / max(1, row["energy_count"]) for row in per_agent])
    mean_fallback_rate = safe_mean([row["fallback_sum"] / max(1, row["energy_count"]) for row in per_agent])
    mean_action_entropy = safe_mean([row["entropy_sum"] / max(1, row["energy_count"]) for row in per_agent])
    mean_useful_transition_score = safe_mean([row["useful_transition_sum"] / max(1, row["energy_count"]) for row in per_agent])
    mean_hist_max = safe_mean([row["action_histogram_max_fraction"] for row in per_agent])
    mean_position_novelty = safe_mean([row["position_novelty"] for row in per_agent])
    mean_food_per_collision = safe_mean([row.get("food_per_collision", 0.0) for row in per_agent])
    mean_food_per_100 = safe_mean([row.get("food_per_100_steps", 0.0) for row in per_agent])
    mean_collisions_per_100 = safe_mean([row.get("collisions_per_100_steps", 0.0) for row in per_agent])
    mean_movement_cost_per_food = safe_mean([row.get("movement_cost_per_food", 0.0) for row in per_agent])
    mean_useful_score_per_100 = safe_mean([row.get("useful_score_per_100_steps", 0.0) for row in per_agent])
    mean_local_loop_score = safe_mean([row.get("local_loop_score", 0.0) for row in per_agent])
    mean_wall_contact_rate = safe_mean([row.get("wall_contact_rate", 0.0) for row in per_agent])
    max_steps_since_food = int(max([row["steps_since_food"] for row in per_agent] or [0]))
    losses = [row.get("loss") for row in per_agent if row.get("loss") is not None]
    value_losses = [row.get("value_loss") for row in per_agent if row.get("value_loss") is not None]
    distill_losses = [row.get("distill_loss") for row in per_agent if row.get("distill_loss") is not None]
    world_loss_emas = [row.get("world_loss_ema") for row in per_agent if row.get("world_loss_ema") is not None]
    skipped_updates = int(sum(row.get("skipped_updates", 0) or 0 for row in per_agent))
    nan_recoveries = int(sum(row.get("nan_recoveries", 0) or 0 for row in per_agent))
    return {
        "mean_energy": mean_energy,
        "total_food_eaten": total_food_eaten,
        "total_collision_count": total_collision_count,
        "mean_leakage": mean_leakage,
        "mean_compute_budget": mean_compute_budget,
        "mean_fallback_rate": mean_fallback_rate,
        "mean_action_entropy": mean_action_entropy,
        "mean_useful_transition_score": mean_useful_transition_score,
        "mean_action_histogram_max_fraction": mean_hist_max,
        "mean_position_novelty": mean_position_novelty,
        "mean_food_per_collision": mean_food_per_collision,
        "mean_food_per_100_steps": mean_food_per_100,
        "mean_collisions_per_100_steps": mean_collisions_per_100,
        "mean_movement_cost_per_food": mean_movement_cost_per_food,
        "mean_useful_score_per_100_steps": mean_useful_score_per_100,
        "mean_local_loop_score": mean_local_loop_score,
        "mean_wall_contact_rate": mean_wall_contact_rate,
        "max_steps_since_food": max_steps_since_food,
        "final_total_loss": safe_mean(losses),
        "value_loss": safe_mean(value_losses),
        "distill_loss": safe_mean(distill_losses),
        "demo_memory_size": int(max([row.get("demo_memory_size", 0) for row in per_agent] or [0])),
        "demo_score_mean": safe_mean([row.get("demo_score_mean", 0.0) for row in per_agent]),
        "world_loss_ema": safe_mean(world_loss_emas),
        "skipped_updates": skipped_updates,
        "nan_recoveries": nan_recoveries,
        "checkpoint_path": checkpoint_path or "",
        "loaded_checkpoint": loaded_checkpoint or "",
        "checkpoint_step": int(checkpoint_step),
        "latent_diagnostics": latent_summary,
    }


def write_metrics_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            for loss_key in (
                "world_loss",
                "world_loss_ema",
                "reservoir_loss",
                "policy_loss",
                "value_loss",
                "raw_value_loss",
                "clipped_value_loss",
                "value_mean",
                "value_std",
                "value_target_mean",
                "value_target_std",
                "advantage_mean",
                "advantage_std",
                "advantage_max_abs",
                "raw_policy_loss",
                "clipped_policy_loss",
                "entropy_target_loss",
                "action_diversity_loss",
                "distill_loss",
                "teacher_entropy",
                "student_entropy",
                "teacher_student_kl",
                "collision_bce_loss",
                "movement_cost_huber_loss",
                "progress_huber_loss",
                "clean_utility_aux_loss",
                "raw_total_loss",
                "total_loss",
            ):
                value = serialized[loss_key]
                serialized[loss_key] = "" if value is None or not np.isfinite(float(value)) else f"{float(value):.6f}"
            writer.writerow(serialized)


def entropy(probs):
    if probs is None:
        return 0.0
    probs = np.asarray(probs, dtype=np.float32)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(probs.sum())
    if total <= 0.0:
        return 0.0
    probs = probs / total
    return float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())


def _first_scalar(value):
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(np.nan_to_num(array[0], nan=0.0, posinf=0.0, neginf=0.0)) if array.size else 0.0


def np_mean_dict(values):
    if not values:
        return 0.0
    return float(np.mean(list(values.values())))


def safe_mean(values):
    values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def fmt_loss(value):
    return "n/a" if value is None or not np.isfinite(float(value)) else f"{float(value):.4f}"


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    return value


def main():
    run_session(parse_args())


if __name__ == "__main__":
    main()
