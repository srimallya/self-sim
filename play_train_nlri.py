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
    "reservoir_loss",
    "policy_loss",
    "total_loss",
    "ablation",
    "eval_mode",
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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fallback-prob", type=float, default=1.0)
    parser.add_argument("--fallback-decay", type=float, default=0.0)
    parser.add_argument("--min-fallback-prob", type=float, default=0.0)
    parser.add_argument("--force-no-fallback", action="store_true")
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="full")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
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
    if step < args.warmup_steps:
        return max(args.fallback_prob, args.min_fallback_prob)

    decay_steps = max(0, step - args.warmup_steps)
    fallback_prob = max(args.min_fallback_prob, args.fallback_prob - args.fallback_decay * decay_steps)
    if args.disable_fallback_after > 0 and step >= args.disable_fallback_after:
        fallback_prob = min(fallback_prob, args.min_fallback_prob)
    return max(0.0, min(1.0, fallback_prob))


def build_agents(observations, args):
    agents = [NLRIAgent(use_legacy_fallback=True, ablation_mode=args.ablation) for _ in observations]
    for agent in agents:
        agent.reset_state()
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
    env = MazeReservoirEnv(render_mode="human", max_steps=max_steps)
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
            "warmup_steps": args.warmup_steps,
            "checkpoint_dir": args.checkpoint_dir,
            "disable_fallback_after": args.disable_fallback_after,
            "eval_mode": args.eval,
            "ablation": args.ablation,
            "policy_weight": 0.1,
        },
    )
    latent_collector = LatentDiagnosticsCollector()
    metrics_rows = []
    last_saved_checkpoint = ""

    loaded_checkpoint = None
    checkpoint_source = args.checkpoint_path or args.checkpoint_dir
    if args.resume:
        loaded_checkpoint = trainer.load_checkpoint(checkpoint_source)
        trainer.reset_live_tracking()
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
                    deterministic=args.eval,
                    fallback_probability=fallback_prob,
                    force_no_fallback=args.force_no_fallback and args.ablation != "legacy-fallback-only",
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
    finally:
        env.close()

    write_metrics_csv(metrics_path, metrics_rows)
    latent_collector.save(latent_path)
    latent_summary = latent_collector.summary()
    final_summary = build_final_summary(trainer.metrics(), latent_summary, last_saved_checkpoint, loaded_checkpoint)
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
                "reservoir_loss": metrics_row.get("reservoir_loss"),
                "policy_loss": metrics_row.get("policy_loss"),
                "total_loss": metrics_row.get("loss"),
                "ablation": args.ablation,
                "eval_mode": bool(args.eval),
            }
        )
    return rows


def format_agent_metrics(row):
    return (
        f"step={row['step']} agent={row['agent_id']} energy={row['energy']:.1f} "
        f"food={row['food_eaten']} leakage={row['leakage']:.3f} "
        f"budget={row['compute_budget']:.3f} z_mean={row['z_mean']:.3f} "
        f"z_std={row['z_std']:.3f} entropy={row['action_entropy']:.3f} "
        f"fallback_rate={row['fallback_rate']:.2f} "
        f"world_loss={fmt_loss(row['world_loss'])} "
        f"reservoir_loss={fmt_loss(row['reservoir_loss'])} "
        f"policy_loss={fmt_loss(row['policy_loss'])} "
        f"total_loss={fmt_loss(row['total_loss'])}"
    )


def build_final_summary(trainer_metrics, latent_summary, checkpoint_path, loaded_checkpoint):
    per_agent = trainer_metrics["per_agent"]
    mean_energy = safe_mean([row["energy_sum"] / max(1, row["energy_count"]) for row in per_agent])
    total_food_eaten = int(sum(row["food_eaten"] for row in per_agent))
    total_collision_count = int(sum(row["collision_count"] for row in per_agent))
    mean_leakage = safe_mean([row["leakage_sum"] / max(1, row["energy_count"]) for row in per_agent])
    mean_compute_budget = safe_mean([row["compute_budget_sum"] / max(1, row["energy_count"]) for row in per_agent])
    mean_fallback_rate = safe_mean([row["fallback_sum"] / max(1, row["energy_count"]) for row in per_agent])
    mean_action_entropy = safe_mean([row["entropy_sum"] / max(1, row["energy_count"]) for row in per_agent])
    losses = [row.get("loss") for row in per_agent if row.get("loss") is not None]
    return {
        "mean_energy": mean_energy,
        "total_food_eaten": total_food_eaten,
        "total_collision_count": total_collision_count,
        "mean_leakage": mean_leakage,
        "mean_compute_budget": mean_compute_budget,
        "mean_fallback_rate": mean_fallback_rate,
        "mean_action_entropy": mean_action_entropy,
        "final_total_loss": safe_mean(losses),
        "checkpoint_path": checkpoint_path or "",
        "loaded_checkpoint": loaded_checkpoint or "",
        "latent_diagnostics": latent_summary,
    }


def write_metrics_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            for loss_key in ("world_loss", "reservoir_loss", "policy_loss", "total_loss"):
                serialized[loss_key] = "" if serialized[loss_key] is None else f"{serialized[loss_key]:.6f}"
            writer.writerow(serialized)


def entropy(probs):
    if probs is None:
        return 0.0
    probs = np.asarray(probs, dtype=np.float32)
    return float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())


def _first_scalar(value):
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return float(array[0]) if array.size else 0.0


def np_mean_dict(values):
    if not values:
        return 0.0
    return float(np.mean(list(values.values())))


def safe_mean(values):
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def fmt_loss(value):
    return "n/a" if value is None else f"{float(value):.4f}"


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    return value


def main():
    run_session(parse_args())


if __name__ == "__main__":
    main()
