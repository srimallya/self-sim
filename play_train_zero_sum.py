from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from nlri.envs.zero_sum_energy_env import ZeroSumEnergyEnv
from nlri.models.dual_system import DualSystemAgent
from nlri.training import ReplayBuffer
from play_train_dual_system import (
    OnlineDualSystemTrainer,
    entropy,
    fallback_probability,
    fmt_loss,
    safe_mean,
)


CSV_COLUMNS = [
    "step",
    "agent_id",
    "energy",
    "energy_gap",
    "energy_gap_abs",
    "food_eaten",
    "captures",
    "steals",
    "capture_share",
    "energy_transferred",
    "reward",
    "reward_sum",
    "reward_sum_abs",
    "fallback_rate",
    "error_ema",
    "z_std",
    "slow_energy_scale",
    "slow_compute_budget",
    "policy_loss",
    "value_loss",
    "total_loss",
    "lead_changes",
    "winner",
    "stress_score",
    "warnings",
]


def build_parser():
    parser = argparse.ArgumentParser(description="Zero-sum energy duel stress test for dual-system agents.")
    parser.add_argument("--steps", type=int, default=0, help="0 means run until the pygame window closes.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--slow-interval", type=int, default=20)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--buffer-capacity", type=int, default=30000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--policy-lr", type=float, default=5e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/zero_sum")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--force-no-fallback", action="store_true")
    parser.add_argument("--fallback-prob", type=float, default=1.0)
    parser.add_argument("--fallback-decay", type=float, default=0.00025)
    parser.add_argument("--min-fallback-prob", type=float, default=0.15)
    parser.add_argument("--eval-fallback-prob", type=float, default=0.0)
    parser.add_argument("--policy-temperature", type=float, default=1.05)
    parser.add_argument("--eval-temperature", type=float, default=0.55)
    parser.add_argument("--deterministic-eval", action="store_true")
    parser.add_argument("--dummy-sdl", action="store_true")
    parser.add_argument("--render-mode", choices=["human", "none"], default="human")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--food-count", type=int, default=14)
    parser.add_argument("--food-energy-gain", type=float, default=24.0)
    parser.add_argument("--steal-on-contact", type=float, default=8.0)
    parser.add_argument("--scarcity-pressure", type=float, default=0.02)
    parser.add_argument("--stress-preset", choices=["standard", "scarce", "contact", "collapse"], default="standard")
    return parser


def parse_args():
    args = build_parser().parse_args()
    apply_stress_preset(args)
    return args


def apply_stress_preset(args):
    if args.stress_preset == "scarce":
        args.food_count = min(args.food_count, 8)
        args.food_energy_gain = max(args.food_energy_gain, 28.0)
        args.scarcity_pressure = max(args.scarcity_pressure, 0.04)
    elif args.stress_preset == "contact":
        args.food_count = min(args.food_count, 12)
        args.steal_on_contact = max(args.steal_on_contact, 16.0)
        args.scarcity_pressure = max(args.scarcity_pressure, 0.03)
    elif args.stress_preset == "collapse":
        args.food_count = min(args.food_count, 6)
        args.food_energy_gain = max(args.food_energy_gain, 32.0)
        args.steal_on_contact = max(args.steal_on_contact, 18.0)
        args.scarcity_pressure = max(args.scarcity_pressure, 0.06)


def run_session(args):
    if args.dummy_sdl:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        args.render_mode = "none"
    run_dir = make_run_dir(args)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({key: _jsonable(value) for key, value in vars(args).items()}, handle, indent=2, sort_keys=True)

    max_steps = args.steps if args.steps > 0 else 1_000_000_000
    env = ZeroSumEnergyEnv(
        render_mode=None if args.render_mode == "none" else "human",
        max_steps=max_steps,
        food_count=args.food_count,
        food_energy_gain=args.food_energy_gain,
        steal_on_contact=args.steal_on_contact,
        scarcity_pressure=args.scarcity_pressure,
    )
    if env.viewer is not None:
        env.viewer.fps = args.fps
    observations, _ = env.reset(seed=args.seed)
    agents = [DualSystemAgent(use_legacy_fallback=True) for _ in observations]
    for agent in agents:
        agent.reset_state()
        agent.eval() if args.eval else agent.train()
    trainer = OnlineDualSystemTrainer(
        agents,
        ReplayBuffer(capacity=args.buffer_capacity),
        {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "policy_lr": args.policy_lr,
            "grad_clip": args.grad_clip,
            "warmup_steps": args.warmup_steps,
            "eval_mode": args.eval,
        },
    )
    loaded_checkpoint = None
    if args.resume:
        loaded_checkpoint = trainer.load_checkpoint(args.checkpoint_path or checkpoint_dir)
        if args.eval:
            trainer.reset_live_tracking()
        if not args.quiet:
            print(f"resume={'loaded' if loaded_checkpoint else 'not-found'} path={args.checkpoint_path or checkpoint_dir / 'latest.pt'}")

    rows = []
    last_checkpoint = ""
    last_rewards = np.zeros(len(agents), dtype=np.float32)
    step = 0
    try:
        while True:
            if env.viewer is not None and env.viewer.closed:
                break
            if args.steps > 0 and step >= args.steps:
                break

            fallback_prob = fallback_probability(step, args)
            slow_tick = step % max(1, args.slow_interval) == 0
            actions = []
            debug_rows = []
            for agent_id, (agent, obs) in enumerate(zip(agents, observations)):
                agent.use_legacy_fallback = not args.force_no_fallback
                action, debug = agent.act(
                    obs,
                    error_summary=trainer.error_summary(agent_id),
                    reservoir=env.last_info.get("agents", [{}])[agent_id].get("reservoir_next") if env.last_info else {},
                    history_summary=trainer.history_summary(agent_id),
                    slow_tick=slow_tick,
                    deterministic=args.eval and args.deterministic_eval,
                    fallback_probability=fallback_prob,
                    force_no_fallback=args.force_no_fallback,
                    temperature=args.eval_temperature if args.eval else args.policy_temperature,
                )
                actions.append(action)
                debug_rows.append(debug)

            next_observations, rewards, terminated, truncated, info = env.step(actions)
            last_rewards = rewards
            done = bool(terminated or truncated)
            for agent_id, (obs, next_obs, reward, debug, agent_info) in enumerate(
                zip(observations, next_observations, rewards, debug_rows, info["agents"])
            ):
                trainer.observe_transition(
                    agent_id=agent_id,
                    obs=obs,
                    action=actions[agent_id],
                    next_obs=next_obs,
                    reward=float(reward),
                    done=done,
                    agent_info=agent_info,
                    debug=debug,
                    collision=bool(info["collisions"][agent_id]),
                    movement_cost=float(info["movement_costs"][agent_id]),
                    slow_tick=slow_tick,
                )
            trainer.env_step = step + 1

            if not args.eval and step >= args.warmup_steps and step % max(1, args.train_every) == 0:
                trainer.train_step(args.batch_size)

            metrics = trainer.metrics()
            duel = info.get("zero_sum", {})
            env.overlay_stats = build_overlay(info, metrics, duel, args.slow_interval, step)

            step += 1
            observations = next_observations
            if step % 100 == 0:
                snapshot = snapshot_rows(step, metrics, duel, last_rewards)
                rows.extend(snapshot)
                if not args.quiet:
                    for row in snapshot:
                        print(format_metrics(row))
            if not args.eval and step % max(1, args.checkpoint_every) == 0:
                last_checkpoint = trainer.save_checkpoint(checkpoint_dir, step, periodic=True)[0]
            if done:
                observations, _ = env.reset(seed=args.seed + step)
                for agent in agents:
                    agent.reset_state()
    finally:
        env.close()

    final_metrics = trainer.metrics()
    final_duel = getattr(env, "last_info", {}).get("zero_sum", {})
    if step > 0 and (not rows or rows[-1]["step"] != step):
        rows.extend(snapshot_rows(step, final_metrics, final_duel, last_rewards))
    if not args.eval:
        last_checkpoint = trainer.save_checkpoint(checkpoint_dir, step, periodic=False)[0]
    write_metrics_csv(run_dir / "metrics.csv", rows)
    summary = build_final_summary(final_metrics, final_duel, last_checkpoint, loaded_checkpoint, step, rows)
    with (run_dir / "final_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return {"run_dir": str(run_dir), "summary": summary}


def build_overlay(info, metrics, duel, slow_interval, step):
    countdown = (slow_interval - ((step + 1) % slow_interval)) % slow_interval
    overlay = []
    for agent_id, (agent_info, metric_row) in enumerate(zip(info["agents"], metrics["per_agent"])):
        overlay.append(
            {
                "mode": "ZERO SUM",
                "color": agent_info["color"],
                "energy": agent_info["energy"],
                "food_eaten": metric_row["food_eaten"],
                "fallback_rate": metric_row["fallback_rate"],
                "world_error": metric_row["error_ema"],
                "slow_energy_scale": metric_row["slow_energy_scale"],
                "slow_countdown": countdown,
                "slow_goal_probs": metric_row.get("slow_goal_probs"),
                "selected_action": metric_row["selected_action"],
                "energy_gap": duel.get("energy_gap", 0.0),
                "captures": (duel.get("capture_counts") or [0, 0])[agent_id],
                "steals": (duel.get("steal_counts") or [0, 0])[agent_id],
                "lead_changes": duel.get("lead_changes", 0),
                "reward_sum": duel.get("reward_sum", 0.0),
            }
        )
    return overlay


def snapshot_rows(step, metrics, duel, rewards):
    rows = []
    capture_counts = duel.get("capture_counts") or [0 for _ in metrics["per_agent"]]
    steal_counts = duel.get("steal_counts") or [0 for _ in metrics["per_agent"]]
    transferred = duel.get("energy_transferred") or [0.0 for _ in metrics["per_agent"]]
    total_captures = max(1.0, float(sum(capture_counts)))
    winner = int(duel.get("leader")) if duel.get("leader") is not None else -1
    reward_sum = float(duel.get("reward_sum", float(np.sum(rewards))))
    for agent_id, row in enumerate(metrics["per_agent"]):
        captures = int(capture_counts[agent_id]) if agent_id < len(capture_counts) else 0
        steals = int(steal_counts[agent_id]) if agent_id < len(steal_counts) else 0
        energy_gap_abs = float(duel.get("energy_gap_abs", 0.0))
        reward = float(rewards[agent_id]) if agent_id < len(rewards) else 0.0
        rows.append(
            {
                "step": step,
                "agent_id": agent_id,
                "energy": float(row.get("energy", 0.0)),
                "energy_gap": float(duel.get("energy_gap", 0.0)),
                "energy_gap_abs": energy_gap_abs,
                "food_eaten": int(row.get("food_eaten", 0)),
                "captures": captures,
                "steals": steals,
                "capture_share": float(captures / total_captures),
                "energy_transferred": float(transferred[agent_id]) if agent_id < len(transferred) else 0.0,
                "reward": reward,
                "reward_sum": reward_sum,
                "reward_sum_abs": abs(reward_sum),
                "fallback_rate": float(row.get("fallback_rate", 0.0)),
                "error_ema": float(row.get("error_ema", 0.0)),
                "z_std": float(row.get("z_std", 0.0)),
                "slow_energy_scale": float(row.get("slow_energy_scale", 1.0)),
                "slow_compute_budget": float(row.get("slow_compute_budget", 0.0)),
                "policy_loss": row.get("policy_loss"),
                "value_loss": row.get("value_loss"),
                "total_loss": row.get("total_loss"),
                "lead_changes": int(duel.get("lead_changes", 0)),
                "winner": winner,
                "stress_score": stress_score(energy_gap_abs, captures, steals, row.get("fallback_rate", 0.0), abs(reward_sum)),
                "warnings": "|".join(row.get("warning_flags", [])),
            }
        )
    return rows


def stress_score(energy_gap_abs, captures, steals, fallback_rate, reward_sum_abs):
    competition = min(1.0, float(energy_gap_abs) / 500.0)
    contact = min(1.0, float(steals) / 20.0)
    resource = min(1.0, float(captures) / 20.0)
    zero_sum_penalty = min(1.0, float(reward_sum_abs))
    learned_pressure = 1.0 - float(np.clip(fallback_rate, 0.0, 1.0))
    return float(competition + contact + resource + learned_pressure - zero_sum_penalty)


def format_metrics(row):
    return (
        f"step={row['step']} agent={row['agent_id']} energy={row['energy']:.1f} "
        f"gap={row['energy_gap']:.1f} cap={row['captures']} steal={row['steals']} "
        f"share={row['capture_share']:.2f} r={row['reward']:.2f} rsum={row['reward_sum']:.4f} "
        f"fb={row['fallback_rate']:.2f} err={row['error_ema']:.3f} "
        f"stress={row['stress_score']:.2f} loss={fmt_loss(row['total_loss'])}"
    )


def write_metrics_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_final_summary(metrics, duel, checkpoint_path, loaded_checkpoint, step, rows):
    per_agent = metrics["per_agent"]
    return {
        "steps": int(step),
        "winner": int(duel.get("leader")) if duel.get("leader") is not None else -1,
        "energy_gap_abs": float(duel.get("energy_gap_abs", 0.0)),
        "reward_sum_abs_max": max([row.get("reward_sum_abs", 0.0) for row in rows] or [0.0]),
        "lead_changes": int(duel.get("lead_changes", 0)),
        "capture_counts": duel.get("capture_counts", []),
        "steal_counts": duel.get("steal_counts", []),
        "energy_transferred": duel.get("energy_transferred", []),
        "mean_fallback_rate": safe_mean([row.get("fallback_sum", 0.0) / max(1, row.get("count", 0)) for row in per_agent]),
        "mean_error_ema": safe_mean([row.get("error_ema", 0.0) for row in per_agent]),
        "mean_stress_score": safe_mean([row.get("stress_score", 0.0) for row in rows]),
        "checkpoint_path": checkpoint_path or "",
        "loaded_checkpoint": loaded_checkpoint or "",
    }


def make_run_dir(args):
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        mode = "eval" if args.eval else "train"
        run_dir = Path("runs/zero_sum") / f"{stamp}_{args.stress_preset}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    return value


def main():
    run_session(parse_args())


if __name__ == "__main__":
    main()
