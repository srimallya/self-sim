import argparse
from pathlib import Path

from nlri.envs.maze_reservoir_env import MazeReservoirEnv
from nlri.models.nlri_agent import NLRIAgent
from nlri.training.online_trainer import OnlineNLRITrainer
from nlri.training.replay_buffer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="One-button realtime NLRI training.")
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
    parser.add_argument("--buffer-capacity", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def fallback_probability(step: int, warmup_steps: int, disable_after: int):
    if step < warmup_steps:
        return 1.0
    if disable_after <= 0:
        return 0.35
    if step < disable_after:
        return max(0.2, 1.0 - ((step - warmup_steps) / max(1, disable_after - warmup_steps)))
    return 0.05


def format_agent_metrics(step: int, agent_id: int, metrics_row):
    return (
        f"step={step} agent={agent_id} energy={metrics_row['energy']:.1f} "
        f"food={metrics_row['food_eaten']} leakage={metrics_row['reservoir_leakage']:.3f} "
        f"budget={metrics_row['compute_budget']:.3f} z_mean={metrics_row['z_mean']:.3f} "
        f"z_std={metrics_row['z_std']:.3f} entropy={metrics_row['action_entropy']:.3f} "
        f"fallback_rate={metrics_row['fallback_used_rate']:.2f} "
        f"world_loss={_fmt_loss(metrics_row.get('world_prediction_loss'))} "
        f"reservoir_loss={_fmt_loss(metrics_row.get('reservoir_loss'))} "
        f"policy_loss={_fmt_loss(metrics_row.get('policy_loss'))} "
        f"total_loss={_fmt_loss(metrics_row.get('loss'))}"
    )


def _fmt_loss(value):
    return "n/a" if value is None else f"{value:.4f}"


def main():
    args = parse_args()
    max_steps = args.steps if args.steps > 0 else 1_000_000_000
    env = MazeReservoirEnv(render_mode="human", max_steps=max_steps)
    if env.viewer is not None:
        env.viewer.fps = args.fps

    observations, _info = env.reset(seed=args.seed)
    agents = [NLRIAgent(use_legacy_fallback=True) for _ in observations]
    for agent in agents:
        agent.reset_state()

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
        },
    )
    if args.resume:
        loaded = trainer.load_checkpoint(args.checkpoint_dir)
        print(f"resume={'loaded' if loaded else 'not-found'} path={Path(args.checkpoint_dir) / 'latest.pt'}")

    step = 0
    try:
        while True:
            if env.viewer is not None and env.viewer.closed:
                break
            if args.steps > 0 and step >= args.steps:
                break

            actions = []
            debug_rows = []
            current_obs = observations
            fallback_prob = fallback_probability(step, args.warmup_steps, args.disable_fallback_after)
            for agent in agents:
                agent.use_legacy_fallback = True
            for agent, obs in zip(agents, current_obs):
                action, debug = agent.act(obs, fallback_probability=fallback_prob)
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
            trainer.env_step = step + 1

            if step >= args.warmup_steps and step % args.train_every == 0:
                trainer.train_step(batch_size=args.batch_size)

            trainer_metrics = trainer.metrics()
            env.overlay_stats = []
            for agent_id, (agent_info, agent_metrics) in enumerate(zip(info["agents"], trainer_metrics["per_agent"])):
                env.overlay_stats.append(
                    {
                        "color": agent_info["color"],
                        "energy": agent_info["energy"],
                        "leakage": agent_metrics["reservoir_leakage"],
                        "compute_budget": agent_metrics["compute_budget"],
                        "selected_action": agent_metrics["selected_action"],
                        "fallback_used": agent_metrics["fallback_used"],
                        "loss": agent_metrics.get("loss"),
                    }
                )

            step += 1
            observations = next_observations

            if step % 100 == 0:
                for agent_id, row in enumerate(trainer_metrics["per_agent"]):
                    print(format_agent_metrics(step, agent_id, row))

            if step % args.checkpoint_every == 0:
                trainer.save_checkpoint(args.checkpoint_dir, step, periodic=True)

            if terminated or truncated:
                observations, _info = env.reset(seed=args.seed + step)
                for agent in agents:
                    agent.reset_state()

        trainer.save_checkpoint(args.checkpoint_dir, step, periodic=False)
    finally:
        env.close()


if __name__ == "__main__":
    main()
