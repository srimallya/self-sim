import argparse
from pathlib import Path

from play_train_nlri import build_runtime_args, run_session


def parse_args():
    parser = argparse.ArgumentParser(description="Compare NLRI ablations under dummy-SDL evaluation.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-no-distill", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    ablations = [("full", "full", 1.0)]
    if args.include_no_distill:
        ablations.append(("full-no-distill", "full", 0.0))
    ablations += [
        ("no-router", "no-router", 1.0),
        ("no-reservoir", "no-reservoir", 1.0),
        ("random-policy", "random-policy", 1.0),
        ("legacy-fallback-only", "legacy-fallback-only", 1.0),
    ]
    rows = []
    for index, (label, ablation, distill_weight) in enumerate(ablations):
        session_args = build_runtime_args(
            {
                "steps": args.steps,
                "fps": args.fps,
                "seed": args.seed + index,
                "eval": True,
                "resume": ablation not in {"random-policy", "legacy-fallback-only"},
                "checkpoint_path": args.checkpoint,
                "checkpoint_dir": str(Path(args.checkpoint).parent),
                "force_no_fallback": ablation != "legacy-fallback-only",
                "fallback_prob": 0.0 if ablation != "legacy-fallback-only" else 1.0,
                "fallback_decay": 0.0,
                "min_fallback_prob": 0.0,
                "ablation": ablation,
                "self_distill_weight": distill_weight,
                "run_dir": "",
                "quiet": True,
            }
        )
        result = run_session(session_args)
        summary = result["summary"]
        rows.append(
            {
                "ablation": label,
                "mean_energy": summary["mean_energy"],
                "food_eaten": summary["total_food_eaten"],
                "mean_leakage": summary["mean_leakage"],
                "collision_count": summary["total_collision_count"],
                "fallback_rate": summary["mean_fallback_rate"],
                "useful_transition_score": summary.get("mean_useful_transition_score", 0.0),
                "world_loss_ema": summary.get("world_loss_ema", 0.0),
                "compute_budget": summary.get("mean_compute_budget", 0.0),
                "z_std": summary.get("latent_diagnostics", {}).get("z_variance", 0.0) ** 0.5,
                "value_loss": summary.get("value_loss", 0.0),
                "distill_loss": summary.get("distill_loss", 0.0),
                "action_histogram_max_fraction": summary.get("mean_action_histogram_max_fraction", 0.0),
                "position_novelty": summary.get("mean_position_novelty", 0.0),
                "steps_since_food": summary.get("max_steps_since_food", 0),
                "skipped_updates": summary.get("skipped_updates", 0),
                "nan_recoveries": summary.get("nan_recoveries", 0),
                "action_entropy": summary["mean_action_entropy"],
                "checkpoint_step": summary.get("checkpoint_step", 0),
            }
        )

    print(
        f"{'ablation':<22} {'energy':>9} {'food':>6} {'leakage':>9} "
        f"{'util':>8} {'world_ema':>10} {'value':>8} {'distill':>8} {'budget':>8} {'z_std':>8} "
        f"{'hist':>7} {'novelty':>8} {'no_food':>8} {'collisions':>11} "
        f"{'fallback':>9} {'entropy':>9} {'skips':>6} {'nan':>4} {'ckpt':>6}"
    )
    for row in rows:
        print(
            f"{row['ablation']:<22} {row['mean_energy']:>9.2f} {row['food_eaten']:>6d} "
            f"{row['mean_leakage']:>9.3f} {row['useful_transition_score']:>8.2f} "
            f"{row['world_loss_ema']:>10.2f} {row['value_loss']:>8.2f} {row['distill_loss']:>8.2f} "
            f"{row['compute_budget']:>8.3f} {row['z_std']:>8.3f} "
            f"{row['action_histogram_max_fraction']:>7.2f} {row['position_novelty']:>8.2f} "
            f"{row['steps_since_food']:>8d} {row['collision_count']:>11d} "
            f"{row['fallback_rate']:>9.2f} {row['action_entropy']:>9.3f} "
            f"{row['skipped_updates']:>6d} {row['nan_recoveries']:>4d} {row['checkpoint_step']:>6d}"
        )


if __name__ == "__main__":
    main()
