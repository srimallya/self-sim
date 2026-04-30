import argparse
from pathlib import Path

from play_train_nlri import build_runtime_args, run_session


def parse_args():
    parser = argparse.ArgumentParser(description="Compare NLRI ablations under dummy-SDL evaluation.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    ablations = ["full", "no-router", "no-reservoir", "random-policy", "legacy-fallback-only"]
    rows = []
    for index, ablation in enumerate(ablations):
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
                "run_dir": "",
                "quiet": True,
            }
        )
        result = run_session(session_args)
        summary = result["summary"]
        rows.append(
            {
                "ablation": ablation,
                "mean_energy": summary["mean_energy"],
                "food_eaten": summary["total_food_eaten"],
                "mean_leakage": summary["mean_leakage"],
                "collision_count": summary["total_collision_count"],
                "fallback_rate": summary["mean_fallback_rate"],
                "action_entropy": summary["mean_action_entropy"],
            }
        )

    print(
        f"{'ablation':<22} {'energy':>10} {'food':>8} {'leakage':>10} "
        f"{'collisions':>12} {'fallback':>10} {'entropy':>10}"
    )
    for row in rows:
        print(
            f"{row['ablation']:<22} {row['mean_energy']:>10.2f} {row['food_eaten']:>8d} "
            f"{row['mean_leakage']:>10.3f} {row['collision_count']:>12d} "
            f"{row['fallback_rate']:>10.2f} {row['action_entropy']:>10.3f}"
        )


if __name__ == "__main__":
    main()
