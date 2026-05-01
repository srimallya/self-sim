import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from play_train_nlri import build_runtime_args, run_session


CONDITIONS = [
    {
        "name": "full",
        "ablation": "full",
        "resume": True,
        "force_no_fallback": False,
        "fallback_prob": 0.0,
        "min_fallback_prob": 0.0,
        "self_distill_weight": 1.0,
        "hybrid_distill_weight": 0.5,
    },
    {
        "name": "full-with-fallback",
        "ablation": "full",
        "resume": True,
        "force_no_fallback": False,
        "fallback_prob": 0.25,
        "min_fallback_prob": 0.0,
        "self_distill_weight": 1.0,
        "hybrid_distill_weight": 0.5,
    },
    {
        "name": "full-no-fallback",
        "ablation": "full",
        "resume": True,
        "force_no_fallback": True,
        "fallback_prob": 0.0,
        "min_fallback_prob": 0.0,
        "self_distill_weight": 1.0,
        "hybrid_distill_weight": 0.5,
    },
    {
        "name": "no-distill",
        "ablation": "full",
        "resume": True,
        "force_no_fallback": True,
        "fallback_prob": 0.0,
        "min_fallback_prob": 0.0,
        "self_distill_weight": 0.0,
        "hybrid_distill_weight": 0.0,
    },
    {
        "name": "no-router",
        "ablation": "no-router",
        "resume": True,
        "force_no_fallback": True,
        "fallback_prob": 0.0,
        "min_fallback_prob": 0.0,
        "self_distill_weight": 1.0,
        "hybrid_distill_weight": 0.5,
    },
    {
        "name": "no-reservoir",
        "ablation": "no-reservoir",
        "resume": True,
        "force_no_fallback": True,
        "fallback_prob": 0.0,
        "min_fallback_prob": 0.0,
        "self_distill_weight": 1.0,
        "hybrid_distill_weight": 0.5,
    },
    {
        "name": "random-policy",
        "ablation": "random-policy",
        "resume": False,
        "force_no_fallback": True,
        "fallback_prob": 0.0,
        "min_fallback_prob": 0.0,
        "self_distill_weight": 0.0,
        "hybrid_distill_weight": 0.0,
    },
    {
        "name": "legacy-fallback-only",
        "ablation": "legacy-fallback-only",
        "resume": False,
        "force_no_fallback": False,
        "fallback_prob": 1.0,
        "min_fallback_prob": 1.0,
        "self_distill_weight": 0.0,
        "hybrid_distill_weight": 0.0,
    },
]


PER_SEED_COLUMNS = [
    "condition",
    "seed",
    "steps",
    "final_energy",
    "food_eaten",
    "mean_leakage",
    "useful_transition_score",
    "collision_count",
    "mean_entropy",
    "fallback_rate",
    "position_novelty",
    "food_per_collision",
    "food_per_100_steps",
    "collisions_per_100_steps",
    "movement_cost_per_food",
    "useful_score_per_100_steps",
    "local_loop_score",
    "wall_contact_rate",
    "hybrid_distill_enabled",
    "hybrid_distill_weight",
    "student_teacher_kl",
    "stagnation_events",
    "survived",
    "no_nan",
    "run_dir",
]


SUMMARY_COLUMNS = [
    "condition",
    "seeds",
    "steps",
    "mean_final_energy",
    "mean_food_eaten",
    "mean_leakage",
    "mean_useful_transition_score",
    "collision_count",
    "mean_entropy",
    "fallback_rate",
    "position_novelty",
    "food_per_collision",
    "food_per_100_steps",
    "collisions_per_100_steps",
    "movement_cost_per_food",
    "useful_score_per_100_steps",
    "local_loop_score",
    "wall_contact_rate",
    "hybrid_distill_enabled",
    "hybrid_distill_weight",
    "student_teacher_kl",
    "stagnation_events",
    "survival_rate",
    "no_nan_rate",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Long-run fixed-seed NLRI evaluation protocol.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--fps", type=int, default=None, help="Defaults to uncapped with --dummy-sdl, otherwise 24.")
    parser.add_argument("--dummy-sdl", action="store_true")
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--stagnation-threshold", type=int, default=500)
    parser.add_argument("--eval-fallback-prob", type=float, default=0.0)
    parser.add_argument("--eval-use-training-fallback-schedule", action="store_true")
    parser.add_argument("--eval-temperature", type=float, default=1.0)
    parser.add_argument("--evolutionary-outer-loop", action="store_true")
    parser.add_argument("--evolution-window", type=int, default=2000)
    parser.add_argument("--evolution-warmup-windows", type=int, default=1)
    parser.add_argument("--evolution-score", type=str, default="food_energy_clean")
    parser.add_argument("--mutation-std", type=float, default=0.005)
    parser.add_argument("--mutation-prob", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dummy_sdl:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    fps = 0 if args.fps is None and args.dummy_sdl else (24 if args.fps is None else args.fps)

    run_dir = Path(args.run_dir) if args.run_dir else Path("runs/nlri_eval") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["conditions"] = [condition["name"] for condition in CONDITIONS]
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    per_seed_rows = []
    for condition in CONDITIONS:
        for offset in range(args.seeds):
            seed = args.seed_start + offset
            session_run_dir = run_dir / "sessions" / condition["name"] / f"seed_{seed}"
            session_args = build_runtime_args(
                {
                    "steps": args.steps,
                    "fps": fps,
                    "seed": seed,
                    "eval": True,
                    "resume": condition["resume"],
                    "checkpoint_path": args.checkpoint,
                    "checkpoint_dir": str(Path(args.checkpoint).parent),
                    "force_no_fallback": condition["force_no_fallback"],
                    "fallback_prob": args.eval_fallback_prob if condition["fallback_prob"] is None else condition["fallback_prob"],
                    "eval_fallback_prob": args.eval_fallback_prob if condition["fallback_prob"] is None else condition["fallback_prob"],
                    "eval_use_training_fallback_schedule": args.eval_use_training_fallback_schedule,
                    "eval_temperature": args.eval_temperature,
                    "evolutionary_outer_loop": args.evolutionary_outer_loop,
                    "evolution_window": args.evolution_window,
                    "evolution_warmup_windows": args.evolution_warmup_windows,
                    "evolution_score": args.evolution_score,
                    "mutation_std": args.mutation_std,
                    "mutation_prob": args.mutation_prob,
                    "fallback_decay": 0.0,
                    "min_fallback_prob": condition["min_fallback_prob"],
                    "ablation": condition["ablation"],
                    "self_distill_weight": condition["self_distill_weight"],
                    "hybrid_distill_weight": condition["hybrid_distill_weight"],
                    "run_dir": str(session_run_dir),
                    "quiet": True,
                    "render_mode": "none" if args.dummy_sdl else "human",
                }
            )
            result = run_session(session_args)
            per_seed_rows.append(row_from_summary(condition, seed, args.steps, result, args.stagnation_threshold))

    summary_rows = summarize(per_seed_rows, args.steps)
    write_csv(run_dir / "per_seed.csv", PER_SEED_COLUMNS, per_seed_rows)
    write_csv(run_dir / "summary.csv", SUMMARY_COLUMNS, summary_rows)
    print_ranked_table(summary_rows)


def row_from_summary(condition, seed, steps, result, stagnation_threshold):
    summary = result["summary"]
    rows = result.get("rows", [])
    final_energy = float(summary.get("mean_energy", 0.0))
    stagnation_events = sum(1 for row in rows if int(row.get("steps_since_food", 0)) >= stagnation_threshold)
    no_nan = no_nan_summary(summary) and no_nan_rows(rows)
    return {
        "condition": condition["name"],
        "seed": seed,
        "steps": steps,
        "final_energy": final_energy,
        "food_eaten": int(summary.get("total_food_eaten", 0)),
        "mean_leakage": float(summary.get("mean_leakage", 0.0)),
        "useful_transition_score": float(summary.get("mean_useful_transition_score", 0.0)),
        "collision_count": int(summary.get("total_collision_count", 0)),
        "mean_entropy": float(summary.get("mean_action_entropy", 0.0)),
        "fallback_rate": float(summary.get("mean_fallback_rate", 0.0)),
        "position_novelty": float(summary.get("mean_position_novelty", 0.0)),
        "food_per_collision": float(summary.get("mean_food_per_collision", 0.0)),
        "food_per_100_steps": float(summary.get("mean_food_per_100_steps", 0.0)),
        "collisions_per_100_steps": float(summary.get("mean_collisions_per_100_steps", 0.0)),
        "movement_cost_per_food": float(summary.get("mean_movement_cost_per_food", 0.0)),
        "useful_score_per_100_steps": float(summary.get("mean_useful_score_per_100_steps", 0.0)),
        "local_loop_score": float(summary.get("mean_local_loop_score", 0.0)),
        "wall_contact_rate": float(summary.get("mean_wall_contact_rate", 0.0)),
        "hybrid_distill_enabled": bool(float(condition.get("hybrid_distill_weight", 0.0)) > 0.0),
        "hybrid_distill_weight": float(condition.get("hybrid_distill_weight", 0.0)),
        "student_teacher_kl": float(summary.get("student_teacher_kl", 0.0)),
        "stagnation_events": int(stagnation_events),
        "survived": bool(final_energy > 0.0),
        "no_nan": bool(no_nan),
        "run_dir": result.get("run_dir", ""),
    }


def summarize(rows, steps):
    by_condition = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)
    summary_rows = []
    for condition, items in by_condition.items():
        summary_rows.append(
            {
                "condition": condition,
                "seeds": len(items),
                "steps": steps,
                "mean_final_energy": mean(items, "final_energy"),
                "mean_food_eaten": mean(items, "food_eaten"),
                "mean_leakage": mean(items, "mean_leakage"),
                "mean_useful_transition_score": mean(items, "useful_transition_score"),
                "collision_count": int(sum(item["collision_count"] for item in items)),
                "mean_entropy": mean(items, "mean_entropy"),
                "fallback_rate": mean(items, "fallback_rate"),
                "position_novelty": mean(items, "position_novelty"),
                "food_per_collision": mean(items, "food_per_collision"),
                "food_per_100_steps": mean(items, "food_per_100_steps"),
                "collisions_per_100_steps": mean(items, "collisions_per_100_steps"),
                "movement_cost_per_food": mean(items, "movement_cost_per_food"),
                "useful_score_per_100_steps": mean(items, "useful_score_per_100_steps"),
                "local_loop_score": mean(items, "local_loop_score"),
                "wall_contact_rate": mean(items, "wall_contact_rate"),
                "hybrid_distill_enabled": mean_bool(items, "hybrid_distill_enabled"),
                "hybrid_distill_weight": mean(items, "hybrid_distill_weight"),
                "student_teacher_kl": mean(items, "student_teacher_kl"),
                "stagnation_events": int(sum(item["stagnation_events"] for item in items)),
                "survival_rate": mean_bool(items, "survived"),
                "no_nan_rate": mean_bool(items, "no_nan"),
            }
        )
    return sorted(summary_rows, key=rank_key, reverse=True)


def rank_key(row):
    return (
        float(row["mean_food_eaten"]),
        float(row["survival_rate"]),
        -float(row["mean_leakage"]),
        float(row["mean_useful_transition_score"]),
    )


def print_ranked_table(rows):
    print(
        f"{'rank':>4} {'condition':<22} {'food':>8} {'survive':>8} {'leakage':>9} "
        f"{'utility':>9} {'coll/100':>8} {'food/coll':>9} {'energy':>9} "
        f"{'entropy':>8} {'fallback':>9} {'nan_ok':>7}"
    )
    for index, row in enumerate(rows, start=1):
        print(
            f"{index:>4d} {row['condition']:<22} {row['mean_food_eaten']:>8.2f} "
            f"{row['survival_rate']:>8.2f} {row['mean_leakage']:>9.3f} "
            f"{row['mean_useful_transition_score']:>9.2f} {row['collisions_per_100_steps']:>8.2f} "
            f"{row['food_per_collision']:>9.3f} {row['mean_final_energy']:>9.2f} "
            f"{row['mean_entropy']:>8.3f} {row['fallback_rate']:>9.2f} {row['no_nan_rate']:>7.2f}"
        )


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(rows, key):
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else 0.0


def mean_bool(rows, key):
    return float(np.mean([1.0 if row[key] else 0.0 for row in rows])) if rows else 0.0


def no_nan_summary(summary):
    def walk(value):
        if isinstance(value, dict):
            return all(walk(item) for item in value.values())
        if isinstance(value, list):
            return all(walk(item) for item in value)
        if isinstance(value, (int, float)):
            return bool(np.isfinite(float(value)))
        return True

    return walk(summary)


def no_nan_rows(rows):
    for row in rows:
        for value in row.values():
            if isinstance(value, (int, float)) and not np.isfinite(float(value)):
                return False
    return True


if __name__ == "__main__":
    main()
