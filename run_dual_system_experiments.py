from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


EXPERIMENTS = {
    "short_control": {
        "steps": 5000,
        "slow_interval": 25,
        "warmup_steps": 500,
        "min_fallback_prob": 0.25,
        "fallback_decay": None,
    },
    "longer_dual": {
        "steps": 30000,
        "slow_interval": 25,
        "warmup_steps": 1000,
        "min_fallback_prob": 0.05,
        "fallback_decay": 0.00008,
    },
    "faster_slow_loop": {
        "steps": 30000,
        "slow_interval": 10,
        "warmup_steps": 1000,
        "min_fallback_prob": 0.05,
        "fallback_decay": 0.00008,
    },
    "slower_slow_loop": {
        "steps": 30000,
        "slow_interval": 50,
        "warmup_steps": 1000,
        "min_fallback_prob": 0.05,
        "fallback_decay": 0.00008,
    },
    "no_fallback_pressure": {
        "steps": 30000,
        "slow_interval": 25,
        "warmup_steps": 500,
        "min_fallback_prob": 0.0,
        "fallback_decay": 0.00012,
    },
    "faster_slow_aux": {
        "steps": 30000,
        "slow_interval": 10,
        "warmup_steps": 1000,
        "min_fallback_prob": 0.05,
        "fallback_decay": 0.00008,
        "slow_aux_weight": 1.0,
        "slow_goal_shaping_weight": 0.02,
    },
    "faster_slow_aux_stronger_collision": {
        "steps": 30000,
        "slow_interval": 10,
        "warmup_steps": 1000,
        "min_fallback_prob": 0.05,
        "fallback_decay": 0.00008,
        "slow_aux_weight": 1.0,
        "slow_collision_weight": 0.25,
        "slow_goal_shaping_weight": 0.04,
    },
}

PHASES = ("train", "eval_fallback", "eval_no_fallback")
SUMMARY_CSV = Path("dual_system_experiment_summary.csv")
SUMMARY_MD = Path("dual_system_experiment_summary.md")


@dataclass
class PhaseResult:
    experiment: str
    seed: int
    phase: str
    run_dir: Path
    checkpoint_dir: Path
    command: List[str]
    returncode: Optional[int]
    status: str


def build_parser():
    parser = argparse.ArgumentParser(description="Run controlled dual-system train/eval experiments.")
    parser.add_argument("--experiments", default="all", help="all or a comma-separated experiment list.")
    parser.add_argument("--seeds", default="1", help="Comma-separated integer seeds.")
    parser.add_argument("--dummy-sdl", action="store_true", default=True, help="Run pygame with SDL_VIDEODRIVER=dummy.")
    parser.add_argument("--no-dummy-sdl", action="store_false", dest="dummy_sdl", help="Allow the pygame window to open.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write a planned summary without running them.")
    parser.add_argument("--eval-steps", type=int, default=3000)
    parser.add_argument("--checkpoint-root", default="checkpoints/dual_system_experiments")
    parser.add_argument("--run-root", default="runs/dual_system_experiments")
    parser.add_argument("--analysis-output-dir", default="runs/dual_system_experiments/analysis")
    return parser


def main():
    args = build_parser().parse_args()
    experiments = parse_experiments(args.experiments)
    seeds = parse_seeds(args.seeds)
    results: List[PhaseResult] = []

    for experiment in experiments:
        for seed in seeds:
            results.extend(run_experiment(args, experiment, seed))

    successful_run_dirs = [result.run_dir for result in results if result.status == "ok" and (result.run_dir / "metrics.csv").exists()]
    analyze_result = run_analyzer(args, successful_run_dirs)
    if analyze_result is not None:
        results.append(analyze_result)

    if args.dry_run:
        print("dry_run=true")
        return

    write_experiment_csv(SUMMARY_CSV, results, experiments, seeds)
    write_experiment_markdown(SUMMARY_MD, results, experiments, seeds)
    print(f"wrote={SUMMARY_MD}")
    print(f"wrote={SUMMARY_CSV}")


def run_experiment(args, experiment: str, seed: int):
    config = EXPERIMENTS[experiment]
    checkpoint_dir = Path(args.checkpoint_root) / experiment / f"seed_{seed}"
    run_root = Path(args.run_root) / experiment
    commands = [
        (
            "train",
            build_train_command(args, config, seed, checkpoint_dir, run_root / "train" / f"seed_{seed}"),
        ),
        (
            "eval_fallback",
            build_eval_command(args, seed, checkpoint_dir, run_root / "eval_fallback" / f"seed_{seed}", force_no_fallback=False),
        ),
        (
            "eval_no_fallback",
            build_eval_command(args, seed, checkpoint_dir, run_root / "eval_no_fallback" / f"seed_{seed}", force_no_fallback=True),
        ),
    ]
    results = []
    for phase, command in commands:
        run_dir = command_run_dir(command)
        result = run_command(
            args,
            experiment=experiment,
            seed=seed,
            phase=phase,
            checkpoint_dir=checkpoint_dir,
            run_dir=run_dir,
            command=command,
        )
        results.append(result)
    return results


def build_train_command(args, config: Dict[str, object], seed: int, checkpoint_dir: Path, run_dir: Path):
    command = [
        "python3",
        "play_train_dual_system.py",
        "--steps",
        str(config["steps"]),
        "--seed",
        str(seed),
        "--slow-interval",
        str(config["slow_interval"]),
        "--warmup-steps",
        str(config["warmup_steps"]),
        "--min-fallback-prob",
        str(config["min_fallback_prob"]),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--run-dir",
        str(run_dir),
        "--quiet",
    ]
    if config["fallback_decay"] is not None:
        command.extend(["--fallback-decay", str(config["fallback_decay"])])
    optional_args = {
        "slow_aux_weight": "--slow-aux-weight",
        "slow_error_weight": "--slow-error-weight",
        "slow_collision_weight": "--slow-collision-weight",
        "slow_energy_weight": "--slow-energy-weight",
        "slow_goal_shaping_weight": "--slow-goal-shaping-weight",
    }
    for key, flag in optional_args.items():
        if key in config:
            command.extend([flag, str(config[key])])
    if args.dummy_sdl:
        command.append("--dummy-sdl")
    return command


def build_eval_command(args, seed: int, checkpoint_dir: Path, run_dir: Path, force_no_fallback: bool):
    command = [
        "python3",
        "play_train_dual_system.py",
        "--eval",
        "--resume",
        "--steps",
        str(args.eval_steps),
        "--seed",
        str(seed),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--run-dir",
        str(run_dir),
        "--quiet",
    ]
    if force_no_fallback:
        command.append("--force-no-fallback")
    else:
        command.extend(["--eval-fallback-prob", "1.0"])
    if args.dummy_sdl:
        command.append("--dummy-sdl")
    return command


def run_command(args, experiment: str, seed: int, phase: str, checkpoint_dir: Path, run_dir: Path, command: List[str]):
    print(format_command(command))
    if args.dry_run:
        return PhaseResult(experiment, seed, phase, run_dir, checkpoint_dir, command, None, "dry-run")

    if phase.startswith("eval") and not (checkpoint_dir / "latest.pt").exists():
        print(f"missing_checkpoint={checkpoint_dir / 'latest.pt'}")
        return PhaseResult(experiment, seed, phase, run_dir, checkpoint_dir, command, None, "missing-checkpoint")

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.dummy_sdl:
        env.setdefault("SDL_VIDEODRIVER", "dummy")
    completed = subprocess.run(command, env=env)
    status = "ok" if completed.returncode == 0 else "failed"
    return PhaseResult(experiment, seed, phase, run_dir, checkpoint_dir, command, completed.returncode, status)


def run_analyzer(args, run_dirs: List[Path]):
    command = [
        "python3",
        "analyze_dual_system_runs.py",
        "--run-dir",
        str(args.run_root),
        "--output-dir",
        str(args.analysis_output_dir),
    ]
    for run_dir in run_dirs:
        command.extend(["--include", str(run_dir)])
    print(format_command(command))
    if args.dry_run:
        return PhaseResult("all", 0, "analysis", Path(args.analysis_output_dir), Path(""), command, None, "dry-run")
    Path(args.analysis_output_dir).mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command)
    return PhaseResult("all", 0, "analysis", Path(args.analysis_output_dir), Path(""), command, completed.returncode, "ok" if completed.returncode == 0 else "failed")


def command_run_dir(command: List[str]):
    try:
        return Path(command[command.index("--run-dir") + 1])
    except (ValueError, IndexError):
        return Path("")


def write_experiment_csv(path: Path, results: List[PhaseResult], experiments: List[str], seeds: List[int]):
    columns = [
        "experiment",
        "seed",
        "status",
        "train_final_fallback_rate",
        "train_final_error_ema",
        "train_final_mean_world_error_100",
        "train_final_mean_reservoir_error_100",
        "train_final_slow_goal_entropy",
        "eval_fallback_total_food_eaten",
        "eval_no_fallback_total_food_eaten",
        "eval_no_fallback_mean_energy",
        "eval_no_fallback_survival_steps",
        "eval_no_fallback_action_entropy",
        "policy_vs_fallback_agreement",
        "no_fallback_viability_score",
        "food_score",
        "energy_score",
        "entropy_score",
        "collision_penalty",
        "failures",
    ]
    rows = [summary_row(results, experiment, seed) for experiment in experiments for seed in seeds]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_experiment_markdown(path: Path, results: List[PhaseResult], experiments: List[str], seeds: List[int]):
    rows = [summary_row(results, experiment, seed) for experiment in experiments for seed in seeds]
    lines = [
        "# Dual-System Experiment Summary",
        "",
        "Viability formula:",
        "",
        "`no_fallback_viability_score = food_score + energy_score + entropy_score - collision_penalty`",
        "",
        "- `food_score = min(eval_no_fallback_total_food_eaten / 10, 1)`",
        "- `energy_score = clamp(eval_no_fallback_mean_energy / 1000, 0, 1)`",
        "- `entropy_score = clamp(eval_no_fallback_action_entropy / ln(5), 0, 1)`",
        "- `collision_penalty = min(eval_no_fallback_collisions_per_100_steps / 100, 1)`",
        "",
        "| Experiment | Seed | Status | Train fallback | Train err EMA | World100 | Reservoir100 | Goal H | Eval fallback food | Eval no-fallback food | No-fallback energy | Survival | Action H | Policy/fallback agreement | Viability |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['experiment']}` | {row['seed']} | {row['status']} | "
            f"{row['train_final_fallback_rate']} | {row['train_final_error_ema']} | "
            f"{row['train_final_mean_world_error_100']} | {row['train_final_mean_reservoir_error_100']} | "
            f"{row['train_final_slow_goal_entropy']} | {row['eval_fallback_total_food_eaten']} | "
            f"{row['eval_no_fallback_total_food_eaten']} | {row['eval_no_fallback_mean_energy']} | "
            f"{row['eval_no_fallback_survival_steps']} | {row['eval_no_fallback_action_entropy']} | "
            f"{row['policy_vs_fallback_agreement']} | {row['no_fallback_viability_score']} |"
        )
    failures = [row for row in rows if row.get("failures")]
    if failures:
        lines.extend(["", "## Failures", ""])
        for row in failures:
            lines.append(f"- `{row['experiment']}` seed `{row['seed']}`: {row['failures']}")
    lines.extend(
        [
            "",
            "The existing analyzer is also run over successful phase directories. Its outputs are under `runs/dual_system_experiments/analysis/` by default.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_row(results: List[PhaseResult], experiment: str, seed: int):
    phase_results = {result.phase: result for result in results if result.experiment == experiment and result.seed == seed}
    train = phase_results.get("train")
    eval_fallback = phase_results.get("eval_fallback")
    eval_no_fallback = phase_results.get("eval_no_fallback")
    train_last = final_metrics_row(train.run_dir if train else None)
    eval_fallback_summary = final_summary(eval_fallback.run_dir if eval_fallback else None)
    eval_no_fallback_summary = final_summary(eval_no_fallback.run_dir if eval_no_fallback else None)
    eval_no_fallback_last = final_metrics_row(eval_no_fallback.run_dir if eval_no_fallback else None)
    score = viability_score(eval_no_fallback_summary, eval_no_fallback_last)
    failures = [
        f"{phase} rc={result.returncode}"
        for phase, result in phase_results.items()
        if result.status not in ("ok", "dry-run")
    ]
    missing = [phase for phase in PHASES if phase not in phase_results]
    failures.extend(f"missing {phase}" for phase in missing)
    status = "ok" if not failures and all(phase_results.get(phase, PhaseResult("", 0, phase, Path(""), Path(""), [], None, "missing")).status == "ok" for phase in PHASES) else "incomplete"
    if any(result.status == "dry-run" for result in phase_results.values()):
        status = "dry-run"
    return {
        "experiment": experiment,
        "seed": seed,
        "status": status,
        "train_final_fallback_rate": fmt(train_last.get("fallback_rate")),
        "train_final_error_ema": fmt(train_last.get("error_ema")),
        "train_final_mean_world_error_100": fmt(train_last.get("mean_world_error_100")),
        "train_final_mean_reservoir_error_100": fmt(train_last.get("mean_reservoir_error_100")),
        "train_final_slow_goal_entropy": fmt(train_last.get("slow_goal_entropy")),
        "eval_fallback_total_food_eaten": fmt(eval_fallback_summary.get("total_food_eaten"), digits=0),
        "eval_no_fallback_total_food_eaten": fmt(eval_no_fallback_summary.get("total_food_eaten"), digits=0),
        "eval_no_fallback_mean_energy": fmt(eval_no_fallback_summary.get("mean_energy")),
        "eval_no_fallback_survival_steps": fmt(eval_no_fallback_summary.get("survival_steps", eval_no_fallback_summary.get("steps")), digits=0),
        "eval_no_fallback_action_entropy": fmt(eval_no_fallback_last.get("no_fallback_action_entropy")),
        "policy_vs_fallback_agreement": fmt(eval_no_fallback_last.get("policy_vs_fallback_agreement")),
        "no_fallback_viability_score": fmt(score["score"]),
        "food_score": fmt(score["food_score"]),
        "energy_score": fmt(score["energy_score"]),
        "entropy_score": fmt(score["entropy_score"]),
        "collision_penalty": fmt(score["collision_penalty"]),
        "failures": "; ".join(failures),
    }


def viability_score(summary: Dict[str, object], last: Dict[str, object]):
    food_score = clamp(parse_float(summary.get("total_food_eaten")) / 10.0, 0.0, 1.0)
    energy_score = clamp(parse_float(summary.get("mean_energy")) / 1000.0, 0.0, 1.0)
    entropy_score = clamp(parse_float(last.get("no_fallback_action_entropy")) / math.log(5.0), 0.0, 1.0)
    collision_penalty = clamp(parse_float(summary.get("mean_collisions_per_100_steps")) / 100.0, 0.0, 1.0)
    return {
        "score": food_score + energy_score + entropy_score - collision_penalty,
        "food_score": food_score,
        "energy_score": energy_score,
        "entropy_score": entropy_score,
        "collision_penalty": collision_penalty,
    }


def final_metrics_row(run_dir: Optional[Path]):
    if not run_dir:
        return {}
    rows = read_csv(run_dir / "metrics.csv")
    if not rows:
        return {}
    max_step = max(parse_float(row.get("step")) for row in rows)
    final_rows = [row for row in rows if parse_float(row.get("step")) == max_step]
    if not final_rows:
        return {}
    numeric_keys = set().union(*(row.keys() for row in final_rows))
    return {key: mean(row.get(key) for row in final_rows) for key in numeric_keys}


def final_summary(run_dir: Optional[Path]):
    if not run_dir:
        return {}
    path = run_dir / "final_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_experiments(value: str):
    if value == "all":
        return list(EXPERIMENTS.keys())
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiments: {', '.join(unknown)}")
    return names


def parse_seeds(value: str):
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit("--seeds must be a comma-separated list of integers") from exc


def mean(values: Iterable[object]):
    parsed = [parse_float(value, default=float("nan")) for value in values]
    parsed = [value for value in parsed if math.isfinite(value)]
    return float("nan") if not parsed else sum(parsed) / len(parsed)


def parse_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def clamp(value: float, low: float, high: float):
    return max(low, min(high, value))


def fmt(value, digits: int = 6):
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    if digits == 0:
        return str(int(round(number)))
    return f"{number:.{digits}f}"


def format_command(command: List[str]):
    return " ".join(str(part) for part in command)


if __name__ == "__main__":
    main()
