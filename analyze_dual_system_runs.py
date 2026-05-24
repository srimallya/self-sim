from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LEARNING_METRICS = [
    "total_food_eaten",
    "mean_energy",
    "fallback_rate",
    "error_ema",
    "mean_world_error_100",
    "mean_reservoir_error_100",
    "policy_vs_fallback_agreement",
    "no_fallback_action_entropy",
    "slow_error_loss",
    "slow_collision_loss",
    "slow_energy_loss",
    "slow_goal_shaping_loss",
    "pred_future_error",
    "pred_slow_collision_risk",
    "pred_slow_energy_delta",
]

SLOW_METRICS = [
    "slow_energy_scale",
    "slow_compute_budget",
    "goal_avoid",
    "goal_forage",
    "goal_conserve",
    "goal_explore",
]

GOAL_LABELS = ["avoid", "forage", "conserve", "explore"]
GOAL_COLUMNS = {
    "avoid": "goal_avoid",
    "forage": "goal_forage",
    "conserve": "goal_conserve",
    "explore": "goal_explore",
}

COMPARE_COLUMNS = {
    "collision_rate": "collisions_per_100_steps",
    "energy": "mean_energy",
    "food": "total_food_eaten",
    "prediction_error": "error_ema",
}


@dataclass
class RunAnalysis:
    run_name: str
    run_dir: Path
    rows: List[Dict[str, float]]
    summary: Dict[str, object]
    correlations: Dict[str, float]
    flags: List[str]
    verdict: str
    notes: List[str]


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze dual-system training/eval runs.")
    parser.add_argument("--run-dir", type=str, default="runs/dual_system", help="Directory containing dual-system run folders.")
    parser.add_argument("--latest", type=int, default=0, help="Analyze only the latest N run directories. 0 means all.")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory for dual_system_analysis.md/csv and plots.")
    parser.add_argument("--include", action="append", default=[], help="Specific run directory to include. May be repeated.")
    return parser


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_runs(Path(args.run_dir), args.latest, args.include)
    analyses = [analyze_run(run_dir) for run_dir in run_dirs]
    analyses = [analysis for analysis in analyses if analysis is not None]

    write_analysis_csv(output_dir / "dual_system_analysis.csv", analyses)
    write_markdown(output_dir / "dual_system_analysis.md", analyses, run_dirs)
    write_plots(plot_dir, analyses)

    print(f"analyzed_runs={len(analyses)}")
    print(f"wrote={output_dir / 'dual_system_analysis.md'}")
    print(f"wrote={output_dir / 'dual_system_analysis.csv'}")
    print(f"plots={plot_dir}")


def discover_runs(base_dir: Path, latest: int, include: List[str]):
    explicit = [Path(path) for path in include]
    discovered = []
    if base_dir.exists():
        discovered = [path for path in base_dir.iterdir() if path.is_dir() and (path / "metrics.csv").exists()]
    run_dirs = explicit or discovered
    run_dirs = [path for path in run_dirs if (path / "metrics.csv").exists()]
    run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if latest and latest > 0:
        run_dirs = run_dirs[:latest]
    return run_dirs


def analyze_run(run_dir: Path) -> Optional[RunAnalysis]:
    raw_rows = read_metrics(run_dir / "metrics.csv")
    if not raw_rows:
        return None
    rows = aggregate_by_step(raw_rows)
    summary = read_json(run_dir / "final_summary.json")
    correlations = compute_correlations(rows)
    flags = collapse_flags(rows, summary)
    notes = quality_notes(rows, summary, correlations)
    verdict = classify_verdict(rows, summary, flags, notes)
    return RunAnalysis(
        run_name=run_dir.name,
        run_dir=run_dir,
        rows=rows,
        summary=summary,
        correlations=correlations,
        flags=flags,
        verdict=verdict,
        notes=notes,
    )


def read_metrics(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def read_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError:
            return {}


def aggregate_by_step(raw_rows: List[Dict[str, str]]):
    grouped: Dict[int, List[Dict[str, str]]] = {}
    for row in raw_rows:
        step = int(parse_float(row.get("step"), default=0.0))
        grouped.setdefault(step, []).append(row)

    aggregated = []
    for step in sorted(grouped):
        rows = grouped[step]
        total_food = sum(parse_float(row.get("food_eaten")) for row in rows)
        agg = {
            "step": float(step),
            "total_food_eaten": total_food,
            "mean_energy": mean(row.get("energy") for row in rows),
            "fallback_rate": mean(row.get("fallback_rate") for row in rows),
            "error_ema": mean(row.get("error_ema") for row in rows),
            "world_error": mean(row.get("world_error") for row in rows),
            "reservoir_error": mean(row.get("reservoir_error") for row in rows),
            "mean_world_error_100": mean(row.get("mean_world_error_100") for row in rows),
            "mean_reservoir_error_100": mean(row.get("mean_reservoir_error_100") for row in rows),
            "policy_vs_fallback_agreement": mean(row.get("policy_vs_fallback_agreement") for row in rows),
            "no_fallback_action_entropy": mean(row.get("no_fallback_action_entropy") for row in rows),
            "slow_error_loss": mean(row.get("slow_error_loss") for row in rows),
            "slow_collision_loss": mean(row.get("slow_collision_loss") for row in rows),
            "slow_energy_loss": mean(row.get("slow_energy_loss") for row in rows),
            "slow_goal_shaping_loss": mean(row.get("slow_goal_shaping_loss") for row in rows),
            "pred_future_error": mean(row.get("pred_future_error") for row in rows),
            "pred_slow_collision_risk": mean(row.get("pred_slow_collision_risk") for row in rows),
            "pred_slow_energy_delta": mean(row.get("pred_slow_energy_delta") for row in rows),
            "slow_energy_scale": mean(row.get("slow_energy_scale") for row in rows),
            "slow_compute_budget": mean(row.get("slow_compute_budget") for row in rows),
            "slow_goal_entropy": mean(row.get("slow_goal_entropy") for row in rows),
            "z_std": mean(row.get("z_std") for row in rows),
            "z_std_min": min_value(row.get("z_std") for row in rows),
            "collisions_per_100_steps": mean(row.get("collisions_per_100_steps") for row in rows),
            "entropy": mean(row.get("entropy") for row in rows),
            "slow_tick_count": max_value(row.get("slow_tick_count") for row in rows),
        }
        for label, column in GOAL_COLUMNS.items():
            agg[column] = mean(row.get(column) for row in rows)
        aggregated.append(agg)
    return aggregated


def compute_correlations(rows: List[Dict[str, float]]):
    correlations = {}
    for label, goal_column in GOAL_COLUMNS.items():
        for compare_name, compare_column in COMPARE_COLUMNS.items():
            correlations[f"{goal_column}_vs_{compare_name}"] = corr(series(rows, goal_column), series(rows, compare_column))
    correlations["error_vs_collision_rate"] = corr(series(rows, "error_ema"), series(rows, "collisions_per_100_steps"))
    correlations["error_vs_energy"] = corr(series(rows, "error_ema"), series(rows, "mean_energy"))
    correlations["error_vs_energy_drop"] = corr(series(rows, "error_ema"), negative_deltas(series(rows, "mean_energy")))
    correlations["error_delta_vs_slow_energy_delta"] = corr(deltas(series(rows, "error_ema")), deltas(series(rows, "slow_energy_scale")))
    correlations["error_delta_vs_compute_delta"] = corr(deltas(series(rows, "error_ema")), deltas(series(rows, "slow_compute_budget")))
    return correlations


def collapse_flags(rows: List[Dict[str, float]], summary: Dict[str, object]):
    flags = []
    if any(is_finite(row.get("z_std_min")) and row.get("z_std_min", 1.0) < 0.01 for row in rows):
        flags.append("latent_z_std_below_0.01")
    action_entropy = choose_available_series(rows, ["no_fallback_action_entropy", "entropy"])
    if finite_count(action_entropy) and safe_nanmean(action_entropy) < 0.05:
        flags.append("action_entropy_near_zero")
    late = late_rows(rows)
    if late and safe_nanmean(series(late, "fallback_rate")) > 0.8:
        flags.append("fallback_rate_above_0.8_after_warmup")
    eval_poor = is_eval_poor(summary)
    if late and safe_nanmean(series(late, "policy_vs_fallback_agreement")) > 0.95 and eval_poor:
        flags.append("policy_copies_fallback_with_poor_eval")
    goal_entropy = series(rows, "slow_goal_entropy")
    if finite_count(goal_entropy) and safe_nanstd(goal_entropy) < 1e-3:
        flags.append("slow_goal_entropy_flat")
    slow_energy = series(rows, "slow_energy_scale")
    if finite_count(slow_energy) and safe_nanstd(slow_energy) < 1e-3:
        flags.append("slow_energy_scale_constant")
    return flags


def quality_notes(rows: List[Dict[str, float]], summary: Dict[str, object], correlations: Dict[str, float]):
    notes = []
    if improves(rows, "mean_world_error_100"):
        notes.append("world_error_decreases_or_stabilizes")
    else:
        notes.append("world_error_not_improving")
    if improves(rows, "mean_reservoir_error_100"):
        notes.append("reservoir_error_decreases_or_stabilizes")
    else:
        notes.append("reservoir_error_not_improving")
    if declines(rows, "fallback_rate"):
        notes.append("fallback_rate_declines")
    else:
        notes.append("fallback_rate_not_declining")
    if safe_nanmean(series(late_rows(rows), "no_fallback_action_entropy")) > 0.1:
        notes.append("no_fallback_entropy_nonzero")
    else:
        notes.append("no_fallback_entropy_weak")
    if safe_nanstd(series(rows, "slow_energy_scale")) > 1e-3 or safe_nanstd(series(rows, "slow_compute_budget")) > 1e-3:
        notes.append("slow_modulation_nontrivial")
    else:
        notes.append("slow_modulation_flat")
    if correlations.get("error_vs_collision_rate", 0.0) > 0.2:
        notes.append("error_spikes_correlate_with_collisions")
    if correlations.get("error_vs_energy_drop", 0.0) > 0.2:
        notes.append("error_spikes_correlate_with_energy_drops")
    response = modulation_response_score(rows)
    if response > 0.25:
        notes.append("slow_modulation_changes_after_error_rise")
    elif math.isfinite(response):
        notes.append("slow_modulation_response_to_error_rise_weak")
    if is_eval_run(summary) and not is_eval_poor(summary):
        notes.append("no_fallback_eval_remains_alive")
    elif is_eval_run(summary):
        notes.append("no_fallback_eval_weak")
    return notes


def classify_verdict(rows: List[Dict[str, float]], summary: Dict[str, object], flags: List[str], notes: List[str]):
    if any(flag in flags for flag in ("latent_z_std_below_0.01", "action_entropy_near_zero")):
        return "FAIL"
    if "fallback_rate_above_0.8_after_warmup" in flags and "no_fallback_eval_weak" in notes:
        return "FAIL"
    positive = {
        "world_error_decreases_or_stabilizes",
        "reservoir_error_decreases_or_stabilizes",
        "fallback_rate_declines",
        "no_fallback_entropy_nonzero",
        "slow_modulation_nontrivial",
        "no_fallback_eval_remains_alive",
    }
    score = sum(1 for note in notes if note in positive)
    if score >= 5 and not flags:
        return "PASS"
    if score >= 3 or not flags:
        return "PARTIAL"
    return "FAIL"


def write_analysis_csv(path: Path, analyses: List[RunAnalysis]):
    columns = [
        "run_name",
        "run_dir",
        "verdict",
        "steps",
        "final_total_food_eaten",
        "final_mean_energy",
        "final_fallback_rate",
        "final_error_ema",
        "final_mean_world_error_100",
        "final_mean_reservoir_error_100",
        "final_no_fallback_action_entropy",
        "final_policy_vs_fallback_agreement",
        "final_slow_error_loss",
        "final_slow_collision_loss",
        "final_slow_energy_loss",
        "final_slow_goal_shaping_loss",
        "final_pred_future_error",
        "final_pred_slow_collision_risk",
        "final_pred_slow_energy_delta",
        "slow_energy_scale_std",
        "slow_compute_budget_std",
        "goal_entropy_std",
        "error_vs_collision_rate",
        "error_vs_energy_drop",
        "error_delta_vs_slow_energy_delta",
        "flags",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for analysis in analyses:
            last = analysis.rows[-1] if analysis.rows else {}
            writer.writerow(
                {
                    "run_name": analysis.run_name,
                    "run_dir": str(analysis.run_dir),
                    "verdict": analysis.verdict,
                    "steps": int(last.get("step", 0)),
                    "final_total_food_eaten": fmt_float(last.get("total_food_eaten")),
                    "final_mean_energy": fmt_float(last.get("mean_energy")),
                    "final_fallback_rate": fmt_float(last.get("fallback_rate")),
                    "final_error_ema": fmt_float(last.get("error_ema")),
                    "final_mean_world_error_100": fmt_float(last.get("mean_world_error_100")),
                    "final_mean_reservoir_error_100": fmt_float(last.get("mean_reservoir_error_100")),
                    "final_no_fallback_action_entropy": fmt_float(last.get("no_fallback_action_entropy")),
                    "final_policy_vs_fallback_agreement": fmt_float(last.get("policy_vs_fallback_agreement")),
                    "final_slow_error_loss": fmt_float(last.get("slow_error_loss")),
                    "final_slow_collision_loss": fmt_float(last.get("slow_collision_loss")),
                    "final_slow_energy_loss": fmt_float(last.get("slow_energy_loss")),
                    "final_slow_goal_shaping_loss": fmt_float(last.get("slow_goal_shaping_loss")),
                    "final_pred_future_error": fmt_float(last.get("pred_future_error")),
                    "final_pred_slow_collision_risk": fmt_float(last.get("pred_slow_collision_risk")),
                    "final_pred_slow_energy_delta": fmt_float(last.get("pred_slow_energy_delta")),
                    "slow_energy_scale_std": fmt_float(safe_nanstd(series(analysis.rows, "slow_energy_scale"))),
                    "slow_compute_budget_std": fmt_float(safe_nanstd(series(analysis.rows, "slow_compute_budget"))),
                    "goal_entropy_std": fmt_float(safe_nanstd(series(analysis.rows, "slow_goal_entropy"))),
                    "error_vs_collision_rate": fmt_float(analysis.correlations.get("error_vs_collision_rate")),
                    "error_vs_energy_drop": fmt_float(analysis.correlations.get("error_vs_energy_drop")),
                    "error_delta_vs_slow_energy_delta": fmt_float(analysis.correlations.get("error_delta_vs_slow_energy_delta")),
                    "flags": "|".join(analysis.flags),
                    "notes": "|".join(analysis.notes),
                }
            )


def write_markdown(path: Path, analyses: List[RunAnalysis], requested_dirs: List[Path]):
    lines = ["# Dual-System Analysis", ""]
    if not analyses:
        lines.extend(["No usable runs were found.", "", "Requested directories:", ""])
        lines.extend([f"- `{run_dir}`" for run_dir in requested_dirs])
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    verdict_counts = {name: sum(1 for analysis in analyses if analysis.verdict == name) for name in ("PASS", "PARTIAL", "FAIL")}
    overall = "PASS" if verdict_counts["FAIL"] == 0 and verdict_counts["PARTIAL"] == 0 else "PARTIAL"
    if verdict_counts["FAIL"] > verdict_counts["PASS"]:
        overall = "FAIL"
    lines.extend(
        [
            f"Overall verdict: **{overall}**",
            "",
            f"Runs analyzed: {len(analyses)}",
            f"Verdicts: PASS={verdict_counts['PASS']}, PARTIAL={verdict_counts['PARTIAL']}, FAIL={verdict_counts['FAIL']}",
            "",
            "## Per-Run Verdicts",
            "",
            "| Run | Verdict | Final food | Final energy | Fallback | Error EMA | Flags |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for analysis in analyses:
        last = analysis.rows[-1]
        lines.append(
            f"| `{analysis.run_name}` | {analysis.verdict} | "
            f"{fmt_float(last.get('total_food_eaten'))} | {fmt_float(last.get('mean_energy'))} | "
            f"{fmt_float(last.get('fallback_rate'))} | {fmt_float(last.get('error_ema'))} | "
            f"{', '.join(analysis.flags) if analysis.flags else 'none'} |"
        )

    lines.extend(["", "## Required Checks", ""])
    for analysis in analyses:
        last = analysis.rows[-1]
        lines.extend(
            [
                f"### `{analysis.run_name}`",
                "",
                f"- Verdict: **{analysis.verdict}**",
                f"- Fallback dependence: final fallback `{fmt_float(last.get('fallback_rate'))}`, "
                f"no-fallback entropy `{fmt_float(last.get('no_fallback_action_entropy'))}`, "
                f"policy/fallback agreement `{fmt_float(last.get('policy_vs_fallback_agreement'))}`.",
                f"- Prediction quality: world100 `{fmt_float(last.get('mean_world_error_100'))}`, "
                f"reservoir100 `{fmt_float(last.get('mean_reservoir_error_100'))}`, "
                f"error/collision corr `{fmt_float(analysis.correlations.get('error_vs_collision_rate'))}`, "
                f"error/energy-drop corr `{fmt_float(analysis.correlations.get('error_vs_energy_drop'))}`.",
                f"- Slow behavior: energy-scale std `{fmt_float(safe_nanstd(series(analysis.rows, 'slow_energy_scale')))}`, "
                f"compute-budget std `{fmt_float(safe_nanstd(series(analysis.rows, 'slow_compute_budget')))}`, "
                f"goal-entropy std `{fmt_float(safe_nanstd(series(analysis.rows, 'slow_goal_entropy')))}`.",
                f"- Slow auxiliary: future-error pred `{fmt_float(last.get('pred_future_error'))}`, "
                f"collision-risk pred `{fmt_float(last.get('pred_slow_collision_risk'))}`, "
                f"energy-delta pred `{fmt_float(last.get('pred_slow_energy_delta'))}`, "
                f"losses `{fmt_float(last.get('slow_error_loss'))}/"
                f"{fmt_float(last.get('slow_collision_loss'))}/"
                f"{fmt_float(last.get('slow_energy_loss'))}/"
                f"{fmt_float(last.get('slow_goal_shaping_loss'))}`.",
                f"- Notes: {', '.join(analysis.notes) if analysis.notes else 'none'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Plots",
            "",
            "Plots are written under `plots/`. Each file contains one chart.",
            "",
            "## Caveats",
            "",
            "- Older runs may not contain named goal probability columns; those analyses are skipped or reported as unavailable.",
            "- This script analyzes logged behavior. It does not alter the dual-system model or action-selection boundary.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(plot_dir: Path, analyses: List[RunAnalysis]):
    for metric in LEARNING_METRICS + ["collisions_per_100_steps"]:
        plot_metric(plot_dir / f"{metric}.png", analyses, metric, metric)
    for metric in SLOW_METRICS:
        plot_metric(plot_dir / f"{metric}.png", analyses, metric, metric)
    for label, goal_column in GOAL_COLUMNS.items():
        for compare_name, compare_column in COMPARE_COLUMNS.items():
            plot_scatter(
                plot_dir / f"{goal_column}_vs_{compare_name}.png",
                analyses,
                x_column=compare_column,
                y_column=goal_column,
                title=f"{label} vs {compare_name}",
            )


def plot_metric(path: Path, analyses: List[RunAnalysis], column: str, title: str):
    plt.figure()
    plotted = False
    for analysis in analyses:
        x = series(analysis.rows, "step")
        y = series(analysis.rows, column)
        if finite_count(y) == 0:
            continue
        plt.plot(x, y, label=analysis.run_name)
        plotted = True
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(column)
    if plotted and len(analyses) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_scatter(path: Path, analyses: List[RunAnalysis], x_column: str, y_column: str, title: str):
    plt.figure()
    plotted = False
    for analysis in analyses:
        x = series(analysis.rows, x_column)
        y = series(analysis.rows, y_column)
        mask = np.isfinite(x) & np.isfinite(y)
        if not mask.any():
            continue
        plt.scatter(x[mask], y[mask], label=analysis.run_name)
        plotted = True
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    if plotted and len(analyses) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def improves(rows: List[Dict[str, float]], column: str):
    values = series(rows, column)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return False
    first = float(np.nanmean(values[: max(1, values.size // 3)]))
    last = float(np.nanmean(values[-max(1, values.size // 3) :]))
    return last <= first * 1.1


def declines(rows: List[Dict[str, float]], column: str):
    values = series(rows, column)
    values = values[np.isfinite(values)]
    if values.size < 3:
        return False
    first = float(np.nanmean(values[: max(1, values.size // 3)]))
    last = float(np.nanmean(values[-max(1, values.size // 3) :]))
    return last < first


def modulation_response_score(rows: List[Dict[str, float]]):
    errors = series(rows, "error_ema")
    energy = series(rows, "slow_energy_scale")
    compute = series(rows, "slow_compute_budget")
    if len(errors) < 4:
        return float("nan")
    error_delta = deltas(errors)
    modulation_delta = np.maximum(np.abs(deltas(energy)), np.abs(deltas(compute)))
    mask = np.isfinite(error_delta) & np.isfinite(modulation_delta)
    if not mask.any():
        return float("nan")
    threshold = np.nanpercentile(error_delta[mask], 70)
    high = mask & (error_delta >= threshold)
    if not high.any():
        return 0.0
    return float(np.nanmean(modulation_delta[high] > 1e-3))


def late_rows(rows: List[Dict[str, float]]):
    if not rows:
        return []
    start = max(0, int(len(rows) * 0.5))
    return rows[start:]


def is_eval_run(summary: Dict[str, object]):
    return bool(summary.get("loaded_checkpoint")) and float(summary.get("mean_fallback_rate", 1.0) or 1.0) <= 0.05


def is_eval_poor(summary: Dict[str, object]):
    steps = float(summary.get("steps", 0.0) or 0.0)
    energy = float(summary.get("mean_energy", 0.0) or 0.0)
    food = float(summary.get("total_food_eaten", 0.0) or 0.0)
    return steps < 500 or energy <= 50.0 or food <= 0.0


def choose_available_series(rows: List[Dict[str, float]], columns: List[str]):
    for column in columns:
        values = series(rows, column)
        if finite_count(values):
            return values
    return np.asarray([], dtype=np.float32)


def series(rows: List[Dict[str, float]], column: str):
    return np.asarray([parse_float(row.get(column), default=float("nan")) for row in rows], dtype=np.float64)


def deltas(values: Iterable[float]):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(values)


def negative_deltas(values: Iterable[float]):
    return np.maximum(0.0, -deltas(values))


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.size, b.size)
    if n < 2:
        return float("nan")
    a = a[:n]
    b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mean(values):
    parsed = [parse_float(value, default=float("nan")) for value in values]
    finite = [value for value in parsed if math.isfinite(value)]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def min_value(values):
    finite = [parse_float(value, default=float("nan")) for value in values]
    finite = [value for value in finite if math.isfinite(value)]
    return min(finite) if finite else float("nan")


def max_value(values):
    finite = [parse_float(value, default=float("nan")) for value in values]
    finite = [value for value in finite if math.isfinite(value)]
    return max(finite) if finite else float("nan")


def parse_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def finite_count(values):
    array = np.asarray(values, dtype=np.float64)
    return int(np.isfinite(array).sum())


def safe_nanmean(values, default=float("nan")):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return default
    return float(np.mean(array))


def safe_nanstd(values, default=float("nan")):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return default
    return float(np.std(array))


def is_finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt_float(value):
    if not is_finite(value):
        return ""
    return f"{float(value):.6f}"


if __name__ == "__main__":
    main()
