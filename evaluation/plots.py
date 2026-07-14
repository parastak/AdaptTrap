
# evaluation/plots.py
"""
Plot generation for the AdaptTrap final report.
 
Design choices, on purpose:
- Per-attacker bar charts use the two seed runs as two independent data
  points, shown as min/max whiskers -- NOT a std-dev or fabricated 95% CI.
  Two points is not enough to estimate a distribution; a whisker spanning
  the observed min/max is the honest amount of uncertainty to show.
- Zero-height bars (RL's 0% detection) are explicitly annotated with text,
  since a bar of height zero is visually indistinguishable from "no data."
- Training curve shows raw points (low alpha) + rolling mean, with vertical
  lines marking curriculum phase boundaries pulled from the log's own
  metadata (not hardcoded), so this stays correct if you retrain later.
"""
 
from __future__ import annotations
 
import json
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
 
from evaluation.stats_utils import ATTACKER_ORDER, DEFENDER_ORDER, load_run
 
DEFENDER_LABELS = {
    "static": "Static",
    "random": "Random",
    "rule_based": "Rule-based",
    "rl_greedy": "RL (trained)",
}
DEFENDER_COLORS = {
    "static": "#9aa5b1",
    "random": "#e07a5f",
    "rule_based": "#f2cc8f",
    "rl_greedy": "#3d5a80",
}
ATTACKER_LABELS = {
    "recon_probe": "recon_probe\n(easy)",
    "scripted_exploit": "scripted_exploit\n(medium)",
    "ai_probe": "ai_probe\n(hard)",
}
 
 
def _rolling_mean(values: list[float], window: int) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")
 
 
def plot_training_curve(training_log_path: str | Path, out_path: str | Path) -> Path:
    with open(training_log_path, "r", encoding="utf-8") as f:
        log = json.load(f)
 
    episodes = log["episodes"]
    meta = log.get("metadata", {})
    n_total = meta.get("training", {}).get("n_episodes", len(episodes))
    phase1_end = meta.get("training", {}).get("curriculum", {}).get("phase1_end")
    phase2_end = meta.get("training", {}).get("curriculum", {}).get("phase2_end")
 
    fig, axes = plt.subplots(len(ATTACKER_ORDER), 1, figsize=(9, 8), sharex=True)
 
    for ax, attacker in zip(axes, ATTACKER_ORDER):
        atk_eps = [e for e in episodes if e["attacker_type"] == attacker]
        if not atk_eps:
            ax.set_title(f"{attacker} (no episodes logged)")
            continue
 
        x = [e["episode"] for e in atk_eps]
        y = [e["raw_total_reward"] for e in atk_eps]
 
        ax.scatter(x, y, s=6, alpha=0.15, color=DEFENDER_COLORS["rl_greedy"], linewidths=0)
 
        window = min(30, max(3, len(y) // 10))
        roll = _rolling_mean(y, window)
        if len(roll) > 0:
            roll_x = x[window - 1:]
            ax.plot(roll_x, roll, color="#293241", linewidth=1.8,
                     label=f"rolling mean (window={window})")
 
        if phase1_end:
            ax.axvline(phase1_end * n_total, color="#bcbcbc", linestyle="--", linewidth=1)
        if phase2_end:
            ax.axvline(phase2_end * n_total, color="#bcbcbc", linestyle="--", linewidth=1)
 
        ax.set_ylabel("raw episode\nreward")
        ax.set_title(attacker, loc="left", fontsize=10, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
 
    axes[-1].set_xlabel("training episode")
    fig.suptitle(
        "Training reward curve by attacker tier\n"
        "(dashed lines = curriculum phase boundaries)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
 
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
 
 
def _collect_metric_by_run(
    run_paths: list[str | Path], metric: str
) -> dict[str, dict[str, list[float]]]:
    """
    Returns: {attacker: {defender: [value_run1, value_run2, ...]}}
    """
    result = {atk: {d: [] for d in DEFENDER_ORDER} for atk in ATTACKER_ORDER}
    for path in run_paths:
        data = load_run(path)
        for attacker in ATTACKER_ORDER:
            for defender in DEFENDER_ORDER:
                block = data["scoreboard"][defender]["by_attacker"].get(attacker, {})
                if metric in block:
                    result[attacker][defender].append(float(block[metric]))
    return result
 
 
def _grouped_bar_with_range(
    values_by_attacker: dict[str, dict[str, list[float]]],
    ylabel: str,
    title: str,
    out_path: str | Path,
    as_percent: bool = False,
    annotate_zero: bool = False,
) -> Path:
    n_attackers = len(ATTACKER_ORDER)
    n_defenders = len(DEFENDER_ORDER)
    bar_width = 0.8 / n_defenders
 
    fig, ax = plt.subplots(figsize=(9, 5))
 
    for d_idx, defender in enumerate(DEFENDER_ORDER):
        means, err_low, err_high = [], [], []
        for attacker in ATTACKER_ORDER:
            vals = values_by_attacker[attacker][defender]
            scale = 100.0 if as_percent else 1.0
            vals = [v * scale for v in vals]
            if not vals:
                means.append(0.0)
                err_low.append(0.0)
                err_high.append(0.0)
                continue
            mean_v = float(np.mean(vals))
            means.append(mean_v)
            err_low.append(mean_v - min(vals))
            err_high.append(max(vals) - mean_v)
 
        x = np.arange(n_attackers) + (d_idx - (n_defenders - 1) / 2) * bar_width
        bars = ax.bar(
            x, means, width=bar_width * 0.9,
            label=DEFENDER_LABELS[defender], color=DEFENDER_COLORS[defender],
        )
        # only show whiskers where run count > 1 (avoids fake error bars on n=1)
        has_range = [eh > 1e-9 or el > 1e-9 for eh, el in zip(err_high, err_low)]
        if any(has_range):
            ax.errorbar(
                x, means, yerr=[err_low, err_high],
                fmt="none", ecolor="black", elinewidth=1, capsize=3,
            )
 
        if annotate_zero:
            for xi, mi in zip(x, means):
                if abs(mi) < 1e-9:
                    ax.annotate("0%", (xi, 0.0), textcoords="offset points",
                                xytext=(0, 4), ha="center", fontsize=8)
 
    ax.set_xticks(np.arange(n_attackers))
    ax.set_xticklabels([ATTACKER_LABELS[a] for a in ATTACKER_ORDER])
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\n(whiskers = min/max across the 2 seed runs, not a fitted CI)",
                 fontsize=11)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
 
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
 
 
def plot_reward_by_attacker(run_paths: list[str | Path], out_path: str | Path) -> Path:
    values = _collect_metric_by_run(run_paths, "mean_total_raw_reward")
    return _grouped_bar_with_range(
        values,
        ylabel="mean raw reward per episode",
        title="Raw reward by defender, per attacker tier",
        out_path=out_path,
    )
 
 
def plot_detection_rate_by_attacker(run_paths: list[str | Path], out_path: str | Path) -> Path:
    values = _collect_metric_by_run(run_paths, "detection_rate_steps")
    return _grouped_bar_with_range(
        values,
        ylabel="detection rate (% of sessions flagged)",
        title="Detection rate by defender, per attacker tier",
        out_path=out_path,
        as_percent=True,
        annotate_zero=True,
    )
 
 
def plot_dwell_time_by_attacker(run_paths: list[str | Path], out_path: str | Path) -> Path:
    values = _collect_metric_by_run(run_paths, "mean_total_dwell_time_s")
    return _grouped_bar_with_range(
        values,
        ylabel="mean total dwell time (s)",
        title="Attacker dwell time by defender, per attacker tier",
        out_path=out_path,
    )
 
 
def generate_all_plots(
    run_paths: list[str | Path],
    training_log_path: str | Path,
    out_dir: str | Path = "logs/plots",
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    paths = {
        "training_curve": plot_training_curve(training_log_path, out_dir / "training_reward_curve.png"),
        "reward_by_attacker": plot_reward_by_attacker(run_paths, out_dir / "reward_by_attacker.png"),
        "detection_rate_by_attacker": plot_detection_rate_by_attacker(
            run_paths, out_dir / "detection_rate_by_attacker.png"
        ),
        "dwell_time_by_attacker": plot_dwell_time_by_attacker(
            run_paths, out_dir / "dwell_time_by_attacker.png"
        ),
    }
    return paths
 
 
if __name__ == "__main__":
    import sys
 
    run_paths = sys.argv[1:] or ["logs/eval_results_run1.json", "logs/eval_results_run2.json"]
    result = generate_all_plots(run_paths, training_log_path="logs/training2.json")
    for name, path in result.items():
        print(f"{name}: {path}")
 