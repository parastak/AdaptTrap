
# evaluation/stats_utils.py

"""
Statistical helpers for the AdaptTrap evaluation report.

Two jobs, deliberately kept separate from plotting/report assembly:

1. Recompute cross-seed deltas *directly from the raw eval JSONs* every time
   this is run, instead of trusting a previously-printed number.

2. Compute honest binomial confidence intervals on detection rates using the
   exact Clopper-Pearson method, at two different units of analysis:

   - "episode-level" (PRIMARY, reported as the headline claim): each full
     20-session episode is one independent trial, and the trial "succeeds"
     (counts as a detection) if ANY session inside it was flagged. This is
     the conservative, defensible unit -- each episode uses an independent
     RNG seed (see benchmark.make_episode_seed), so episodes themselves are
     genuinely independent draws.

   - "session-level" (SECONDARY, descriptive telemetry only -- NOT used to
     justify a tighter confidence interval): each attacker session is a
     fresh attacker instance with no carried-over state (see
     HoneypotEnv._build_attacker()), so the ATTACKER has no memory across
     sessions. However, the DEFENDER's observation includes session_count
     and consec_detects within an episode (see HoneypotEnv._get_obs()), so
     the defender's chosen action can depend on prior session outcomes
     within the same episode. That breaks strict session-to-session
     independence for CI purposes even though the attacker side is memoryless.
     Session counts are reported as raw telemetry (e.g. "X of Y sessions
     flagged") but the tighter CI from treating all sessions as independent
     trials is not used as the reported claim.

Report both, but treat episode-level as ground truth.
"""
from __future__ import annotations
 
import json
from pathlib import Path
from typing import Any
 
from scipy.stats import beta
 
ATTACKER_ORDER = ["recon_probe", "scripted_exploit", "ai_probe"]
DEFENDER_ORDER = ["static", "random", "rule_based", "rl_greedy"]
 
 
# ---------------------------------------------------------------------------
# Binomial confidence intervals
# ---------------------------------------------------------------------------
 
def clopper_pearson_ci(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Exact Clopper-Pearson two-sided confidence interval for a binomial
    proportion. Returns (lower, upper) as fractions in [0, 1].
 
    successes = number of "events" observed (e.g. detections)
    n         = number of trials
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if not (0 <= successes <= n):
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
 
    alpha = 1.0 - confidence
 
    lower = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, n - successes + 1)
    upper = 1.0 if successes == n else beta.ppf(1 - alpha / 2, successes + 1, n - successes)
 
    return float(lower), float(upper)
 
 
def format_rate_with_ci(successes: int, n: int, confidence: float = 0.95) -> str:
    """
    Human-readable string for a detection-rate claim, honest about sample size.
    e.g. "0.0% (n=2000, 95% CI upper bound 0.18%)"
    """
    rate = successes / n
    lower, upper = clopper_pearson_ci(successes, n, confidence)
    return (
        f"{rate * 100:.2f}% observed (n={n}, {int(confidence * 100)}% CI: "
        f"[{lower * 100:.3f}%, {upper * 100:.3f}%])"
    )
 
 
# ---------------------------------------------------------------------------
# Loading raw eval results
# ---------------------------------------------------------------------------
 
def load_run(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
 
 
def _find_run(data: dict[str, Any], defender: str, attacker: str) -> dict[str, Any]:
    match = next(
        (
            r
            for r in data["runs"]
            if r["defender_name"] == defender and r["attacker_type"] == attacker
        ),
        None,
    )
    if match is None:
        raise KeyError(f"No run found for defender={defender!r} attacker={attacker!r}")
    return match
 
 
def session_level_counts(
    data: dict[str, Any], defender: str, attacker: str
) -> tuple[int, int]:
    """
        Descriptive telemetry only: (detected_sessions, total_sessions) for one
        (defender, attacker) run. NOT an independent-trials count -- see module
        docstring. Do not feed this into clopper_pearson_ci() as if each session
        freeze_release.shwere an independent draw for a headline claim.
    """
    run = _find_run(data, defender, attacker)
    total_flags = sum(m["total_flags"] for m in run["episode_metrics"])
    total_steps = sum(m["n_steps"] for m in run["episode_metrics"])
    return int(total_flags), int(total_steps)
 
 
def episode_level_counts(
    data: dict[str, Any], defender: str, attacker: str
) -> tuple[int, int]:
    """
    Returns (episodes_with_any_detection, total_episodes) for one
    (defender, attacker) run.
    """
    run = _find_run(data, defender, attacker)
    metrics = run["episode_metrics"]
    flagged_episodes = sum(1 for m in metrics if m["total_flags"] > 0)
    return int(flagged_episodes), len(metrics)
 
 
def pooled_session_level_counts(
    run_paths: list[str | Path], defender: str, attacker: str
) -> tuple[int, int]:
    detected, total = 0, 0
    for path in run_paths:
        data = load_run(path)
        d, t = session_level_counts(data, defender, attacker)
        detected += d
        total += t
    return detected, total
 
 
def pooled_episode_level_counts(
    run_paths: list[str | Path], defender: str, attacker: str
) -> tuple[int, int]:
    detected, total = 0, 0
    for path in run_paths:
        data = load_run(path)
        d, t = episode_level_counts(data, defender, attacker)
        detected += d
        total += t
    return detected, total
 
 
# ---------------------------------------------------------------------------
# Cross-seed delta recomputation
# ---------------------------------------------------------------------------
 
def per_attacker_metric(
    data: dict[str, Any], defender: str, attacker: str, metric: str
) -> float:
    return float(data["scoreboard"][defender]["by_attacker"][attacker][metric])
 
 
def recompute_cross_seed_table(
    run_paths: list[str | Path],
    metric: str = "mean_total_raw_reward",
    baseline_defender: str = "static",
    rl_defender: str = "rl_greedy",
) -> dict[str, dict[str, Any]]:
    """
    Pulls `metric` for `baseline_defender` and `rl_defender` directly out of
    each run's JSON (no trusting anyone's printed console summary), and
    computes the RL-vs-baseline percentage delta independently per run, per
    attacker. This is the "recompute it yourself" step.
 
    Returns:
        {
          attacker_name: {
            "per_run": [{"seed": ..., "baseline": ..., "rl": ..., "pct_delta": ...}, ...],
            "min_pct_delta": ...,
            "max_pct_delta": ...,
          },
          ...
        }
    """
    runs_data = [load_run(p) for p in run_paths]
 
    table: dict[str, dict[str, Any]] = {}
    for attacker in ATTACKER_ORDER:
        per_run = []
        for data in runs_data:
            seed = data["config"]["seed"]
            baseline_val = per_attacker_metric(data, baseline_defender, attacker, metric)
            rl_val = per_attacker_metric(data, rl_defender, attacker, metric)
            pct_delta = (
                ((rl_val - baseline_val) / abs(baseline_val)) * 100.0
                if abs(baseline_val) > 1e-12
                else None
            )
            per_run.append(
                {
                    "seed": seed,
                    "baseline": round(baseline_val, 4),
                    "rl": round(rl_val, 4),
                    "pct_delta": round(pct_delta, 4) if pct_delta is not None else None,
                }
            )
 
        deltas = [r["pct_delta"] for r in per_run if r["pct_delta"] is not None]
        table[attacker] = {
            "per_run": per_run,
            "min_pct_delta": round(min(deltas), 4) if deltas else None,
            "max_pct_delta": round(max(deltas), 4) if deltas else None,
        }
 
    return table
 
 
if __name__ == "__main__":
    # Quick self-check when run directly: python -m evaluation.stats_utils
    import sys
 
    run_paths = sys.argv[1:] or ["logs/eval_results_run1.json", "logs/eval_results_run2.json"]
 
    print("=== Cross-seed raw reward deltas (recomputed from JSON, not trusted from console) ===")
    table = recompute_cross_seed_table(run_paths)
    for attacker, block in table.items():
        print(f"\n{attacker}:")
        for row in block["per_run"]:
            print(f"  seed={row['seed']:<5} baseline={row['baseline']:>8} rl={row['rl']:>8} "
                  f"pct_delta={row['pct_delta']:>7}%")
        print(f"  range across seeds: {block['min_pct_delta']}% to {block['max_pct_delta']}%")
 
    print("\n=== Detection-rate confidence intervals (ai_probe, rl_greedy) ===")
    d_sess, n_sess = pooled_session_level_counts(run_paths, "rl_greedy", "ai_probe")
    d_ep, n_ep = pooled_episode_level_counts(run_paths, "rl_greedy", "ai_probe")
    print(f"Session-level: {format_rate_with_ci(d_sess, n_sess)}")
    print(f"Episode-level: {format_rate_with_ci(d_ep, n_ep)}")
 