# evaluation/metrics.py

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import fsum
from typing import Any


ATTACKER_ORDER = ["recon_probe", "scripted_exploit", "ai_probe"]
UNKNOWN_ATTACKER_BUCKET = "__unknown__"


@dataclass
class EpisodeStep:
    step: int
    attacker_type: str
    action: list[int]
    suspicion: float
    depth: float
    duration: float
    flagged: bool
    raw_reward: float
    normalized_reward: float
    terminated: bool
    truncated: bool
    changes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeMetrics:
    attacker_type: str
    defender_name: str
    seed: int
    episode_index: int
    n_steps: int
    total_dwell_time_s: float
    mean_step_dwell_time_s: float
    deception_success: bool
    deception_success_rate_steps: float
    detection_rate_steps: float
    total_flags: int
    total_interaction_depth: float
    mean_interaction_depth: float
    total_raw_reward: float
    total_normalized_reward: float
    ended_by_detection_streak: bool
    ended_by_max_steps: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(fsum(values) / len(values))


def _empty_summary() -> dict[str, Any]:
    return {
        "episodes": 0,
        "mean_total_dwell_time_s": 0.0,
        "mean_step_dwell_time_s": 0.0,
        "deception_success_rate_episodes": 0.0,
        "deception_success_rate_steps": 0.0,
        "detection_rate_steps": 0.0,
        "mean_total_interaction_depth": 0.0,
        "mean_interaction_depth": 0.0,
        "mean_total_raw_reward": 0.0,
        "mean_total_normalized_reward": 0.0,
        "early_termination_rate": 0.0,
        "max_step_truncation_rate": 0.0,
    }


def compute_episode_metrics(
    *,
    steps: list[EpisodeStep],
    attacker_type: str,
    defender_name: str,
    seed: int,
    episode_index: int,
) -> EpisodeMetrics:
    if not steps:
        return EpisodeMetrics(
            attacker_type=attacker_type,
            defender_name=defender_name,
            seed=seed,
            episode_index=episode_index,
            n_steps=0,
            total_dwell_time_s=0.0,
            mean_step_dwell_time_s=0.0,
            deception_success=False,
            deception_success_rate_steps=0.0,
            detection_rate_steps=0.0,
            total_flags=0,
            total_interaction_depth=0.0,
            mean_interaction_depth=0.0,
            total_raw_reward=0.0,
            total_normalized_reward=0.0,
            ended_by_detection_streak=False,
            ended_by_max_steps=False,
        )

    n_steps = len(steps)
    durations = [float(s.duration) for s in steps]
    depths = [float(s.depth) for s in steps]
    raw_rewards = [float(s.raw_reward) for s in steps]
    normalized_rewards = [float(s.normalized_reward) for s in steps]
    flags = [1 if bool(s.flagged) else 0 for s in steps]

    last = steps[-1]
    total_flags = int(sum(flags))
    non_flagged_steps = n_steps - total_flags

    ended_by_detection_streak = bool(last.terminated)
    ended_by_max_steps = bool(last.truncated)

    deception_success = not ended_by_detection_streak

    return EpisodeMetrics(
        attacker_type=attacker_type,
        defender_name=defender_name,
        seed=seed,
        episode_index=episode_index,
        n_steps=n_steps,
        total_dwell_time_s=round(float(sum(durations)), 6),
        mean_step_dwell_time_s=round(safe_mean(durations), 6),
        deception_success=deception_success,
        deception_success_rate_steps=round(float(non_flagged_steps / n_steps), 6),
        detection_rate_steps=round(float(total_flags / n_steps), 6),
        total_flags=total_flags,
        total_interaction_depth=round(float(sum(depths)), 6),
        mean_interaction_depth=round(safe_mean(depths), 6),
        total_raw_reward=round(float(sum(raw_rewards)), 6),
        total_normalized_reward=round(float(sum(normalized_rewards)), 6),
        ended_by_detection_streak=ended_by_detection_streak,
        ended_by_max_steps=ended_by_max_steps,
    )


def aggregate_metrics(
    episode_metrics: list[EpisodeMetrics],
) -> dict[str, Any]:
    if not episode_metrics:
        return _empty_summary()

    return {
        "episodes": len(episode_metrics),
        "mean_total_dwell_time_s": round(
            safe_mean([m.total_dwell_time_s for m in episode_metrics]), 6
        ),
        "mean_step_dwell_time_s": round(
            safe_mean([m.mean_step_dwell_time_s for m in episode_metrics]), 6
        ),
        "deception_success_rate_episodes": round(
            safe_mean([1.0 if m.deception_success else 0.0 for m in episode_metrics]), 6
        ),
        "deception_success_rate_steps": round(
            safe_mean([m.deception_success_rate_steps for m in episode_metrics]), 6
        ),
        "detection_rate_steps": round(
            safe_mean([m.detection_rate_steps for m in episode_metrics]), 6
        ),
        "mean_total_interaction_depth": round(
            safe_mean([m.total_interaction_depth for m in episode_metrics]), 6
        ),
        "mean_interaction_depth": round(
            safe_mean([m.mean_interaction_depth for m in episode_metrics]), 6
        ),
        "mean_total_raw_reward": round(
            safe_mean([m.total_raw_reward for m in episode_metrics]), 6
        ),
        "mean_total_normalized_reward": round(
            safe_mean([m.total_normalized_reward for m in episode_metrics]), 6
        ),
        "early_termination_rate": round(
            safe_mean(
                [1.0 if m.ended_by_detection_streak else 0.0 for m in episode_metrics]
            ),
            6,
        ),
        "max_step_truncation_rate": round(
            safe_mean([1.0 if m.ended_by_max_steps else 0.0 for m in episode_metrics]),
            6,
        ),
    }


def aggregate_metric_dicts(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not metric_rows:
        return _empty_summary()

    metric_objs = [EpisodeMetrics(**row) for row in metric_rows]
    return aggregate_metrics(metric_objs)


def by_attacker(episode_metrics: list[EpisodeMetrics]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[EpisodeMetrics]] = {
        name: [] for name in [*ATTACKER_ORDER, UNKNOWN_ATTACKER_BUCKET]
    }

    for metric in episode_metrics:
        if metric.attacker_type in ATTACKER_ORDER:
            grouped[metric.attacker_type].append(metric)
        else:
            grouped[UNKNOWN_ATTACKER_BUCKET].append(metric)

    result = {
        attacker_name: aggregate_metrics(grouped.get(attacker_name, []))
        for attacker_name in ATTACKER_ORDER
    }

    if grouped[UNKNOWN_ATTACKER_BUCKET]:
        result[UNKNOWN_ATTACKER_BUCKET] = aggregate_metrics(grouped[UNKNOWN_ATTACKER_BUCKET])

    return result


def _pct_change(new_value: float, baseline_value: float) -> float | None:
    if abs(baseline_value) < 1e-12:
        return None
    return round(((new_value - baseline_value) / abs(baseline_value)) * 100.0, 6)


def compute_baseline_delta(
    *,
    rl_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> dict[str, dict[str, float | None]]:
    fields = [
        "mean_total_dwell_time_s",
        "deception_success_rate_episodes",
        "deception_success_rate_steps",
        "detection_rate_steps",
        "mean_total_interaction_depth",
        "mean_total_raw_reward",
        "mean_total_normalized_reward",
    ]

    deltas: dict[str, dict[str, float | None]] = {}
    for field in fields:
        rl_val = float(rl_summary.get(field, 0.0))
        base_val = float(baseline_summary.get(field, 0.0))
        abs_delta = round(rl_val - base_val, 6)
        pct_delta = _pct_change(rl_val, base_val)

        deltas[field] = {
            "rl": round(rl_val, 6),
            "baseline": round(base_val, 6),
            "abs_delta": abs_delta,
            "pct_delta": pct_delta,
        }

    return deltas


def best_non_rl_baseline_summary(
    baseline_summaries: dict[str, dict[str, Any]],
    key: str = "mean_total_raw_reward",
) -> tuple[str, dict[str, Any]]:
    valid = {
        name: summary
        for name, summary in baseline_summaries.items()
        if isinstance(summary, dict) and summary
    }

    if not valid:
        return "none", {}

    best_name = max(
        valid.keys(),
        key=lambda name: float(valid[name].get(key, float("-inf"))),
    )
    return best_name, valid[best_name]