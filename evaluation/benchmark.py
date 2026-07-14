# evaluation/run_benchmark.py

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch

from env.honeypot_env import HoneypotEnv
from evaluation.baselines import build_defender, BaseDefender
from evaluation.metrics import (
    EpisodeStep,
    aggregate_metric_dicts,
    best_non_rl_baseline_summary,
    compute_baseline_delta,
    compute_episode_metrics,
)

ATTACKER_ORDER = ["recon_probe", "scripted_exploit", "ai_probe"]
DEFENDER_ORDER = ["static", "random", "rule_based", "rl_greedy"]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def validate_names(defenders: list[str], attackers: list[str]) -> None:
    valid_defenders = {
        "static",
        "random",
        "rule_based",
        "rule-based",
        "rule",
        "rl_greedy",
        "rl",
        "trained",
        "trained_rl",
    }
    valid_attackers = set(ATTACKER_ORDER)

    unknown_defenders = [name for name in defenders if name.strip().lower() not in valid_defenders]
    unknown_attackers = [name for name in attackers if name not in valid_attackers]

    if unknown_defenders:
        raise ValueError(
            f"Unknown defender(s): {unknown_defenders}. "
            f"Valid: {sorted(valid_defenders)}"
        )
    if unknown_attackers:
        raise ValueError(
            f"Unknown attacker(s): {unknown_attackers}. "
            f"Valid: {ATTACKER_ORDER}"
        )


def normalize_defender_name(name: str) -> str:
    lowered = name.strip().lower()
    if lowered in {"rule", "rule-based"}:
        return "rule_based"
    if lowered in {"rl", "trained", "trained_rl"}:
        return "rl_greedy"
    return lowered


def make_episode_seed(
    base_seed: int,
    defender_name: str,
    attacker_type: str,
    episode_index: int,
) -> int:
    normalized_defender = normalize_defender_name(defender_name)
    if normalized_defender not in DEFENDER_ORDER:
        raise ValueError(
            f"Defender '{defender_name}' normalized to '{normalized_defender}', "
            f"which is not in DEFENDER_ORDER={DEFENDER_ORDER}"
        )
    if attacker_type not in ATTACKER_ORDER:
        raise ValueError(
            f"Attacker '{attacker_type}' is not in ATTACKER_ORDER={ATTACKER_ORDER}"
        )

    defender_offset = DEFENDER_ORDER.index(normalized_defender) * 10_000
    attacker_offset = ATTACKER_ORDER.index(attacker_type) * 1_000
    return int(base_seed + defender_offset + attacker_offset + episode_index)


def run_single_episode(
    *,
    env: HoneypotEnv,
    defender: BaseDefender,
    attacker_type: str,
    episode_index: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    obs, _ = env.reset(seed=seed, attacker_name=attacker_type)
    defender.reset(attacker_type)

    step_objs: list[EpisodeStep] = []
    step_rows: list[dict[str, Any]] = []
    last_info: dict[str, Any] | None = None

    while True:
        action = defender.act(obs, info=last_info)

        obs_next, _, terminated, truncated, info = env.step(np.array(action, dtype=np.int64))

        step_obj = EpisodeStep(
            step=len(step_objs) + 1,
            attacker_type=attacker_type,
            action=[int(action[0]), int(action[1]), int(action[2])],
            suspicion=float(info["suspicion"]),
            depth=float(info["depth"]),
            duration=float(info["duration"]),
            flagged=bool(info["flagged"]),
            raw_reward=float(info["raw_reward"]),
            normalized_reward=float(info["normalized_reward"]),
            terminated=bool(terminated),
            truncated=bool(truncated),
            changes=list(info.get("changes", [])),
        )
        step_objs.append(step_obj)
        step_rows.append(step_obj.to_dict())

        obs = obs_next
        last_info = info

        if terminated or truncated:
            break

    metric_obj = compute_episode_metrics(
        steps=step_objs,
        attacker_type=attacker_type,
        defender_name=defender.name,
        seed=seed,
        episode_index=episode_index,
    )
    metric_row = metric_obj.to_dict()
    steps_payload = {
        "defender_name": defender.name,
        "attacker_type": attacker_type,
        "seed": seed,
        "episode_index": episode_index,
        "steps": step_rows,
    }
    return metric_row, steps_payload



def run_defender_vs_attacker(
    *,
    defender_name: str,
    attacker_type: str,
    episodes: int,
    seed: int,
    checkpoint_path: str | None,
    fast_training: bool,
    target: str,
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError(f"episodes must be > 0, got {episodes}")
    if attacker_type not in ATTACKER_ORDER:
        raise ValueError(f"Unknown attacker_type '{attacker_type}'")

    normalized_name = normalize_defender_name(defender_name)
    defender = build_defender(
        normalized_name,
        checkpoint_path=checkpoint_path,
        seed=seed,
        device=device,
    )

    episode_metrics: list[dict[str, Any]] = []
    episode_steps: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    env = HoneypotEnv(target=target, fast_training=fast_training)

    try:
        for episode_index in range(1, episodes + 1):
            ep_seed = make_episode_seed(seed, normalized_name, attacker_type, episode_index)

            if verbose:
                print(
                    f"[benchmark] defender={normalized_name} attacker={attacker_type} "
                    f"episode={episode_index}/{episodes} seed={ep_seed}"
                )

            try:
                metric_row, steps_payload = run_single_episode(
                    env=env,
                    defender=defender,
                    attacker_type=attacker_type,
                    episode_index=episode_index,
                    seed=ep_seed,
                )
                episode_metrics.append(metric_row)
                episode_steps.append(steps_payload)
            except Exception as exc:
                errors.append(
                    {
                        "episode_index": episode_index,
                        "seed": ep_seed,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
    finally:
        env.close()

    summary = aggregate_metric_dicts(episode_metrics)

    return {
        "defender_name": defender.name,
        "attacker_type": attacker_type,
        "seed": seed,
        "episodes_requested": episodes,
        "episodes_completed": len(episode_metrics),
        "episodes_failed": len(errors),
        "summary": summary,
        "episode_metrics": episode_metrics,
        "episode_steps": episode_steps,
        "errors": errors,
    }


def build_scoreboard(
    *,
    runs: list[dict[str, Any]],
    defenders: list[str],
    attackers: list[str],
) -> dict[str, dict[str, Any]]:
    scoreboard: dict[str, dict[str, Any]] = {}
    normalized_defenders = [normalize_defender_name(name) for name in defenders]

    for defender_name in normalized_defenders:
        defender_runs = [r for r in runs if r["defender_name"] == defender_name]

        overall_episode_metrics: list[dict[str, Any]] = []
        total_failed = 0
        total_completed = 0

        for run in defender_runs:
            overall_episode_metrics.extend(run["episode_metrics"])
            total_failed += int(run.get("episodes_failed", 0))
            total_completed += int(run.get("episodes_completed", 0))

        overall_summary = aggregate_metric_dicts(overall_episode_metrics)

        by_attacker_summary: dict[str, dict[str, Any]] = {}
        for attacker in attackers:
            matched_run = next(
                (r for r in defender_runs if r["attacker_type"] == attacker),
                None,
            )
            if matched_run is None:
                by_attacker_summary[attacker] = {
                    "error": True,
                    "reason": "missing_run",
                }
            else:
                by_attacker_summary[attacker] = matched_run["summary"]

        scoreboard[defender_name] = {
            "overall": overall_summary,
            "by_attacker": by_attacker_summary,
            "episodes_completed": total_completed,
            "episodes_failed": total_failed,
        }

    return scoreboard


def compute_rl_deltas(
    *,
    scoreboard: dict[str, dict[str, Any]],
    defenders: list[str],
    attackers: list[str],
) -> dict[str, Any]:
    normalized_defenders = [normalize_defender_name(name) for name in defenders]
    baseline_names = [name for name in normalized_defenders if name != "rl_greedy"]

    baseline_summaries = {
        name: scoreboard[name]["overall"]
        for name in baseline_names
        if name in scoreboard
    }

    best_baseline_name, best_baseline_summary = best_non_rl_baseline_summary(
        baseline_summaries,
        key="mean_total_raw_reward",
    )

    rl_delta_overall: dict[str, Any] = {}
    rl_delta_by_attacker: dict[str, dict[str, Any]] = {}

    if "rl_greedy" in scoreboard and best_baseline_summary:
        rl_delta_overall = compute_baseline_delta(
            rl_summary=scoreboard["rl_greedy"]["overall"],
            baseline_summary=best_baseline_summary,
        )

        for attacker in attackers:
            attacker_baseline_summaries = {
                base_name: scoreboard[base_name]["by_attacker"].get(attacker, {})
                for base_name in baseline_names
                if base_name in scoreboard
                and isinstance(scoreboard[base_name]["by_attacker"].get(attacker, {}), dict)
                and "error" not in scoreboard[base_name]["by_attacker"].get(attacker, {})
            }

            best_attacker_baseline_name, best_attacker_baseline_summary = (
                best_non_rl_baseline_summary(
                    attacker_baseline_summaries,
                    key="mean_total_raw_reward",
                )
            )

            rl_attacker_summary = scoreboard["rl_greedy"]["by_attacker"].get(attacker, {})
            rl_delta_by_attacker[attacker] = {
                "baseline_name": best_attacker_baseline_name,
                "delta": (
                    compute_baseline_delta(
                        rl_summary=rl_attacker_summary,
                        baseline_summary=best_attacker_baseline_summary,
                    )
                    if best_attacker_baseline_summary and "error" not in rl_attacker_summary
                    else {}
                ),
            }

    return {
        "best_non_rl_baseline_overall": {
            "name": best_baseline_name,
            "summary": best_baseline_summary,
        },
        "rl_delta_overall_vs_best_baseline": rl_delta_overall,
        "rl_delta_by_attacker_vs_best_baseline": rl_delta_by_attacker,
    }


def run_benchmark(
    *,
    episodes_per_attacker: int = 20,
    seed: int = 42,
    checkpoint_path: str = "logs/model_best2.pt",
    fast_training: bool = True,
    target: str = "127.0.0.1",
    defenders: list[str] | None = None,
    attackers: list[str] | None = None,
    output_path: str = "logs/eval_results.json",
    device: str = "cpu",
    verbose: bool = True,
) -> dict[str, Any]:
    if episodes_per_attacker <= 0:
        raise ValueError(
            f"episodes_per_attacker must be > 0, got {episodes_per_attacker}"
        )

    defenders = defenders or DEFENDER_ORDER
    attackers = attackers or ATTACKER_ORDER

    validate_names(defenders, attackers)
    set_global_seed(seed)
    ensure_parent_dir(output_path)

    normalized_defenders = [normalize_defender_name(name) for name in defenders]
    need_rl_checkpoint = any(name == "rl_greedy" for name in normalized_defenders)

    if need_rl_checkpoint and not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"RL checkpoint not found at '{checkpoint_path}'. "
            f"Train first or provide a valid checkpoint_path."
        )

    runs: list[dict[str, Any]] = []

    for defender_name in defenders:
        for attacker_type in attackers:
            run_result = run_defender_vs_attacker(
                defender_name=defender_name,
                attacker_type=attacker_type,
                episodes=episodes_per_attacker,
                seed=seed,
                checkpoint_path=checkpoint_path,
                fast_training=fast_training,
                target=target,
                device=device,
                verbose=verbose,
            )
            runs.append(run_result)

    scoreboard = build_scoreboard(
        runs=runs,
        defenders=defenders,
        attackers=attackers,
    )

    delta_block = compute_rl_deltas(
        scoreboard=scoreboard,
        defenders=defenders,
        attackers=attackers,
    )

    payload = {
        "config": {
            "episodes_per_attacker": episodes_per_attacker,
            "seed": seed,
            "checkpoint_path": checkpoint_path,
            "fast_training": fast_training,
            "target": target,
            "device": device,
            "defenders": [normalize_defender_name(name) for name in defenders],
            "attackers": attackers,
        },
        "meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner": "evaluation.run_benchmark",
        },
        "runs": runs,
        "scoreboard": scoreboard,
        **delta_block,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


if __name__ == "__main__":
    result = run_benchmark()
    print(json.dumps(result["scoreboard"], indent=2))