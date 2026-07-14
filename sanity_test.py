#!/usr/bin/env python3
"""
sanity_test.py -- AdaptTrap end-to-end health check.

Run this after cloning the repo, after any refactor, and before a demo.
It does NOT re-validate your research results (that's what
evaluation/final_report.md is for) -- it only checks that the system
actually runs: environment boots, all three attacker tiers respond, the
policy network does a forward pass, the non-RL baselines act, a trained
checkpoint (if present) loads and runs greedy inference, and a tiny
end-to-end benchmark completes without exploding.

Exit code 0 = everything checked passed. Non-zero = something is broken.
This is intentionally the first thing a reviewer, recruiter, or future-you
should run.

Usage:
    python sanity_test.py
    python sanity_test.py --checkpoint logs/model_best2.pt
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass, field


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    duration_s: float = 0.0


@dataclass
class CheckRunner:
    results: list[CheckResult] = field(default_factory=list)

    def run(self, name: str, fn) -> None:
        start = time.time()
        try:
            detail = fn() or ""
            self.results.append(CheckResult(name, True, detail, time.time() - start))
        except Exception as exc:  # noqa: BLE001 -- sanity check wants to catch everything
            tb = traceback.format_exc(limit=3)
            self.results.append(
                CheckResult(name, False, f"{type(exc).__name__}: {exc}\n{tb}", time.time() - start)
            )

    def summarize(self) -> bool:
        print("\n" + "=" * 70)
        print("SANITY CHECK SUMMARY")
        print("=" * 70)
        all_passed = True
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            all_passed &= r.passed
            print(f"[{status}] {r.name:<45} ({r.duration_s:.2f}s)")
            if not r.passed:
                print(f"       -> {r.detail.splitlines()[0]}")
            elif r.detail:
                print(f"       -> {r.detail}")
        print("=" * 70)
        print("ALL CHECKS PASSED" if all_passed else "ONE OR MORE CHECKS FAILED")
        print("=" * 70 + "\n")
        return all_passed


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_env_boots() -> str:
    from env.honeypot_env import HoneypotEnv

    env = HoneypotEnv(fast_training=True)
    env.close()
    return "HoneypotEnv() constructs and closes cleanly"


def check_all_attackers_reset_and_step() -> str:
    import numpy as np
    from env.honeypot_env import HoneypotEnv

    env = HoneypotEnv(fast_training=True)
    details = []
    try:
        for attacker_name in ["recon_probe", "scripted_exploit", "ai_probe"]:
            obs, info = env.reset(seed=0, attacker_name=attacker_name)
            assert obs is not None, f"{attacker_name}: reset() returned no observation"
            action = env.action_space.sample()
            obs2, reward, terminated, truncated, step_info = env.step(action)
            assert "suspicion" in step_info, f"{attacker_name}: step info missing 'suspicion'"
            assert "flagged" in step_info, f"{attacker_name}: step info missing 'flagged'"
            details.append(f"{attacker_name}: reward={reward:.3f} flagged={step_info['flagged']}")
    finally:
        env.close()
    return "; ".join(details)


def check_actor_critic_forward_pass() -> str:
    import torch
    from defender.actor_critic import ActorCritic, N_ATTACKERS

    model = ActorCritic(state_dim=12, n_attackers=N_ATTACKERS)
    state = torch.zeros(1, 12)
    outputs = []
    for idx in range(N_ATTACKERS):
        actions, log_probs, value = model.get_action(state, attacker_idx=idx)
        assert len(actions) == 3, f"expected 3 sub-actions, got {len(actions)}"
        outputs.append(f"attacker_idx={idx} actions={actions}")
    param_counts = model.head_parameter_count()
    return f"{'; '.join(outputs)} | total_params={param_counts['total']}"


def check_baseline_defenders_act() -> str:
    import numpy as np
    from evaluation.baselines import build_defender, EXPECTED_OBS_DIM

    dummy_obs = np.zeros(EXPECTED_OBS_DIM, dtype=np.float32)
    results = []
    for name in ["static", "random", "rule_based"]:
        defender = build_defender(name, seed=0)
        defender.reset("ai_probe")
        action = defender.act(dummy_obs, info=None)
        assert len(action) == 3, f"{name}: expected 3-int action, got {action}"
        results.append(f"{name}={action}")
    return "; ".join(results)


def check_rl_checkpoint(checkpoint_path: str) -> str:
    from pathlib import Path

    if not Path(checkpoint_path).exists():
        return f"SKIPPED -- no checkpoint at '{checkpoint_path}' (train first, or pass --checkpoint)"

    import numpy as np
    from evaluation.baselines import build_defender, EXPECTED_OBS_DIM

    defender = build_defender("rl_greedy", checkpoint_path=checkpoint_path, device="cpu")
    dummy_obs = np.zeros(EXPECTED_OBS_DIM, dtype=np.float32)
    for attacker in ["recon_probe", "scripted_exploit", "ai_probe"]:
        defender.reset(attacker)
        action = defender.act(dummy_obs, info=None)
        assert len(action) == 3
    return f"loaded '{checkpoint_path}' and ran greedy inference for all 3 attacker tiers"


def check_mini_benchmark() -> str:
    """
    Runs a tiny real benchmark (2 episodes, static + random only, one
    attacker tier) end-to-end through the actual evaluation pipeline, not a
    mock. This is the closest thing to "does the whole system work together."
    """
    from evaluation.benchmark import run_defender_vs_attacker

    result = run_defender_vs_attacker(
        defender_name="static",
        attacker_type="recon_probe",
        episodes=2,
        seed=0,
        checkpoint_path=None,
        fast_training=True,
        target="127.0.0.1",
        verbose=False,
    )
    assert result["episodes_completed"] == 2, f"expected 2 completed episodes, got {result}"
    assert result["episodes_failed"] == 0, f"episodes failed: {result['errors']}"
    reward = result["summary"]["mean_total_raw_reward"]
    return f"2-episode static-vs-recon_probe run completed, mean_total_raw_reward={reward:.3f}"


def check_attacker_modules_importable() -> str:
    from attackers.base_attacker import BaseAttacker
    from attackers.level1_nmap import Level1Scanner
    from attackers.level2_scripted import Level2ScriptedAttacker
    from attackers.level3_bandit import Level3BanditAttacker

    return "base_attacker, level1_nmap, level2_scripted, level3_bandit all import cleanly"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AdaptTrap end-to-end sanity check.")
    parser.add_argument(
        "--checkpoint", default="logs/model_best2.pt",
        help="Path to a trained checkpoint to test greedy RL inference against.",
    )
    args = parser.parse_args()

    runner = CheckRunner()
    runner.run("Attacker modules import cleanly", check_attacker_modules_importable)
    runner.run("HoneypotEnv boots and closes", check_env_boots)
    runner.run("All 3 attacker tiers reset + step", check_all_attackers_reset_and_step)
    runner.run("ActorCritic forward pass (untrained)", check_actor_critic_forward_pass)
    runner.run("Non-RL baseline defenders act correctly", check_baseline_defenders_act)
    runner.run("RL checkpoint loads + greedy inference", lambda: check_rl_checkpoint(args.checkpoint))
    runner.run("Mini end-to-end benchmark (2 episodes)", check_mini_benchmark)

    all_passed = runner.summarize()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()