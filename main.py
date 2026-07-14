# main.py
from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from attackers.level1_nmap import Level1Scanner
from attackers.level2_scripted import Level2ScriptedAttacker
from env.honeypot_env import HoneypotEnv
from env.service_simulator import run_all_services
from train import BEST_SAVE_PATH, FINETUNE_BEST_PATH, finetune, train as run_training

try:
    from evaluation.benchmark import run_benchmark
except ImportError:
    run_benchmark = None 


def run_service_mode() -> None:
    try:
        asyncio.run(run_all_services())
    except KeyboardInterrupt:
        print("\n[AdaptTrap] Services stopped cleanly.")


def run_env_test() -> None:
    """
    Smoke test: runs 3 episodes with random actions.
    Prints per-step info directly.
    """
    env = HoneypotEnv(fast_training=True)
    total_rewards: list[float] = []

    try:
        for episode in range(3):
            _, reset_info = env.reset()
            episode_reward = 0.0
            step = 0

            print(f"\n{'=' * 60}")
            print(f" Episode {episode + 1} / 3 [random policy] attacker={reset_info['attacker']}")
            print(f"{'=' * 60}")
            print(
                f" {'Step':>4} {'Action':>12} {'Susp':>6} {'Depth':>6} "
                f"{'Dur':>7} {'Rew':>7} {'Flag':>5}"
            )
            print(f" {'-' * 58}")

            while True:
                step += 1
                action = env.action_space.sample()
                _, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                flag_str = "YES" if info["flagged"] else "-"
                print(
                    f" {step:4d} "
                    f"[{action[0]},{action[1]},{action[2]}]{' ':>6} "
                    f"{info['suspicion']:6.3f} "
                    f"{info['depth']:6.0f} "
                    f"{info['duration']:6.2f}s "
                    f"{reward:7.3f} "
                    f"{flag_str:>5}"
                )

                if terminated or truncated:
                    reason = "TERMINATED (caught 5x)" if terminated else "TRUNCATED (max steps)"
                    print(f"\n [{reason}]")
                    break

            print(f" Episode {episode + 1} total reward: {episode_reward:.3f}")
            total_rewards.append(episode_reward)

    finally:
        env.close()

    mean_reward = float(np.mean(total_rewards)) if total_rewards else 0.0
    print(f"\n[env_test] Mean reward over 3 episodes: {mean_reward:.3f}")
    print("[env_test] If rewards are non-zero and env closes cleanly, base loop is healthy.")


def run_step1_smoke() -> None:
    """
    Validation:
    - env boots
    - all three attacker names reset successfully
    - one random step executes for each attacker
    """
    env = HoneypotEnv(fast_training=True)

    try:
        for attacker_name in ["recon_probe", "scripted_exploit", "ai_probe"]:
            _, _ = env.reset(attacker_name=attacker_name)
            action = env.action_space.sample()
            _, reward, _, _, step_info = env.step(action)

            print(
                f"[step1_smoke] attacker={attacker_name} "
                f"reward={reward:.3f} "
                f"susp={step_info['suspicion']:.3f} "
                f"depth={step_info['depth']:.1f} "
                f"flagged={step_info['flagged']}"
            )
    finally:
        env.close()


def run_level1_baseline() -> None:
    scanner = Level1Scanner(target="172.20.0.10")
    scanner.measure_baseline(runs=3)


def run_level2_baseline() -> None:
    attacker = Level2ScriptedAttacker(target="172.20.0.10")
    attacker.measure_baseline(runs=3)


def run_finetune_mode() -> None:
    """
    Fine-tune from the best training checkpoint.

    Loads model_best.pt, runs 500 additional episodes with:
      - Adaptive entropy (per-head, per-attacker targets)
      - Logit clipping active in actor_critic.py
      - Lower LR: trunk frozen-ish, policy heads re-explore
      - Phase 3 only (no curriculum warmup)

    Usage:
        python main.py finetune
        python main.py finetune logs/model_best.pt       # explicit checkpoint
        python main.py finetune logs/model_latest.pt     # or latest
    """
    ckpt_path = Path(sys.argv[2]) if len(sys.argv) > 2 else BEST_SAVE_PATH
    finetune(checkpoint_path=ckpt_path)


def run_eval_mode() -> None:
    if run_benchmark is None:
        raise RuntimeError(
            "evaluation.benchmark could not be imported. "
            "Create evaluation/benchmark.py and ensure it exposes run_benchmark()."
        )

    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else None
    run_tag = sys.argv[3] if len(sys.argv) > 3 else "run1"

    if ckpt_path is None:
        ckpt_path = str(FINETUNE_BEST_PATH) if FINETUNE_BEST_PATH.exists() else str(BEST_SAVE_PATH)

    seed = 42 if run_tag == "run1" else 123
    output_path = f"logs/eval_results_{run_tag}.json"

    result = run_benchmark(
        checkpoint_path=ckpt_path,
        episodes_per_attacker=50,
        seed=seed,
        output_path=output_path,
        verbose=True,
    )

    print(f"\n[eval] Benchmark complete. Checkpoint: {ckpt_path}")
    print(f"[eval] Run tag: {run_tag} | seed={seed} | output={output_path}")
    print("[eval] Scoreboard:")

    for defender_name, block in result["scoreboard"].items():
        overall = block.get("overall", {})
        print(
            f"  - {defender_name}: "
            f"reward={overall.get('mean_total_raw_reward', 0.0):.4f}, "
            f"deception_ep={overall.get('deception_success_rate_episodes', 0.0):.4f}, "
            f"detection_steps={overall.get('detection_rate_steps', 0.0):.4f}, "
            f"completed={block.get('episodes_completed', 0)}, "
            f"failed={block.get('episodes_failed', 0)}"
        ) 

    delta = result.get("rl_delta_overall_vs_best_baseline", {})
    if delta:
        print("\n[eval] RL delta vs best non-RL baseline:")
        for metric_name, metric_block in delta.items():
            print(
                f"  - {metric_name}: "
                f"rl={metric_block.get('rl', 0.0):.4f}, "
                f"baseline={metric_block.get('baseline', 0.0):.4f}, "
                f"abs_delta={metric_block.get('abs_delta', 0.0):.4f}, "
                f"pct_delta={metric_block.get('pct_delta', 0.0)}"
            )


def print_help() -> None:
    print("[main] Available modes:")
    print("  service  - run raw local services")
    print("  env_test - random-policy smoke test")
    print("  step1_smoke - one-step validation for all attackers")
    print("  level1 - Level 1 baseline utility")
    print("  level2 - Level 2 baseline utility")
    print("  train - train RL defender from scratch (1500 eps, static entropy)")
    print("  finetune  - fine-tune from checkpoint (500 eps, adaptive entropy)")
    print("  usage: python main.py finetune [optional_checkpoint_path]")
    print("  eval - run evaluation benchmark")
    print(" usage: python main.py eval [optional_checkpoint_path][run1][run2]")


def main() -> None:
    mode = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "env_test"

    commands: dict[str, Callable[[], None]] = {
        "service":     run_service_mode,
        "env_test":    run_env_test,
        "step1_smoke": run_step1_smoke,
        "level1":      run_level1_baseline,
        "level2":      run_level2_baseline,
        "train":       run_training,
        "finetune":    run_finetune_mode,
        "eval":        run_eval_mode,
        "help":        print_help,
        "--help":      print_help,
        "-h":          print_help,
    }

    handler = commands.get(mode)
    if handler is None:
        print(f"[main] Unknown mode: '{mode}'")
        print_help()
        raise SystemExit(1)

    handler()


if __name__ == "__main__":
    main()