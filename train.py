from __future__ import annotations

import json
import random
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from defender.actor_critic import ATTACKER_IDX, ATTACKER_ORDER, ENTROPY_ADAPTIVE_SCALE, ENTROPY_COEFF, ENTROPY_TARGETS, ActorCritic
from env.honeypot_env import HoneypotEnv


# ------- Hyperparameters ----
N_EPISODES       = 1500
GAMMA            = 0.99
GAE_LAMBDA       = 0.85
LR               = 3e-4
VALUE_LOSS_COEFF = 0.5
GRAD_CLIP_POLICY = 0.5
GRAD_CLIP_CRITIC = 1.0
LOG_EVERY        = 10
STEP_DEBUG_EPS   = 3
SEED             = 42

# ---- Curriculum boundaries (episode fraction) ------

PHASE1_END = 0.20   
PHASE2_END = 0.55   
PHASE3_WEIGHTS = [0.20, 0.35, 0.45] 


#------Fine tune hyperparameters---------------------

FINETUNE_EPISODES       = 500
FINETUNE_SEED           = SEED + 1          
FINETUNE_LR_SCALE       = 0.10             # overall scale vs. original LR
FINETUNE_TRUNK_SCALE    = 0.30             
FINETUNE_CRITIC_SCALE   = 0.15            
FINETUNE_PHASE3_WEIGHTS = [0.25, 0.35, 0.40]  # slightly more scripted to unlock it
FINETUNE_STEP_DEBUG_EPS = 2               # fewer step-detail episodes in finetune


# ------------Critic stabilisation ------------------------
HUBER_DELTA          = 1.0
RETURN_NORM_WINDOW   = 200
CRITIC_LR_SCALE      = 0.10
CRITIC_DAMPEN_FACTOR = 0.15
CRITIC_DAMPEN_EPS    = 60


#--------- Alert thresholds ----------------------------------------
HEAVY_CLIP_MULTIPLIER    = 2.5
HIGH_DETECT_RATE         = 0.80
HIGH_CRITIC_LOSS         = 1.25
ENTROPY_ALERT_THRESHOLD = 1.2

# ----------- Identity ---------------------------
STATE_DIM        = 12
ARCHITECTURE_NAME= "shared_trunk_3_policy_heads_3_value_heads"
ALGORITHM_NAME   = "custom_actor_critic_gae"
PROJECT_VERSION  = "3.7"                   
STAGE_NAME       = "curriculum_20_55_phase3_45ai"
FINETUNE_STAGE   = "finetune_adaptive_entropy_logit_clip"

LOG_DIR             = Path("logs")
BEST_SAVE_PATH      = LOG_DIR / "model_best2.pt"
LATEST_SAVE_PATH    = LOG_DIR / "model_latest2.pt"
FINETUNE_BEST_PATH  = LOG_DIR / "model_finetuned_best.pt"
FINETUNE_LATEST_PATH= LOG_DIR / "model_finetuned_latest.pt"
LOG_PATH            = LOG_DIR / "training2.json"
FINETUNE_LOG_PATH   = LOG_DIR / "finetune.json"

W = 112

PORT_CFG_NAME  = {0: "all_open", 1: "no_redis", 2: "ssh_only"}
TEMPORAL_NAME  = {0: "normal  ", 1: "slow_srv", 2: "fast_cdn", 3: "load_sim"}
PROFILE_NAME   = {0: "ubuntu_web  ", 1: "debian_db  ", 2: "centos_api  ", 3: "windows_iis"}
ATTACKER_SHORT = {"recon_probe": "nmap", "scripted_exploit": "script", "ai_probe": "ai"}


# ------------ Seed -----------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------Curriculum sampler-------------------
def sample_attacker(episode: int, n_episodes: int, weights: list[float] | None = None) -> str:
    """
    During normal training: three-phase curriculum.
    During fine-tuning: pass weights directly to skip phases and always do Phase 3.
    """
    if weights is not None:
        return random.choices(ATTACKER_ORDER, weights=weights)[0]

    progress = episode / n_episodes
    if progress < PHASE1_END:
        return "recon_probe"
    if progress < PHASE2_END:
        return random.choice(["recon_probe", "scripted_exploit"])
    return random.choices(
        ["recon_probe", "scripted_exploit", "ai_probe"],
        weights=PHASE3_WEIGHTS,
    )[0]


#  GAE 
def compute_gae(
    rewards: list[float],
    values:  list[float],
    dones:   list[bool],
    gamma:   float,
    lam:     float,
) -> tuple[list[float], list[float]]:
    advantages: list[float] = []
    gae = 0.0
    next_value = 0.0
    for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
        mask  = 1.0 - float(done)
        delta = reward + gamma * next_value * mask - value
        gae   = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        next_value = value
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


#  Per-attacker return normaliser 
class ReturnNormaliser:
    """
    Online running-statistics return normaliser with separate buffers per attacker.
    Prevents scripted/ai_probe return distributions from contaminating nmap targets.
    """

    def __init__(self, window: int = RETURN_NORM_WINDOW) -> None:
        self._window  = window
        self._buffers: dict[int, list[float]] = {i: [] for i in range(len(ATTACKER_ORDER))}

    def update_and_normalise(self, returns: list[float], attacker_idx: int) -> torch.Tensor:
        buf = self._buffers[attacker_idx]
        buf.extend(returns)

        if len(buf) > self._window:
            del buf[: len(buf) - self._window]

        raw_t = torch.tensor(returns, dtype=torch.float32)

        if len(buf) < 10:
            return torch.clamp(raw_t, -6.0, 6.0)

        arr  = np.array(buf, dtype=np.float32)
        mean = float(np.mean(arr))
        std  = max(float(np.std(arr)), 0.5)

        normed = (raw_t - mean) / std
        return torch.clamp(normed, -4.0, 4.0)


#  Critic LR damper 
class CriticLRDamper:
    """
    Temporarily reduces critic LR after curriculum phase boundaries.
    The value head just entered an OOD regime; a gentle ramp avoids
    bootstrap explosions that cause grad-norm spikes.
    """

    def __init__(
        self,
        base_lr:       float,
        dampen_factor: float = CRITIC_DAMPEN_FACTOR,
        warmup_eps:    int   = CRITIC_DAMPEN_EPS,
    ) -> None:
        self._base     = base_lr
        self._factor   = dampen_factor
        self._warmup   = warmup_eps
        self._countdown = 0

    def trigger(self) -> None:
        self._countdown = self._warmup

    def get_lr(self) -> float:
        if self._countdown <= 0:
            return self._base
        self._countdown -= 1
        t     = 1.0 - (self._countdown / self._warmup)
        scale = self._factor + (1.0 - self._factor) * t
        return self._base * scale

    def apply(self, optimizer: Adam, group_index: int) -> None:
        optimizer.param_groups[group_index]["lr"] = self.get_lr()


#  Gradient clipping (per component) 
def clip_grads_per_component(
    model:        ActorCritic,
    clip_policy:  float,
    clip_critic:  float,
) -> tuple[float, float, bool, bool]:
    """
    Clips policy (trunk + policy_heads) and critic (value_heads) separately.
    Returns pre-clip norms and whether each was clipped.
    """
    policy_params = list(model.trunk.parameters()) + list(model.policy_heads.parameters())
    critic_params = list(model.value_heads.parameters())

    policy_preclip = float(nn.utils.clip_grad_norm_(policy_params, clip_policy))
    critic_preclip = float(nn.utils.clip_grad_norm_(critic_params, clip_critic))

    return (
        policy_preclip,
        critic_preclip,
        policy_preclip > clip_policy,
        critic_preclip > clip_critic,
    )


def compute_adaptive_entropy_loss(
    entropy_parts: dict[str, torch.Tensor],
    attacker_idx:  int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute per-head adaptive entropy loss.

    For each action dimension (profile, temporal, ports):
        effective_coeff = base_coeff + adaptive_scale * max(0, target - current_entropy)

    When entropy is healthy (above target): coefficient = base_coeff only.
    When entropy has collapsed below target: coefficient spikes proportionally,
    pushing the policy back toward exploration.

    Returns:
        adaptive_entropy_loss : scalar tensor, subtract from total loss
        coeff_info            : dict of effective coefficients for logging
    """
    targets = ENTROPY_TARGETS[attacker_idx]
    scale   = ENTROPY_ADAPTIVE_SCALE[attacker_idx]
    base    = ENTROPY_COEFF[attacker_idx]

    ep_mean = entropy_parts["profile"].mean()
    et_mean = entropy_parts["temporal"].mean()
    ec_mean = entropy_parts["ports"].mean()

    coeff_p = base + scale * max(0.0, targets["profile"]  - ep_mean.item())
    coeff_t = base + scale * max(0.0, targets["temporal"] - et_mean.item())
    coeff_c = base + scale * max(0.0, targets["ports"]    - ec_mean.item())

    ENTROPY_FLOOR = 0.05 
    floor_penalty_p = 0.5 * torch.relu(torch.tensor(ENTROPY_FLOOR) - ep_mean) ** 2
    floor_penalty_t = 0.5 * torch.relu(torch.tensor(ENTROPY_FLOOR) - et_mean) ** 2
    floor_penalty_c = 0.5 * torch.relu(torch.tensor(ENTROPY_FLOOR) - ec_mean) ** 2

    loss = (coeff_p * ep_mean + coeff_t * et_mean + coeff_c * ec_mean
        + floor_penalty_p + floor_penalty_t + floor_penalty_c)

    coeff_info = {
        "coeff_profile":  round(coeff_p, 5),
        "coeff_temporal": round(coeff_t, 5),
        "coeff_ports":    round(coeff_c, 5),
    }
    return loss, coeff_info



#  Logging helpers 
def _hdr(title: str = "") -> str:
    if title:
        pad = max(0, W - len(title) - 4)
        return f" ╔══ {title} {'═' * pad}╗"
    return " ╔" + "═" * (W - 2) + "╗"


def _ftr() -> str:
    return " ╚" + "═" * (W - 2) + "╝"


def _row(content: str) -> str:
    return f" ║ {content:<{W - 4}} ║"


def _div(char: str = "─") -> str:
    return " ╟" + char * (W - 2) + "╢"


def _bar(value: float, width: int = 10) -> str:
    value = max(0.0, min(1.0, value))
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_changes(changes: list[str]) -> str:
    return ",".join(changes) if changes else "·"


def _step_line(step: int, actions: list[int], info: dict[str, Any], norm_reward: float) -> str:
    flag = " ⚑ FLAGGED" if info.get("flagged") else ""
    change = _fmt_changes(info.get("changes", []))
    prof = PROFILE_NAME[actions[0]].strip()
    temp = TEMPORAL_NAME[actions[1]].strip()
    port = PORT_CFG_NAME[actions[2]].strip()
    bar = _bar(float(info["suspicion"]))
    return (
        f" St{step:02d} | {prof:<11} {temp:<8} {port:<8} | "
        f"Δ:[{change}] | susp={info['suspicion']:.2f}[{bar}] "
        f"dep={info['depth']:2.0f} dur={info['duration']:.2f}s "
        f"raw={info['raw_reward']:+.3f} norm={norm_reward:+.3f}{flag}"
    )


def _ep_header(ep: int, n_total: int, attacker_type: str, mode: str = "train") -> None:
    print()
    print(_hdr(f"[{mode.upper()}] Episode {ep:03d} / {n_total} [{attacker_type}] [step detail]"))
    print(_row(" St## | Profile     Temporal PortCfg  | Δ changes | susp[bar ] dep dur raw norm"))
    print(_div())


def _ep_summary_header(mode: str = "train") -> None:
    print()
    print(_hdr("Fine-Tune Progress" if mode == "finetune" else "Training Progress"))
    print(_row(
        f"{'Ep':>5} {'NormRew':>9} {'RawRew':>9} {'Susp':>6} {'Dur':>6} "
        f"{'Depth':>5} {'Det%':>5} {'Atk':>6} {'Entr':>6} "
        f"{'Hp':>5} {'Ht':>5} {'Hc':>5} "
        f"{'ActorL':>9} {'CritL':>9} {'PolGN':>7} {'ValGN':>7} {'Best':>5}"
    ))
    print(_div())


def _ep_summary_row(
    ep: int,
    ep_norm_reward: float,
    ep_raw_reward: float,
    mean_susp: float,
    mean_dur: float,
    mean_depth: float,
    detect_rate: float,
    attacker_type: str,
    entropy_val: float,
    entropy_p: float,
    entropy_t: float,
    entropy_c: float,
    actor_loss: float,
    critic_loss: float,
    policy_grad_norm: float,
    value_grad_norm: float,
    is_best: bool,
) -> str:
    best_str = "★ NEW" if is_best else ""
    atk_short = ATTACKER_SHORT.get(attacker_type, attacker_type[:6])
    return _row(
        f"{ep:5d} {ep_norm_reward:+9.2f} {ep_raw_reward:+9.2f} {mean_susp:6.3f} {mean_dur:5.2f}s "
        f"{mean_depth:5.1f} {detect_rate*100:5.1f}% {atk_short:>6} {entropy_val:6.3f} "
        f"{entropy_p:5.2f} {entropy_t:5.2f} {entropy_c:5.2f} "
        f"{actor_loss:+9.5f} {critic_loss:9.5f} {policy_grad_norm:7.4f} {value_grad_norm:7.4f} {best_str:<5}"
    )


#  Checkpoint helpers 
def _build_metadata(
    env:          HoneypotEnv,
    total_params: int,
    stage:        str = STAGE_NAME,
    extra:        dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "project": {
            "name":    "AdaptTrap",
            "version": PROJECT_VERSION,
            "stage":   stage,
        },
        "algorithm":    ALGORITHM_NAME,
        "architecture": ARCHITECTURE_NAME,
        "state_dim":    STATE_DIM,
        "action_space": [env.N_PROFILES, env.N_TEMPORAL, env.N_PORT_CFGS],
        "attacker_order": ATTACKER_ORDER,
        "training": {
            "n_episodes":       N_EPISODES,
            "max_steps":        env.MAX_SESSIONS,
            "gamma":            GAMMA,
            "gae_lambda":       GAE_LAMBDA,
            "lr":               LR,
            "value_loss_coeff": VALUE_LOSS_COEFF,
            "grad_clip_policy": GRAD_CLIP_POLICY,
            "grad_clip_critic": GRAD_CLIP_CRITIC,
            "huber_delta":      HUBER_DELTA,
            "seed":             SEED,
            "entropy_coeff":    ENTROPY_COEFF,
            "entropy_targets":  ENTROPY_TARGETS,
            "entropy_adaptive_scale": ENTROPY_ADAPTIVE_SCALE,
            "curriculum": {
                "phase1_end":    PHASE1_END,
                "phase2_end":    PHASE2_END,
                "phase3_weights":PHASE3_WEIGHTS,
            },
            "critic_loss":    "huber",
            "critic_returns": "per_attacker_online_normalised",
        },
        "environment": {
            "fast_training": True,
            "train_ports":   env.TRAIN_PORTS,
            "eval_ports":    env.EVAL_PORTS,
            "max_sessions":  env.MAX_SESSIONS,
            "max_depth":     env.MAX_DEPTH,
        },
        "model":     {"total_params": total_params},
        "artifacts": {
            "training_log":       str(LOG_PATH),
            "best_checkpoint":    str(BEST_SAVE_PATH),
            "latest_checkpoint":  str(LATEST_SAVE_PATH),
            "finetuned_best":     str(FINETUNE_BEST_PATH),
            "finetuned_latest":   str(FINETUNE_LATEST_PATH),
        },
    }
    if extra:
        meta.update(extra)
    return meta


def _save_checkpoint(
    path:              Path,
    model:             ActorCritic,
    episode:           int,
    best_raw_reward:   float,
    latest_raw_reward: float,
    total_params:      int,
    env:               HoneypotEnv,
    stage:             str = STAGE_NAME,
    extra_meta:        dict[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "model_state_dict":          model.state_dict(),
            "episode":                   episode,
            "best_raw_reward":           float(best_raw_reward),
            "latest_raw_episode_reward": float(latest_raw_reward),
            "metadata": _build_metadata(
                env=env,
                total_params=total_params,
                stage=stage,
                extra=extra_meta,
            ),
        },
        path,
    )

#------shared episode loop 
def _run_episode_loop(
    model:            ActorCritic,
    env:              HoneypotEnv,
    optimizer:        Adam,
    return_norm:      ReturnNormaliser,
    critic_damper:    CriticLRDamper | None,
    all_logs:         list[dict[str, Any]],
    best_raw_reward:  float,
    n_episodes:       int,
    best_save_path:   Path,
    latest_save_path: Path,
    total_params:     int,
    stage:            str,
    use_adaptive_entropy: bool,
    phase3_weights:   list[float] | None,
    step_debug_eps:   int,
    log_every:        int,
    mode:             str,
    extra_meta:       dict[str, Any] | None = None,
) -> float:
    """
    Shared episode loop used by both train() and finetune().

    Returns the best raw reward seen across all episodes in this call.
    """
    summary_printed = False

    eps_phase1 = int(n_episodes * PHASE1_END)
    eps_phase2 = int(n_episodes * PHASE2_END)

    try:
        for episode in range(1, n_episodes + 1):

            # Phase boundary notifications (only relevant during train)
            if mode == "train":
                if episode == eps_phase1:
                    print(f"\n{'-' * 60}")
                    print(f"CURRICULUM: Phase 2 starts (ep {episode}) :- scripted_exploit enters")
                    print(f"{'-' * 60}\n")
                    if critic_damper:
                        critic_damper.trigger()

                if episode == eps_phase2:
                    print(f"\n{'-' * 60}")
                    print(f"CURRICULUM: Phase 3 starts (ep {episode}) :- ai_probe enters weights={PHASE3_WEIGHTS}")
                    print(f"{'-' * 60}\n")
                    if critic_damper:
                        critic_damper.trigger()

            if critic_damper:
                critic_damper.apply(optimizer, group_index=2)

            # Episode setup
            attacker_type = sample_attacker(episode, n_episodes, weights=phase3_weights)
            attacker_idx  = ATTACKER_IDX[attacker_type]
            obs, _        = env.reset(attacker_name=attacker_type)
            obs_t         = torch.tensor(obs, dtype=torch.float32)

            traj_obs:          list[np.ndarray] = []
            traj_actions:      list[list[int]]  = []
            traj_rewards_norm: list[float]      = []
            traj_rewards_raw:  list[float]      = []
            traj_values:       list[float]      = []
            traj_dones:        list[bool]       = []

            ep_suspicions: list[float] = []
            ep_depths:     list[float] = []
            ep_durations:  list[float] = []
            ep_detections  = 0
            step_errors    = 0
            ep_raw_reward  = 0.0

            show_steps = episode <= step_debug_eps
            if show_steps:
                _ep_header(episode, n_episodes, attacker_type, mode=mode)

            # Rollout
            model.eval()
            with torch.no_grad():
                step = 0
                while True:
                    step += 1
                    try:
                        actions, _, value = model.get_action(obs_t.unsqueeze(0), attacker_idx)
                        obs_next, reward, terminated, truncated, info = env.step(
                            np.array(actions, dtype=np.int64)
                        )

                        if show_steps:
                            print(_step_line(step, actions, info, float(reward)))

                        traj_obs.append(obs_t.numpy())
                        traj_actions.append(actions)
                        traj_rewards_norm.append(float(reward))
                        traj_rewards_raw.append(float(info["raw_reward"]))
                        traj_values.append(float(value.squeeze().item()))
                        traj_dones.append(bool(terminated or truncated))

                        ep_suspicions.append(float(info["suspicion"]))
                        ep_depths.append(float(info["depth"]))
                        ep_durations.append(float(info["duration"]))
                        ep_raw_reward += float(info["raw_reward"])

                        if info["flagged"]:
                            ep_detections += 1

                        obs_t = torch.tensor(obs_next, dtype=torch.float32)

                        if terminated or truncated:
                            if terminated and show_steps:
                                print(_row("TERMINATED early - caught 5x consecutively"))
                            break

                    except Exception as e:
                        step_errors += 1
                        print(f"!! STEP ERROR Ep{episode:03d} St{step:02d}: {e}")
                        traceback.print_exc()
                        break

            if show_steps:
                ep_total     = sum(traj_rewards_norm) if traj_rewards_norm else 0.0
                mean_susp_d  = float(np.mean(ep_suspicions)) if ep_suspicions else 0.0
                mean_dep_d   = float(np.mean(ep_depths)) if ep_depths else 0.0
                print(_div())
                print(_row(
                    f" Ep{episode:03d} summary - steps={step} "
                    f"total_norm={ep_total:+.2f} flags={ep_detections}/{step} "
                    f"mean_susp={mean_susp_d:.3f} mean_depth={mean_dep_d:.1f} "
                    f"attacker={attacker_type}"
                ))
                print(_ftr())

            if not traj_rewards_norm:
                print(f"!! Ep{episode:03d}: Empty trajectory - skipping update.")
                continue

            # GAE
            adv_norm, _    = compute_gae(traj_rewards_norm, traj_values, traj_dones, GAMMA, GAE_LAMBDA)
            _, returns_raw = compute_gae(traj_rewards_raw,  traj_values, traj_dones, GAMMA, GAE_LAMBDA)
            returns_t      = return_norm.update_and_normalise(returns_raw, attacker_idx)

            adv_t = torch.tensor(adv_norm, dtype=torch.float32)
            if len(adv_t) > 1:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            # Forward pass
            model.train()
            obs_batch     = torch.tensor(np.array(traj_obs),     dtype=torch.float32)
            actions_batch = torch.tensor(np.array(traj_actions), dtype=torch.long)

            values_pred, total_log_probs, entropy, entropy_parts = model.evaluate_actions(
                obs_batch, actions_batch, attacker_idx
            )

            # Loss
            actor_loss  = -(total_log_probs * adv_t.detach()).mean()
            critic_loss = nn.functional.huber_loss(
                values_pred, returns_t, delta=HUBER_DELTA, reduction="mean"
            )

            if use_adaptive_entropy:
                # Per-head adaptive entropy: coefficient spikes when entropy
                # drops below attacker-specific target floors.
                entropy_loss, coeff_info = compute_adaptive_entropy_loss(entropy_parts, attacker_idx)
            else:
                # Static entropy during initial training
                ent_coeff    = ENTROPY_COEFF[attacker_idx]
                entropy_loss = ent_coeff * entropy.mean()
                coeff_info   = {}

            total_loss = actor_loss + VALUE_LOSS_COEFF * critic_loss - entropy_loss

            # Backward
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            policy_grad_norm, value_grad_norm, policy_was_clipped, value_was_clipped = (
                clip_grads_per_component(model, GRAD_CLIP_POLICY, GRAD_CLIP_CRITIC)
            )
            optimizer.step()

            # Metrics
            ep_norm_reward = float(sum(traj_rewards_norm))
            mean_susp      = float(np.mean(ep_suspicions))
            mean_dur       = float(np.mean(ep_durations))
            mean_depth     = float(np.mean(ep_depths))
            detect_rate    = ep_detections / len(traj_rewards_norm)
            entropy_val    = float(entropy.mean().item())
            entropy_p      = float(entropy_parts["profile"].mean().item())
            entropy_t_val  = float(entropy_parts["temporal"].mean().item())
            entropy_c      = float(entropy_parts["ports"].mean().item())
            actor_l        = float(actor_loss.item())
            critic_l       = float(critic_loss.item())

            log_entry: dict[str, Any] = {
                "episode":            episode,
                "total_reward":       round(ep_norm_reward, 4),
                "raw_total_reward":   round(ep_raw_reward, 4),
                "mean_suspicion":     round(mean_susp, 4),
                "mean_duration_s":    round(mean_dur, 4),
                "mean_depth":         round(mean_depth, 4),
                "detect_rate":        round(detect_rate, 4),
                "actor_loss":         round(actor_l, 6),
                "critic_loss":        round(critic_l, 6),
                "entropy":            round(entropy_val, 6),
                "entropy_profile":    round(entropy_p, 6),
                "entropy_temporal":   round(entropy_t_val, 6),
                "entropy_ports":      round(entropy_c, 6),
                "policy_grad_norm":   round(policy_grad_norm, 6),
                "value_grad_norm":    round(value_grad_norm, 6),
                "policy_was_clipped": bool(policy_was_clipped),
                "value_was_clipped":  bool(value_was_clipped),
                "step_errors":        step_errors,
                "attacker_type":      attacker_type,
                "attacker_idx":       attacker_idx,
                "mode":               mode,
                "adaptive_entropy":   use_adaptive_entropy,
                "curriculum_phase": (
                    1 if episode / n_episodes < PHASE1_END
                    else 2 if episode / n_episodes < PHASE2_END
                    else 3
                ) if mode == "train" else 3,
                "critic_lr": optimizer.param_groups[2]["lr"],
            }
            log_entry.update(coeff_info)
            all_logs.append(log_entry)

            # Save checkpoints
            _save_checkpoint(
                path=latest_save_path, model=model, episode=episode,
                best_raw_reward=best_raw_reward, latest_raw_reward=ep_raw_reward,
                total_params=total_params, env=env, stage=stage, extra_meta=extra_meta,
            )

            is_best = ep_raw_reward > best_raw_reward
            if is_best:
                best_raw_reward = ep_raw_reward
                _save_checkpoint(
                    path=best_save_path, model=model, episode=episode,
                    best_raw_reward=best_raw_reward, latest_raw_reward=ep_raw_reward,
                    total_params=total_params, env=env, stage=stage, extra_meta=extra_meta,
                )

            # Console row
            if episode % log_every == 0 or episode == 1:
                if not summary_printed:
                    _ep_summary_header(mode=mode)
                    summary_printed = True

                print(_ep_summary_row(
                    episode, ep_norm_reward, ep_raw_reward,
                    mean_susp, mean_dur, mean_depth, detect_rate,
                    attacker_type, entropy_val, entropy_p, entropy_t_val, entropy_c,
                    actor_l, critic_l, policy_grad_norm, value_grad_norm, is_best,
                ))

            if episode % (log_every * 3) == 0:
                print(_ftr())
                summary_printed = False

            # Diagnostic alerts
            if entropy_val < ENTROPY_ALERT_THRESHOLD:
                print(
                    f" [Ep{episode:03d}][{mode}] [{attacker_type}] Entropy low "
                    f"({entropy_val:.3f}) â€” policy collapsing on this head"
                )
            if step_errors > 0:
                print(f"[Ep{episode:03d}] {step_errors} step errors")
            if detect_rate > HIGH_DETECT_RATE:
                print(f"[Ep{episode:03d}] HIGH detect rate {detect_rate * 100:.0f}%")
            if critic_l > HIGH_CRITIC_LOSS:
                print(f"[Ep{episode:03d}] Critic loss elevated {critic_l:.3f} (post-Huber)")
            if value_was_clipped and value_grad_norm > (GRAD_CLIP_CRITIC * HEAVY_CLIP_MULTIPLIER):
                print(
                    f"[Ep{episode:03d}] Value head heavily clipped "
                    f"(preclip={value_grad_norm:.4f}, clip={GRAD_CLIP_CRITIC:.2f})"
                )

    finally:
        if summary_printed:
            print(_ftr())

    return best_raw_reward

   

#  Training entry point 
def train() -> None:
    """
    Full training run from scratch: 1500 episodes, three-phase curriculum.
    Uses static entropy coefficients (adaptive entropy is reserved for fine-tune).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    print("\n" + "-" * W)
    print(
        f" AdaptTrap :-” RL Defender |‚ A2C + GAE |‚ v{PROJECT_VERSION} |‚ "
        f"Curriculum 20/55 â”‚ Static entropy"
    )
    print("-" * W)

    env   = HoneypotEnv(fast_training=True)
    model = ActorCritic(state_dim=STATE_DIM)

    ft_lr = LR
    optimizer = Adam(
        [
            {"params": model.trunk.parameters(),        "lr": ft_lr * 0.5},
            {"params": model.policy_heads.parameters(), "lr": ft_lr},
            {"params": model.value_heads.parameters(),  "lr": ft_lr * CRITIC_LR_SCALE},
        ],
        lr=ft_lr,
    )

    return_norm  = ReturnNormaliser(window=RETURN_NORM_WINDOW)
    critic_damper = CriticLRDamper(
        base_lr=ft_lr * CRITIC_LR_SCALE,
        dampen_factor=CRITIC_DAMPEN_FACTOR,
        warmup_eps=CRITIC_DAMPEN_EPS,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters    : {total_params:,}")
    print(f"Episodes      : {N_EPISODES} â”‚ Steps/ep: {env.MAX_SESSIONS}")
    print(f"LR={ft_lr}  Gamma={GAMMA}  Î»={GAE_LAMBDA}  GradClip(policy)={GRAD_CLIP_POLICY}  Seed={SEED}")
    print(f"Entropy (static): nmap={ENTROPY_COEFF[0]}  scripted={ENTROPY_COEFF[1]}  ai={ENTROPY_COEFF[2]}")
    print(f"Best checkpoint : {BEST_SAVE_PATH}")

    all_logs: list[dict[str, Any]] = []

    best_raw_reward = _run_episode_loop(
        model=model, env=env, optimizer=optimizer,
        return_norm=return_norm, critic_damper=critic_damper,
        all_logs=all_logs, best_raw_reward=-float("inf"),
        n_episodes=N_EPISODES,
        best_save_path=BEST_SAVE_PATH, latest_save_path=LATEST_SAVE_PATH,
        total_params=total_params, stage=STAGE_NAME,
        use_adaptive_entropy=False,
        phase3_weights=None,
        step_debug_eps=STEP_DEBUG_EPS, log_every=LOG_EVERY,
        mode="train",
    )

    _write_log(LOG_PATH, all_logs, env, total_params, stage=STAGE_NAME)
    env.close()

    print("\n" + "-" * W)
    print(f"Training complete:-‚ Best raw reward: {best_raw_reward:.2f} |‚ Checkpoint: {BEST_SAVE_PATH}")
    print("-" * W)


#  Fine-tune entry point 
def finetune(checkpoint_path: Path = BEST_SAVE_PATH) -> None:
    """
    Fine-tune from a saved checkpoint for 500 more episodes.

    What changes vs. train():
      - Loads model_best.pt (or any provided checkpoint) instead of init from scratch
      - Lower learning rates: trunk barely moves, policy heads re-open, critic stable
      - Phase 3 only (no warmup phases) â€” all attackers from episode 1
      - FINETUNE_PHASE3_WEIGHTS = [0.25, 0.35, 0.40] â€” slightly more scripted to push
        the locked scripted head back toward exploration
      - Adaptive entropy is ACTIVE â€” per-head coefficients spike when entropy is below
        attacker-specific target floors (the core fix for entropy collapse)
      - Different seed (SEED + 1) |” different episode ordering prevents replay of
        already-seen trajectories
      - Saves to model_finetuned_best.pt / model_finetuned_latest.pt |” does NOT
        overwrite the original training checkpoint
      - Logs to finetune.json â€” separate from training.json1
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(FINETUNE_SEED)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run `python main.py train` first, or pass a valid checkpoint path."
        )

    print("\n" + "-" * W)
    print(
        f" AdaptTrap| Fine-Tune |‚ Adaptive Entropy + Logit Clip |‚ v{PROJECT_VERSION} |‚ "
        f"{FINETUNE_EPISODES} eps from {checkpoint_path.name}"
    )
    print("-" * W)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = ActorCritic(state_dim=STATE_DIM)
    model.load_state_dict(ckpt["model_state_dict"])
    loaded_episode = ckpt.get("episode", "unknown")
    loaded_reward  = ckpt.get("best_raw_reward", "unknown")
    print(f"Loaded checkpoint: episode={loaded_episode}  best_raw_reward={loaded_reward}")

    env = HoneypotEnv(fast_training=True)

    # LR schedule: trunk barely moves, policy heads need room to re-explore,
    # critic is already calibrated so keep it tight.
    ft_lr = LR * FINETUNE_LR_SCALE
    optimizer = Adam(
        [
            {"params": model.trunk.parameters(),        "lr": ft_lr * FINETUNE_TRUNK_SCALE},
            {"params": model.policy_heads.parameters(), "lr": ft_lr},
            {"params": model.value_heads.parameters(),  "lr": ft_lr * FINETUNE_CRITIC_SCALE},
        ],
        lr=ft_lr,
    )

    return_norm = ReturnNormaliser(window=RETURN_NORM_WINDOW)
    # No critic_damper for finetune â€” no phase boundaries to cross,
    # and the critic is already warm. Flat LR throughout.

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters     : {total_params:,}")
    print(f"Episodes       : {FINETUNE_EPISODES} - ”‚ Steps/ep: {env.MAX_SESSIONS}")
    print(f"LR (policy)    : {ft_lr:.2e}  trunk: {ft_lr * FINETUNE_TRUNK_SCALE:.2e}  critic: {ft_lr * FINETUNE_CRITIC_SCALE:.2e}")
    print(f"Attacker weights: nmap={FINETUNE_PHASE3_WEIGHTS[0]}  scripted={FINETUNE_PHASE3_WEIGHTS[1]}  ai={FINETUNE_PHASE3_WEIGHTS[2]}")
    print(f"Entropy        : ADAPTIVE (per-head, per-attacker targets)")
    print(f"Best checkpoint: {FINETUNE_BEST_PATH}")

    all_logs: list[dict[str, Any]] = []

    extra_meta = {
        "finetune": {
            "source_checkpoint": str(checkpoint_path),
            "source_episode":    loaded_episode,
            "source_reward":     loaded_reward,
            "episodes":          FINETUNE_EPISODES,
            "lr_scale":          FINETUNE_LR_SCALE,
            "phase3_weights":    FINETUNE_PHASE3_WEIGHTS,
            "adaptive_entropy":  True,
        }
    }

    best_raw_reward = _run_episode_loop(
        model=model, env=env, optimizer=optimizer,
        return_norm=return_norm, critic_damper=None,
        all_logs=all_logs, best_raw_reward=-float("inf"),
        n_episodes=FINETUNE_EPISODES,
        best_save_path=FINETUNE_BEST_PATH, latest_save_path=FINETUNE_LATEST_PATH,
        total_params=total_params, stage=FINETUNE_STAGE,
        use_adaptive_entropy=True,
        phase3_weights=FINETUNE_PHASE3_WEIGHTS,
        step_debug_eps=FINETUNE_STEP_DEBUG_EPS, log_every=LOG_EVERY,
        mode="finetune",
        extra_meta=extra_meta,
    )

    _write_log(FINETUNE_LOG_PATH, all_logs, env, total_params, stage=FINETUNE_STAGE, extra=extra_meta)
    env.close()

    print("\n" + "-" * W)
    print(
        f"Fine-tune complete - ”‚ Best raw reward: {best_raw_reward:.2f} -”‚ "
        f"Checkpoint: {FINETUNE_BEST_PATH}"
    )
    print(
        f"Use `python main.py eval --checkpoint {FINETUNE_BEST_PATH}` "
        f"to benchmark the fine-tuned model."
    )
    print("-" * W)


#  Log writer
def _write_log(
    path:         Path,
    all_logs:     list[dict[str, Any]],
    env:          HoneypotEnv,
    total_params: int,
    stage:        str = STAGE_NAME,
    extra:        dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "metadata": _build_metadata(env=env, total_params=total_params, stage=stage, extra=extra),
        "attacker_trailing_means": {
            name: {
                "reward":           trailing_mean(all_logs, "total_reward",     name),
                "raw_reward":       trailing_mean(all_logs, "raw_total_reward", name),
                "detect_rate":      trailing_mean(all_logs, "detect_rate",      name),
                "critic_loss":      trailing_mean(all_logs, "critic_loss",      name),
                "policy_grad_norm": trailing_mean(all_logs, "policy_grad_norm", name),
                "value_grad_norm":  trailing_mean(all_logs, "value_grad_norm",  name),
                "entropy":          trailing_mean(all_logs, "entropy",          name),
                "entropy_profile":  trailing_mean(all_logs, "entropy_profile",  name),
                "entropy_temporal": trailing_mean(all_logs, "entropy_temporal", name),
                "entropy_ports":    trailing_mean(all_logs, "entropy_ports",    name),
            }
            for name in ATTACKER_ORDER
        },
        "episodes": all_logs,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[log] Written: {path}  ({len(all_logs)} episodes)")


# CLI entry point 
if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    elif cmd == "finetune":
        ckpt = Path(sys.argv[2]) if len(sys.argv) > 2 else BEST_SAVE_PATH
        finetune(checkpoint_path=ckpt)
    else:
        print(f"[train.py] Unknown command: {cmd}. Use 'train' or 'finetune [checkpoint_path]'.")