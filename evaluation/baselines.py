# evaluation/baselines.py
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from defender.actor_critic import ATTACKER_IDX, ActorCritic

ATTACKER_NAMES = ("recon_probe", "scripted_exploit", "ai_probe")
N_PROFILES = 4
N_TEMPORALS = 4
N_PORT_CFGS = 3
EXPECTED_OBS_DIM = 12

OBS_IDX_PROFILE = 0
OBS_IDX_TEMPORAL = 1
OBS_IDX_SSH = 2
OBS_IDX_HTTP = 3
OBS_IDX_REDIS = 4
OBS_IDX_SUSPICION = 5
OBS_IDX_DEPTH = 6
OBS_IDX_SESSION_PROGRESS = 7
OBS_IDX_CONSEC_DETECTS = 8


def validate_attacker_name(attacker_type: str) -> None:
    if attacker_type not in ATTACKER_NAMES:
        raise ValueError(
            f"Unknown attacker_type '{attacker_type}'. Valid: {list(ATTACKER_NAMES)}"
        )


def _ensure_int_like(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer action index, got bool")
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(f"{name} must be an integer action index, got {type(value).__name__}")


def validate_action(action: Sequence[int]) -> list[int]:
    if len(action) != 3:
        raise ValueError(f"Action must contain 3 integers, got {action}")

    p = _ensure_int_like(action[0], "profile")
    t = _ensure_int_like(action[1], "temporal")
    c = _ensure_int_like(action[2], "port_config")

    if not (0 <= p < N_PROFILES):
        raise ValueError(f"Invalid profile index: {p}")
    if not (0 <= t < N_TEMPORALS):
        raise ValueError(f"Invalid temporal index: {t}")
    if not (0 <= c < N_PORT_CFGS):
        raise ValueError(f"Invalid port config index: {c}")

    return [p, t, c]


class BaseDefender(ABC):
    name: str = "base"

    @abstractmethod
    def reset(self, attacker_type: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def act(self, obs: np.ndarray, info: dict | None = None) -> list[int]:
        raise NotImplementedError


@dataclass
class StaticDefender(BaseDefender):
    fixed_action: tuple[int, int, int] = (0, 0, 0)
    name: str = "static"

    def __post_init__(self) -> None:
        self._action = validate_action(self.fixed_action)

    def reset(self, attacker_type: str) -> None:
        validate_attacker_name(attacker_type)

    def act(self, obs: np.ndarray, info: dict | None = None) -> list[int]:
        _ = obs, info
        return self._action[:]


@dataclass
class RandomDefender(BaseDefender):
    seed: int = 42
    name: str = "random"
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def reset(self, attacker_type: str) -> None:
        validate_attacker_name(attacker_type)

    def act(self, obs: np.ndarray, info: dict | None = None) -> list[int]:
        _ = obs, info
        return [
            self.rng.randrange(N_PROFILES),
            self.rng.randrange(N_TEMPORALS),
            self.rng.randrange(N_PORT_CFGS),
        ]


@dataclass
class RuleBasedDefender(BaseDefender):
    name: str = "rule_based"
    attacker_type: str = field(init=False, default="recon_probe")
    prev_flagged: bool = field(init=False, default=False)
    prev_suspicion: float = field(init=False, default=0.0)

    def reset(self, attacker_type: str) -> None:
        validate_attacker_name(attacker_type)
        self.attacker_type = attacker_type
        self.prev_flagged = False
        self.prev_suspicion = 0.0

    def act(self, obs: np.ndarray, info: dict | None = None) -> list[int]:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 1 or obs.shape[0] != EXPECTED_OBS_DIM:
            raise ValueError(
                f"Observation must be a 1D vector of length {EXPECTED_OBS_DIM}, "
                f"got shape {obs.shape}"
            )

        last_suspicion = float(obs[OBS_IDX_SUSPICION])
        consec_detects_norm = float(obs[OBS_IDX_CONSEC_DETECTS])

        if info is not None:
            last_suspicion = float(info.get("suspicion", last_suspicion))
            self.prev_flagged = bool(info.get("flagged", False))
            self.prev_suspicion = last_suspicion

        high_risk = (
            last_suspicion > 0.30
            or consec_detects_norm > 0.0
            or self.prev_flagged
        )

        if self.attacker_type == "recon_probe":
            action = [0, 1, 2] if high_risk else [0, 0, 1]
        elif self.attacker_type == "scripted_exploit":
            action = [1, 1, 1] if high_risk else [1, 3, 1]
        elif self.attacker_type == "ai_probe":
            action = [1, 1, 2] if (high_risk or last_suspicion > 0.20) else [1, 3, 1]
        else:
            action = [0, 0, 0]

        return validate_action(action)


class RLGreedyDefender(BaseDefender):
    name = "rl_greedy"

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        state_dim: int = EXPECTED_OBS_DIM,
    ) -> None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        self.device = torch.device(device)
        self.model = ActorCritic(state_dim=state_dim)

        payload = torch.load(
            checkpoint,
            map_location=self.device,
            weights_only=False,
        )
        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint at '{checkpoint}' is not a valid dict payload")
        if "model_state_dict" not in payload:
            raise ValueError(
                f"Checkpoint at '{checkpoint}' does not contain 'model_state_dict'"
            )

        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self._attacker_idx = 0

    def reset(self, attacker_type: str) -> None:
        if attacker_type not in ATTACKER_IDX:
            raise ValueError(f"Unknown attacker_type '{attacker_type}'")
        self._attacker_idx = ATTACKER_IDX[attacker_type]

    @torch.no_grad()
    def act(self, obs: np.ndarray, info: dict | None = None) -> list[int]:
        _ = info
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 1 or obs.shape[0] != EXPECTED_OBS_DIM:
            raise ValueError(
                f"Observation must be 1D with length {EXPECTED_OBS_DIM}, got shape {obs.shape}"
            )

        state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions, _ = self.model.act_greedy(state, attacker_idx=self._attacker_idx)
        return validate_action(actions)


def build_defender(
    defender_name: str,
    *,
    checkpoint_path: str | None = None,
    seed: int = 42,
    device: str = "cpu",
) -> BaseDefender:
    normalized = defender_name.lower().strip().replace("-", "_")

    aliases = {
        "static": "static",
        "random": "random",
        "rule": "rulebased",
        "rule_based": "rulebased",
        "rulebased": "rulebased",
        "rl": "rlgreedy",
        "rl_greedy": "rlgreedy",
        "rlgreedy": "rlgreedy",
        "trained": "rlgreedy",
        "trained_rl": "rlgreedy",
    }

    canonical = aliases.get(normalized)
    if canonical is None:
        raise ValueError(
            f"Unknown defender '{defender_name}'. "
            f"Valid options: static | random | rule_based | rl_greedy"
        )

    if canonical == "static":
        return StaticDefender()

    if canonical == "random":
        return RandomDefender(seed=seed)

    if canonical == "rulebased":
        return RuleBasedDefender()

    if canonical == "rlgreedy":
        if not checkpoint_path:
            raise ValueError("checkpoint_path is required for RL defender evaluation")
        return RLGreedyDefender(
            checkpoint_path=checkpoint_path,
            device=device,
        )

    raise RuntimeError(f"Unhandled canonical defender '{canonical}'")