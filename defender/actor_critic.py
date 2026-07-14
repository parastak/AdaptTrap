# defender/actor_critic.py

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


ATTACKER_ORDER = ["recon_probe", "scripted_exploit", "ai_probe"]
ATTACKER_IDX = {a: i for i, a in enumerate(ATTACKER_ORDER)}
N_ATTACKERS = len(ATTACKER_ORDER)

DETECTION_PENALTY = {0: -1.5, 1: -2.0, 2: -3.5,}
EVASION_BONUS = {0: 1.0, 1: 1.2, 2: 2.5, } 


ENTROPY_COEFF = {
    0: 0.045,   # recon_probe
    1: 0.035,   # scripted_exploit
    2: 0.020,   # ai_probe
}

ENTROPY_TARGETS = {
    0: {"profile": 0.80, "temporal": 0.50, "ports": 0.20},
    1: {"profile": 0.40, "temporal": 0.30, "ports": 0.15},
    2: {"profile": 0.70, "temporal": 0.40, "ports": 0.18},
}

ENTROPY_ADAPTIVE_SCALE = {
    0: 0.12, # nmap moderate push
    1: 0.25, # scripted 
    2: 0.35,
}

# logit clip 
LOGIT_CLIP = 5.0


class AttackerPolicyHead(nn.Module):
    """
    One policy head per attacker type.
    Input: shared features
    Output:
        - profile logits (4)
        - temporal logits (4)
        - port-config logits (3)
    """

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 32)
        self.act = nn.ReLU()
        self.head_profile = nn.Linear(32, 4)
        self.head_temporal = nn.Linear(32, 4)
        self.head_ports = nn.Linear(32, 3)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.act(self.fc(features))
        return (
            self.head_profile(x),
            self.head_temporal(x),
            self.head_ports(x),
        )


class AttackerValueHead(nn.Module):
    """
    One value head per attacker type.
    Input: shared features
    Output: scalar state value
    """

    VALUE_SCALE = 6.0

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.net(features)
        return torch.tanh(raw) * self.VALUE_SCALE


class ActorCritic(nn.Module):
    """
    AdaptTrap Actor-Critic with:
    - shared trunk
    - separate policy heads per attacker
    - separate value heads per attacker

    Only the selected attacker's policy/value heads receive gradients.
    The shared trunk is trained on all attackers.
    """

    def __init__(self, state_dim: int = 12, n_attackers: int = N_ATTACKERS):
        super().__init__()
        self.n_attackers = n_attackers

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(p=0.04),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.policy_heads = nn.ModuleList(
            [AttackerPolicyHead(feature_dim=64) for _ in range(n_attackers)]
        )
        self.value_heads = nn.ModuleList(
            [AttackerValueHead(feature_dim=64) for _ in range(n_attackers)]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        for head in self.policy_heads:
            nn.init.orthogonal_(head.fc.weight, gain=np.sqrt(2))
            nn.init.zeros_(head.fc.bias)
            for out_layer in (head.head_profile, head.head_temporal, head.head_ports):
                nn.init.orthogonal_(out_layer.weight, gain=0.01)
                nn.init.zeros_(out_layer.bias)

        for value_head in self.value_heads:
            for m in value_head.net:
                if isinstance(m, nn.Linear):
                    # Smaller gain for value heads â€” cold-start predictions near 0
                    # prevents large initial critic loss from poisoning the trunk.
                    nn.init.orthogonal_(m.weight, gain=0.1)
                    nn.init.zeros_(m.bias)

    def _validate_attacker_idx(self, attacker_idx: int) -> None:
        if not (0 <= attacker_idx < self.n_attackers):
            raise ValueError(
                f"attacker_idx must be in [0, {self.n_attackers - 1}], got {attacker_idx}"
            )

    def _safe_dist(self, logits: torch.Tensor) -> Categorical:
        logits = torch.clamp(logits, -8.0, 8.0)
        return Categorical(logits=logits)


    def forward(
        self, state: torch.Tensor, attacker_idx: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        self._validate_attacker_idx(attacker_idx)
        features = self.trunk(state)
        value    = self.value_heads[attacker_idx](features)
        lp, lt, lc = self.policy_heads[attacker_idx](features)
        lp = torch.clamp(lp, -LOGIT_CLIP, LOGIT_CLIP)
        lt = torch.clamp(lt, -LOGIT_CLIP, LOGIT_CLIP)
        lc = torch.clamp(lc, -LOGIT_CLIP, LOGIT_CLIP)

        return value, (lp, lt, lc)

    def get_action(
        self, state: torch.Tensor, attacker_idx: int
    ) -> tuple[list[int], list[torch.Tensor], torch.Tensor]:
        value, (lp, lt, lc) = self(state, attacker_idx)

        dist_p   = self._safe_dist(lp)
        dist_t   = self._safe_dist(lt)
        dist_cfg = self._safe_dist(lc)

        act_p   = dist_p.sample()
        act_t   = dist_t.sample()
        act_cfg = dist_cfg.sample()

        log_probs = [
            dist_p.log_prob(act_p),
            dist_t.log_prob(act_t),
            dist_cfg.log_prob(act_cfg),
        ]
        actions = [act_p.item(), act_t.item(), act_cfg.item()]
        return actions, log_probs, value


    def evaluate_actions(
        self,
        states:       torch.Tensor,
        actions:      torch.Tensor,
        attacker_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:


        values, (lp, lt, lc) = self(states, attacker_idx)

        dist_p   = self._safe_dist(lp)
        dist_t   = self._safe_dist(lt)
        dist_cfg = self._safe_dist(lc)

        total_log_probs = (
            dist_p.log_prob(actions[:, 0])
            + dist_t.log_prob(actions[:, 1])
            + dist_cfg.log_prob(actions[:, 2])
        )

        entropy_p   = dist_p.entropy()
        entropy_t   = dist_t.entropy()
        entropy_cfg = dist_cfg.entropy()

        entropy_parts = {
            "profile":  entropy_p,
            "temporal": entropy_t,
            "ports":    entropy_cfg,
        }

        return (
            values.squeeze(-1),
            total_log_probs,
            entropy_p + entropy_t + entropy_cfg,
            entropy_parts,
        )
        

    @torch.no_grad()
    def act_greedy(
        self, state: torch.Tensor, attacker_idx: int
    ) -> tuple[list[int], torch.Tensor]:

        value, (lp, lt, lc) = self(state, attacker_idx)
        act_p   = torch.argmax(lp,  dim=-1)
        act_t   = torch.argmax(lt,  dim=-1)
        act_cfg = torch.argmax(lc,  dim=-1)
        actions = [act_p.item(), act_t.item(), act_cfg.item()]
        return actions, value

   
    def head_parameter_count(self) -> dict[str, int]:
        return {
            "trunk":        sum(p.numel() for p in self.trunk.parameters()),
            "policy_heads": sum(p.numel() for p in self.policy_heads.parameters()),
            "value_heads":  sum(p.numel() for p in self.value_heads.parameters()),
            "total":        sum(p.numel() for p in self.parameters()),
        }