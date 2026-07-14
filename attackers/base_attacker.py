from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class AttackerSessionResult:
    target: str
    attacker_type: str
    session_duration_seconds: float
    suspicion_score: float
    suspicion_reasons: list[str]
    flagged_as_honeypot: bool
    interaction_depth: int
    decisions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class BaseAttacker:
    name: str = "base_attacker"

    def __init__(
        self,
        target: str,
        ports: dict,
        fast_training: bool,
        simulated_latency_ms: float,
        active_profile_name: str,
        active_port_config: dict,
        attacker_profile: dict,
    ) -> None:
        self.target = target
        self.ports = ports
        self.fast_training = fast_training
        self.simulated_latency_ms = simulated_latency_ms
        self.active_profile_name = active_profile_name
        self.active_port_config = active_port_config
        self.attacker_profile = attacker_profile

    def run_session(self, verbose: bool = False) -> AttackerSessionResult:
        raise NotImplementedError("Subclasses must implement run_session().")