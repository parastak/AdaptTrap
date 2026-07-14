from __future__ import annotations

import random
import socket
import time
from dataclasses import dataclass
from typing import Callable

from attackers.base_attacker import BaseAttacker, AttackerSessionResult


@dataclass
class ProbeOutcome:
    name: str
    delta_suspicion: float
    depth_gain: int
    duration_gain: float
    notes: str


class Level3BanditAttacker(BaseAttacker):
    """
    Level 3 attacker: adaptive bandit-style probe learner.

    This attacker is stronger than the scripted Level 2 bot because it:
    - chooses probes adaptively within the session,
    - updates action values from observed outcomes,
    - tests timing and cross-service consistency,
    - stops early once suspicion crosses the threshold.

    It is still lightweight enough for CPU-only local training.
    """

    name = "ai_probe"

    ACTIONS = [
        "ssh_banner",
        "ssh_creds",
        "http_root",
        "http_traversal",
        "redis_ping",
        "redis_info",
        "latency_probe",
        "consistency_probe",
    ]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.q: dict[str, float] = {a: 0.0 for a in self.ACTIONS}
        self.n: dict[str, int] = {a: 0 for a in self.ACTIONS}

        self.suspicion_score = 0.0
        self.suspicion_reasons: list[str] = []
        self.decisions: list[str] = []
        self.interaction_depth = 0
        self._duration = 0.0

        self._detect_threshold = float(self.attacker_profile.get("detect_threshold", 0.25))
        self._timing_tolerance = float(self.attacker_profile.get("timing_tolerance", 0.1))
        self._patience = float(self.attacker_profile.get("patience", 1.0))

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _port_open(self, service: str) -> bool:
        return bool(self.active_port_config.get(service, True))

    def _pick_action(self) -> str:
        """
        Epsilon-greedy action selection.
        In fast training we keep slightly more exploration so the attacker
        does not collapse into a single probe too early.
        """
        eps = 0.20 if self.fast_training else 0.10
        if random.random() < eps:
            return random.choice(self.ACTIONS)
        return max(self.ACTIONS, key=lambda a: self.q[a])

    def _update_q(self, action: str, reward: float) -> None:
        self.n[action] += 1
        alpha = 1.0 / self.n[action]
        self.q[action] += alpha * (reward - self.q[action])

    def _record(self, action: str, outcome: ProbeOutcome) -> None:
        self.decisions.append(f"{action}: {outcome.notes}")
        self.suspicion_score = min(
            1.0,
            max(0.0, self.suspicion_score + outcome.delta_suspicion),
        )
        self.interaction_depth += outcome.depth_gain
        self._duration += max(0.0, outcome.duration_gain)

    def _raw_connect(
        self,
        port: int,
        send_data: bytes | None = None,
        timeout: float = 1.0,
    ) -> tuple[str, float]:
        response = ""
        rtt_ms = 0.0

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                t0 = time.time()
                sock.connect((self.target, port))

                if send_data:
                    sock.sendall(send_data)

                chunks: list[bytes] = []
                deadline = t0 + timeout

                while time.time() < deadline:
                    try:
                        sock.settimeout(min(0.1, max(0.01, deadline - time.time())))
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    except socket.timeout:
                        break

                rtt_ms = (time.time() - t0) * 1000.0
                response = b"".join(chunks).decode(errors="replace")

        except ConnectionRefusedError:
            response = "CONNECTION_REFUSED"
        except socket.timeout:
            response = "TIMEOUT"
        except OSError as e:
            response = f"ERROR:{e}"

        return response, round(rtt_ms, 2)

    # ------------------------------------------------------------------
    # Probe actions
    # ------------------------------------------------------------------

    def _probe_ssh_banner(self) -> ProbeOutcome:
        if not self._port_open("ssh"):
            return ProbeOutcome(
                "ssh_banner",
                0.12,
                1,
                0.01,
                "ssh closed unexpectedly",
            )

        resp, rtt = self._raw_connect(self.ports["ssh"], timeout=1.5)
        delta = 0.0

        if "SSH" not in resp:
            delta += 0.18
            self.suspicion_reasons.append("SSH service missing valid banner")

        if resp and rtt < 8.0:
            delta += 0.10
            self.suspicion_reasons.append(f"SSH banner too fast ({rtt}ms)")

        if "OpenSSH_for_Windows" in resp and "windows" not in self.active_profile_name.lower():
            delta += 0.08
            self.suspicion_reasons.append("SSH banner hints Windows while profile suggests something else")

        return ProbeOutcome("ssh_banner", delta, 1, rtt / 1000.0, f"ssh banner rtt={rtt}ms")

    def _probe_ssh_creds(self) -> ProbeOutcome:
        if not self._port_open("ssh"):
            return ProbeOutcome(
                "ssh_creds",
                0.08,
                1,
                0.01,
                "ssh creds skipped, port closed",
            )

        payload = b"SSH-2.0-Prober\r\nUSER:root PASS:root\r\n"
        resp, rtt = self._raw_connect(self.ports["ssh"], send_data=payload, timeout=1.0)
        delta = 0.0

        if not resp:
            delta += 0.08
            self.suspicion_reasons.append("SSH credential probe produced empty response")

        if "Permission denied" in resp or "Authentication" in resp:
            delta -= 0.02

        if "Access denied" in resp and "windows" not in self.active_profile_name.lower():
            delta += 0.05
            self.suspicion_reasons.append("SSH auth wording mismatched to claimed profile")

        return ProbeOutcome("ssh_creds", delta, 1, rtt / 1000.0, f"ssh creds rtt={rtt}ms")

    def _probe_http_root(self) -> ProbeOutcome:
        if not self._port_open("http"):
            return ProbeOutcome(
                "http_root",
                0.14,
                1,
                0.01,
                "http closed",
            )

        req = b"GET / HTTP/1.0\r\nHost: target\r\n\r\n"
        resp, rtt = self._raw_connect(self.ports["http"], send_data=req, timeout=1.5)
        delta = 0.0

        if "Server:" not in resp:
            delta += 0.12
            self.suspicion_reasons.append("HTTP root missing Server header")

        if "200 OK" not in resp and "403" not in resp and "404" not in resp:
            delta += 0.06
            self.suspicion_reasons.append("HTTP root returned malformed status")

        return ProbeOutcome("http_root", delta, 1, rtt / 1000.0, f"http root rtt={rtt}ms")

    def _probe_http_traversal(self) -> ProbeOutcome:
        if not self._port_open("http"):
            return ProbeOutcome(
                "http_traversal",
                0.10,
                1,
                0.01,
                "http traversal impossible, port closed",
            )

        req = b"GET /../../../etc/passwd HTTP/1.0\r\nHost: target\r\n\r\n"
        resp, rtt = self._raw_connect(self.ports["http"], send_data=req, timeout=1.5)
        delta = 0.0

        body = resp.lower()
        if "permission" not in body and "403" not in body and "404" not in body:
            delta += 0.14
            self.suspicion_reasons.append("Traversal handling inconsistent")

        return ProbeOutcome(
            "http_traversal",
            delta,
            1,
            rtt / 1000.0,
            f"http traversal rtt={rtt}ms",
        )

    def _probe_redis_ping(self) -> ProbeOutcome:
        if not self._port_open("redis"):
            return ProbeOutcome(
                "redis_ping",
                0.10,
                1,
                0.01,
                "redis closed",
            )

        resp, rtt = self._raw_connect(self.ports["redis"], send_data=b"PING\r\n", timeout=1.0)
        delta = 0.0

        if "PONG" not in resp and "-NOAUTH" not in resp and "+PONG" not in resp:
            delta += 0.10
            self.suspicion_reasons.append("Redis PING behavior inconsistent")

        return ProbeOutcome("redis_ping", delta, 1, rtt / 1000.0, f"redis ping rtt={rtt}ms")

    def _probe_redis_info(self) -> ProbeOutcome:
        if not self._port_open("redis"):
            return ProbeOutcome(
                "redis_info",
                0.08,
                1,
                0.01,
                "redis info impossible, port closed",
            )

        resp, rtt = self._raw_connect(self.ports["redis"], send_data=b"INFO\r\n", timeout=1.0)
        delta = 0.0

        if "redis_version" not in resp.lower():
            delta += 0.12
            self.suspicion_reasons.append("Redis INFO missing version fields")

        return ProbeOutcome("redis_info", delta, 1, rtt / 1000.0, f"redis info rtt={rtt}ms")

    def _probe_latency(self) -> ProbeOutcome:
        if not self._port_open("ssh"):
            return ProbeOutcome(
                "latency_probe",
                0.05,
                1,
                0.01,
                "latency probe degraded, ssh closed",
            )

        samples = []
        for _ in range(3):
            _, rtt = self._raw_connect(self.ports["ssh"], timeout=0.8)
            if rtt > 0:
                samples.append(rtt)

        delta = 0.0
        depth_gain = max(1, len(samples))
        duration_gain = sum(samples) / 1000.0 if samples else 0.01

        if len(samples) >= 2:
            spread = max(samples) - min(samples)
            threshold = 1.0 if self.fast_training else max(3.0, self._timing_tolerance * 10.0)
            if spread < threshold:
                delta += 0.10
                self.suspicion_reasons.append("Timing profile too deterministic")

        return ProbeOutcome(
            "latency_probe",
            delta,
            depth_gain,
            duration_gain,
            f"latency spread probe samples={samples}",
        )

    def _probe_consistency(self) -> ProbeOutcome:
        delta = 0.0
        depth_gain = 0
        duration_gain = 0.0

        ssh_resp = ""
        http_resp = ""

        if self._port_open("ssh"):
            ssh_resp, ssh_rtt = self._raw_connect(self.ports["ssh"], timeout=1.0)
            depth_gain += 1
            duration_gain += ssh_rtt / 1000.0

        if self._port_open("http"):
            http_resp, http_rtt = self._raw_connect(
                self.ports["http"],
                send_data=b"GET / HTTP/1.0\r\nHost: x\r\n\r\n",
                timeout=1.0,
            )
            depth_gain += 1
            duration_gain += http_rtt / 1000.0

        if ssh_resp and http_resp:
            if "Ubuntu" in ssh_resp and "Microsoft-IIS" in http_resp:
                delta += 0.20
                self.suspicion_reasons.append("Cross-service OS contradiction")
            elif "Debian" in ssh_resp and "Microsoft-IIS" in http_resp:
                delta += 0.20
                self.suspicion_reasons.append("Debian SSH with IIS HTTP contradiction")
            elif "Windows" in ssh_resp and "Apache/2.4.41 (Ubuntu)" in http_resp:
                delta += 0.20
                self.suspicion_reasons.append("Windows SSH with Ubuntu HTTP contradiction")

        elif not ssh_resp and not http_resp:
            delta += 0.08
            self.suspicion_reasons.append("Consistency probe unable to validate multiple services")

        return ProbeOutcome(
            "consistency_probe",
            delta,
            max(1, depth_gain),
            max(0.01, duration_gain),
            "cross-service consistency probe",
        )

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, action: str) -> ProbeOutcome:
        table: dict[str, Callable[[], ProbeOutcome]] = {
            "ssh_banner": self._probe_ssh_banner,
            "ssh_creds": self._probe_ssh_creds,
            "http_root": self._probe_http_root,
            "http_traversal": self._probe_http_traversal,
            "redis_ping": self._probe_redis_ping,
            "redis_info": self._probe_redis_info,
            "latency_probe": self._probe_latency,
            "consistency_probe": self._probe_consistency,
        }
        return table[action]()

    # ------------------------------------------------------------------
    # Main session loop
    # ------------------------------------------------------------------

    def run_session(self, verbose: bool = False) -> AttackerSessionResult:
        start = time.time()

        self.suspicion_score = 0.0
        self.suspicion_reasons = []
        self.decisions = []
        self.interaction_depth = 0
        self._duration = 0.0

        budget = 6 if self.fast_training else 10
        budget = max(4, int(round(budget * self._patience)))

        for step_idx in range(budget):
            action = self._pick_action()
            outcome = self._dispatch(action)
            self._record(action, outcome)

            reward = outcome.delta_suspicion + 0.02 * outcome.depth_gain
            self._update_q(action, reward)

            if verbose:
                print(
                    f"[Level3/{self.name}] step={step_idx+1} "
                    f"action={action} "
                    f"delta={outcome.delta_suspicion:+.3f} "
                    f"susp={self.suspicion_score:.3f} "
                    f"depth={self.interaction_depth}"
                )

            if self.suspicion_score >= self._detect_threshold:
                break

        flagged = self.suspicion_score >= self._detect_threshold
        session_duration = max(self._duration, time.time() - start)

        result = AttackerSessionResult(
            target=self.target,
            attacker_type=self.name,
            session_duration_seconds=round(session_duration, 4),
            suspicion_score=round(self.suspicion_score, 4),
            suspicion_reasons=self.suspicion_reasons[:],
            flagged_as_honeypot=flagged,
            interaction_depth=self.interaction_depth,
            decisions=self.decisions[:],
            metadata={
                "q_values": {k: round(v, 4) for k, v in self.q.items()},
                "counts": dict(self.n),
                "detect_threshold": self._detect_threshold,
                "active_profile_name": self.active_profile_name,
                "active_port_config": dict(self.active_port_config),
            },
        )

        if verbose:
            print(f"\n[Level3/{self.name}] Session Summary")
            print(f" Duration : {result.session_duration_seconds:.4f}s")
            print(f" Depth    : {result.interaction_depth}")
            print(f" Suspicion: {result.suspicion_score:.3f}")
            print(f" Flagged  : {result.flagged_as_honeypot}")
            if result.suspicion_reasons:
                print(" Reasons  :")
                for reason in result.suspicion_reasons:
                    print(f"  - {reason}")

        return result