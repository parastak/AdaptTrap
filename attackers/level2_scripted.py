# attackers/level2_scripted.py

from __future__ import annotations

import json
import os
import random
import socket
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

from attackers.base_attacker import BaseAttacker
from env.identity import PROFILES


@dataclass
class Level2Result:
    target: str
    session_start: str
    session_duration_seconds: float
    ssh_banner: str
    ssh_response_time_ms: float
    ssh_credential_attempts: list[dict]
    http_normal_response: dict
    http_traversal_response: dict
    redis_ping_response: str
    redis_info_snippet: str
    redis_auth_response: str
    suspicion_score: float
    suspicion_reasons: list[str]
    flagged_as_honeypot: bool
    interaction_depth: int
    attacker_type: str = "scripted_exploit"
    decisions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Level2ScriptedAttacker(BaseAttacker):
    """
    Level 2 attacker: stateful scripted exploitation bot.

    Stronger than Level 1 because it:
    - probes multiple services in sequence,
    - reasons about suspicious inconsistencies,
    - varies behavior by attacker profile,
    - can terminate with a honeypot verdict.
    """

    name = "scripted_exploit"

    CREDENTIALS = [
        ("root", "root"),
        ("admin", "admin"),
        ("admin", "password"),
        ("root", "toor"),
        ("user", "user"),
    ]

    HTTP_TRAVERSAL_PATHS = [
        "/../../../etc/passwd",
        "/admin",
        "/.env",
        "/config.php",
    ]

    _DEFAULT_PROFILE = {
        "name": "scripted_exploit",
        "detect_threshold": 0.40,
        "ssh_aggression": 0.80,
        "patience": 0.90,
        "timing_tolerance": 0.20,
    }

    def __init__(
        self,
        target: str = "127.0.0.1",
        ports: dict | None = None,
        fast_training: bool = False,
        simulated_latency_ms: float = 50.0,
        active_profile_name: str = "ubuntu_web",
        active_port_config: dict | None = None,
        attacker_profile: dict | None = None,
    ) -> None:
        super().__init__(
            target=target,
            ports=ports or {"ssh": 2222, "http": 8080, "redis": 6379},
            fast_training=fast_training,
            simulated_latency_ms=simulated_latency_ms,
            active_profile_name=active_profile_name,
            active_port_config=active_port_config or {"ssh": True, "http": True, "redis": True},
            attacker_profile=attacker_profile or self._DEFAULT_PROFILE,
        )

        self.suspicion_score: float = 0.0
        self.suspicion_reasons: list[str] = []
        self.decisions: list[str] = []
        self.interaction_depth: int = 0

        self._profile = self.attacker_profile
        self._detect_threshold = float(self._profile["detect_threshold"])
        self._ssh_aggression = float(self._profile["ssh_aggression"])
        self._patience = float(self._profile["patience"])
        self._timing_tolerance = float(self._profile["timing_tolerance"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _port_open(self, service: str) -> bool:
        return bool(self.active_port_config.get(service, True))

    def _raw_connect(
        self,
        port: int,
        send_data: bytes | None = None,
        timeout: float = 3.0,
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

                try:
                    while time.time() < deadline:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            break
                        sock.settimeout(min(remaining, 0.1))
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        chunks.append(chunk)
                except (socket.timeout, ConnectionResetError):
                    pass

                rtt_ms = (time.time() - t0) * 1000.0
                response = b"".join(chunks).decode(errors="replace")

        except ConnectionRefusedError:
            response = "CONNECTION_REFUSED"
        except socket.timeout:
            response = "TIMEOUT"
        except OSError as e:
            response = f"ERROR:{e}"

        return response, round(rtt_ms, 2)

    @staticmethod
    def _extract_status(response: str) -> int:
        try:
            return int(response.split(" ")[1])
        except (IndexError, ValueError):
            return 0

    @staticmethod
    def _extract_redis_value(info: str, key: str) -> str | None:
        for line in info.splitlines():
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip()
        return None

    def _infer_profile_from_banner(self, banner: str) -> str:
        if "Ubuntu" in banner:
            return "ubuntu_web"
        if "Debian" in banner:
            return "debian_db"
        if "Windows" in banner:
            return "windows_iis"
        return "centos_api"

    def _check_temporal_coherence(self, actual_rtt_ms: float, profile_name: str) -> None:
        identity = next((p for p in PROFILES if p.name == profile_name), None)
        if identity is None:
            return

        lo = identity.expected_latency_min
        hi = identity.expected_latency_max

        if self.fast_training:
            lo *= 0.08
            hi *= 0.08

        if actual_rtt_ms < lo * 0.5:
            self.suspicion_score = min(1.0, self.suspicion_score + 0.20)
            self.suspicion_reasons.append(
                f"RTT {actual_rtt_ms:.1f}ms too fast for claimed {profile_name} "
                f"(expected {lo:.1f}-{hi:.1f}ms)"
            )
        elif actual_rtt_ms > hi * 2.0:
            self.suspicion_score = min(1.0, self.suspicion_score + 0.15)
            self.suspicion_reasons.append(
                f"RTT {actual_rtt_ms:.1f}ms too slow for claimed {profile_name} "
                f"(expected {lo:.1f}-{hi:.1f}ms)"
            )

    def _analyze_timing(self, ssh_rtt: float, http_rtt: float, redis_rtt: float) -> None:
        times = [ssh_rtt, http_rtt, redis_rtt]
        valid = [t for t in times if 0 < t < 5000]

        if len(valid) < 2:
            return

        spread = max(valid) - min(valid)
        threshold = self._timing_tolerance if self.fast_training else self._timing_tolerance * 10.0

        if spread < threshold:
            self.suspicion_score = min(1.0, self.suspicion_score + 0.20)
            self.suspicion_reasons.append(
                f"All services responded within {spread:.1f}ms of each other - "
                "suggests single-process fake service handler"
            )
            self.decisions.append("Timing: suspiciously uniform")
        else:
            self.decisions.append(f"Timing: natural spread {spread:.1f}ms")

    # ------------------------------------------------------------------
    # Probes
    # ------------------------------------------------------------------

    def _probe_ssh(self) -> dict:
        self.decisions.append("Probing SSH")

        if not self._port_open("ssh"):
            self.suspicion_score = min(1.0, self.suspicion_score + 0.20)
            self.suspicion_reasons.append("SSH port closed")
            return {
                "banner": "",
                "rtt_ms": 0.0,
                "credential_attempts": [],
            }

        banner, rtt_ms = self._raw_connect(self.ports["ssh"], timeout=2.0)
        self.interaction_depth += 1

        result = {
            "banner": banner.strip(),
            "rtt_ms": rtt_ms,
            "credential_attempts": [],
        }

        if rtt_ms < 15.0 and banner and "SSH" in banner:
            self.suspicion_score = min(1.0, self.suspicion_score + 0.15)
            self.suspicion_reasons.append(
                f"SSH banner returned in {rtt_ms}ms - unusually fast"
            )

        if "SSH" not in banner:
            self.decisions.append("SSH: no valid banner, skipping credentials")
            return result

        self.decisions.append(f"SSH: got banner '{banner.strip()}'")

        if random.random() > self._ssh_aggression:
            self.decisions.append(
                f"SSH: skipping credential storm (aggression={self._ssh_aggression:.1f})"
            )
            return result

        cred_timeout = 0.30 if self.fast_training else 3.0

        for username, password in self.CREDENTIALS:
            fake_auth = f"SSH-2.0-FakeClient\r\nUSER:{username} PASS:{password}\r\n"
            response, _ = self._raw_connect(
                self.ports["ssh"],
                send_data=fake_auth.encode(),
                timeout=cred_timeout,
            )
            self.interaction_depth += 1
            result["credential_attempts"].append(
                {
                    "username": username,
                    "password": password,
                    "response": response.strip()[:100],
                }
            )

        responses = [a["response"] for a in result["credential_attempts"] if a["response"]]
        if responses and len(set(responses)) == 1:
            self.suspicion_score = min(1.0, self.suspicion_score + 0.25)
            self.suspicion_reasons.append(
                "All SSH credential attempts returned identical response - scripted handler suspected"
            )
            self.decisions.append("SSH: identical auth responses")
        else:
            self.decisions.append("SSH: varied auth responses")

        return result

    def _probe_http(self) -> tuple[dict, dict]:
        self.decisions.append("Probing HTTP")

        normal_result = {
            "path": "/",
            "response_snippet": "",
            "rtt_ms": 0.0,
            "status_code": 0,
        }
        traversal_result = {
            "probes": [],
            "unique_statuses": [],
        }

        if not self._port_open("http"):
            self.suspicion_score = min(1.0, self.suspicion_score + 0.20)
            self.suspicion_reasons.append("HTTP port closed")
            self.decisions.append("HTTP: closed")
            return normal_result, traversal_result

        normal_req = b"GET / HTTP/1.0\r\nHost: target\r\n\r\n"
        normal_resp, normal_rtt = self._raw_connect(
            self.ports["http"],
            send_data=normal_req,
            timeout=2.0,
        )
        self.interaction_depth += 1

        normal_result = {
            "path": "/",
            "response_snippet": normal_resp[:300],
            "rtt_ms": normal_rtt,
            "status_code": self._extract_status(normal_resp),
        }

        if "Running" in normal_resp and "\n\n" in normal_resp:
            self.suspicion_score = min(1.0, self.suspicion_score + 0.15)
            self.suspicion_reasons.append(
                "HTTP response contains generic 'Running' body - possible honeypot template"
            )

        n_paths = max(1, int(len(self.HTTP_TRAVERSAL_PATHS) * self._patience))
        paths_to_probe = self.HTTP_TRAVERSAL_PATHS[:n_paths]

        traversal_results: list[dict] = []
        for path in paths_to_probe:
            req = f"GET {path} HTTP/1.0\r\nHost: target\r\n\r\n".encode()
            resp, rtt = self._raw_connect(self.ports["http"], send_data=req, timeout=2.0)
            self.interaction_depth += 1
            traversal_results.append(
                {
                    "path": path,
                    "response": resp[:200],
                    "rtt_ms": rtt,
                    "status": self._extract_status(resp),
                }
            )

        traversal_statuses = [t["status"] for t in traversal_results]
        if (
            traversal_statuses
            and len(set(traversal_statuses)) == 1
            and traversal_statuses[0] == normal_result["status_code"]
        ):
            self.suspicion_score = min(1.0, self.suspicion_score + 0.10)
            self.suspicion_reasons.append(
                "All HTTP paths return same status - no real routing logic detected"
            )

        traversal_result = {
            "probes": traversal_results,
            "unique_statuses": list(set(traversal_statuses)),
        }

        return normal_result, traversal_result

    def _probe_redis(self) -> tuple[str, str, str, float]:
        self.decisions.append("Probing Redis")

        if not self._port_open("redis"):
            self.suspicion_score = min(1.0, self.suspicion_score + 0.15)
            self.suspicion_reasons.append("Redis port closed")
            self.decisions.append("Redis: closed")
            return "", "", "", 0.0

        ping_resp, ping_rtt = self._raw_connect(
            self.ports["redis"],
            send_data=b"PING\r\n",
            timeout=2.0,
        )
        self.interaction_depth += 1

        info_resp, _ = self._raw_connect(
            self.ports["redis"],
            send_data=b"INFO server\r\n",
            timeout=2.0,
        )
        self.interaction_depth += 1

        auth_resp, _ = self._raw_connect(
            self.ports["redis"],
            send_data=b"AUTH wrongpassword\r\n",
            timeout=2.0,
        )
        self.interaction_depth += 1

        if "+PONG" in ping_resp:
            self.suspicion_score = min(1.0, self.suspicion_score + 0.10)
            self.suspicion_reasons.append(
                "Redis PING accepted without AUTH - suspicious for protected Redis"
            )

        return ping_resp.strip(), info_resp[:300], auth_resp.strip(), ping_rtt

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    def run_session(self, verbose: bool = True) -> Level2Result:
        session_start = datetime.now().isoformat()
        t0 = time.time()

        self.suspicion_score = 0.0
        self.suspicion_reasons = []
        self.decisions = []
        self.interaction_depth = 0

        ssh_result = self._probe_ssh()

        detected_profile = self._infer_profile_from_banner(ssh_result["banner"])
        self._check_temporal_coherence(self.simulated_latency_ms, detected_profile)

        http_normal, http_trav = self._probe_http()
        redis_ping, redis_info, redis_auth, redis_rtt = self._probe_redis()

        duration = time.time() - t0

        valid_rtts = [ssh_result["rtt_ms"]]
        if self._port_open("http"):
            valid_rtts.append(http_normal["rtt_ms"])
        if self._port_open("redis"):
            valid_rtts.append(redis_rtt)

        if len(valid_rtts) >= 2:
            self._analyze_timing(
                ssh_rtt=valid_rtts[0],
                http_rtt=valid_rtts[1] if len(valid_rtts) > 1 else 0.0,
                redis_rtt=valid_rtts[2] if len(valid_rtts) > 2 else 0.0,
            )

        flagged = self.suspicion_score >= self._detect_threshold

        result = Level2Result(
            target=self.target,
            session_start=session_start,
            session_duration_seconds=round(duration, 4),
            ssh_banner=ssh_result["banner"],
            ssh_response_time_ms=ssh_result["rtt_ms"],
            ssh_credential_attempts=ssh_result["credential_attempts"],
            http_normal_response=http_normal,
            http_traversal_response=http_trav,
            redis_ping_response=redis_ping,
            redis_info_snippet=redis_info,
            redis_auth_response=redis_auth,
            suspicion_score=round(self.suspicion_score, 4),
            suspicion_reasons=self.suspicion_reasons[:],
            flagged_as_honeypot=flagged,
            interaction_depth=self.interaction_depth,
            attacker_type=self._profile["name"],
            decisions=self.decisions[:],
        )

        if verbose:
            self._print_result(result)

        return result

    def _print_result(self, result: Level2Result) -> None:
        print(f"\n[Level2/{result.attacker_type}] Session Summary")
        print(f" Duration         : {result.session_duration_seconds:.4f}s")
        print(f" Interaction depth: {result.interaction_depth}")
        print(f" Suspicion score  : {result.suspicion_score:.3f} / 1.0")
        print(f" Detect threshold : {self._detect_threshold:.2f}")
        print(
            f" Verdict          : "
            f"{'HONEYPOT SUSPECTED' if result.flagged_as_honeypot else 'Looks real'}"
        )

        if result.suspicion_reasons:
            print("\n[Level2] Suspicion Reasons")
            for i, reason in enumerate(result.suspicion_reasons, 1):
                print(f" {i}. {reason}")

        if result.decisions:
            print("\n[Level2] Decisions")
            for d in result.decisions:
                print(f" -> {d}")

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def measure_baseline(
        self,
        runs: int = 3,
        log_path: str = "logs/baseline_level2.json",
    ) -> dict:
        os.makedirs("logs", exist_ok=True)

        results: list[dict] = []

        print(f"\n[Level2] Running {runs} baseline sessions...\n")
        for i in range(runs):
            print(f"--- Session {i + 1}/{runs} ---")
            result = self.run_session(verbose=True)
            results.append(result.to_dict())

        durations = [r["session_duration_seconds"] for r in results]
        suspicion_scores = [r["suspicion_score"] for r in results]
        depths = [r["interaction_depth"] for r in results]
        flagged_count = sum(1 for r in results if r["flagged_as_honeypot"])

        baseline = {
            "timestamp": datetime.now().isoformat(),
            "policy": "static_honeypot_no_adaptation",
            "attacker_level": 2,
            "attacker_type": self._profile["name"],
            "runs": runs,
            "mean_session_duration_seconds": round(sum(durations) / len(durations), 4),
            "mean_suspicion_score": round(sum(suspicion_scores) / len(suspicion_scores), 4),
            "mean_interaction_depth": round(sum(depths) / len(depths), 2),
            "honeypot_flag_rate": f"{flagged_count}/{runs}",
            "raw_results": results,
        }

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)

        print(f"\n[Level2] Baseline saved -> {log_path}")
        print(f"[Level2] Mean duration       : {baseline['mean_session_duration_seconds']}s")
        print(f"[Level2] Mean suspicion     : {baseline['mean_suspicion_score']}")
        print(f"[Level2] Mean depth         : {baseline['mean_interaction_depth']}")
        print(f"[Level2] Honeypot flagged   : {baseline['honeypot_flag_rate']}")
        print("\n[!] This is your LEVEL 2 STATIC BASELINE.")

        return baseline