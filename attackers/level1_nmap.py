# attackers/level1_nmap.py

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass
class PortResult:
    port: int
    protocol: str
    state: str
    service: str
    banner: str = ""
    version: str = ""


@dataclass
class NmapScanResult:
    target: str
    scan_start: str
    scan_duration_seconds: float
    ports_found: list[PortResult] = field(default_factory=list)
    raw_output: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class Level1Scanner:
    """
    Level 1 Attacker: dumb automated scanner.

    Runs nmap service/version detection plus banner grabbing.
    Zero memory, zero adaptation, same behavior every run.
    This is the static baseline measurement.
    """

    def __init__(
        self,
        target: str = "127.0.0.1",
        ports: list[int] | None = None,
        timeout: int = 60,
    ) -> None:
        self.target = target
        self.ports = ports or [2222, 8080, 6379]
        self.port_str = ",".join(str(p) for p in self.ports)
        self.timeout = timeout

    def scan(self, verbose: bool = True) -> NmapScanResult:
        """
        Runs:
            nmap -sV -p <ports> --script=banner --open -T4 <target>

        Returns a structured NmapScanResult.
        """
        command = [
            "nmap",
            "-sV",
            "-p",
            self.port_str,
            "--script=banner",
            "--open",
            "-T4",
            self.target,
        ]

        if verbose:
            print(f"[Level1] Running: {' '.join(command)}")

        start_time = time.time()
        scan_start = datetime.now().isoformat()

        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            raw = (proc.stdout or "") + (proc.stderr or "")
        except subprocess.TimeoutExpired:
            raw = "SCAN TIMEOUT"
        except FileNotFoundError:
            raw = "ERROR: nmap not found. Install nmap and ensure it is in PATH."

        duration = time.time() - start_time

        result = NmapScanResult(
            target=self.target,
            scan_start=scan_start,
            scan_duration_seconds=round(duration, 3),
            raw_output=raw,
        )
        result.ports_found = self._parse_output(raw)

        if verbose:
            self._print_result(result)

        return result

    def _parse_output(self, raw: str) -> list[PortResult]:
        """
        Parses nmap text output into structured PortResult objects.

        Example port line:
            2222/tcp open  ssh  OpenSSH 8.2p1 Ubuntu 4ubuntu0.11
        """
        ports: list[PortResult] = []

        port_pattern = re.compile(
            r"^(\d+)/(\w+)\s+(open|closed|filtered)\s+(\S+)?\s*(.*)$"
        )
        banner_pattern = re.compile(r"\|_?banner:\s*(.+)", re.IGNORECASE)

        current_port: int | None = None
        banner_map: dict[int, str] = {}

        for line in raw.splitlines():
            stripped = line.strip()

            port_match = port_pattern.match(stripped)
            if port_match:
                current_port = int(port_match.group(1))
                protocol = port_match.group(2)
                state = port_match.group(3)
                service = port_match.group(4) or "unknown"
                version = (port_match.group(5) or "").strip()

                ports.append(
                    PortResult(
                        port=current_port,
                        protocol=protocol,
                        state=state,
                        service=service,
                        banner="",
                        version=version,
                    )
                )
                continue

            banner_match = banner_pattern.search(stripped)
            if banner_match and current_port is not None:
                banner_map[current_port] = banner_match.group(1).strip()

        for p in ports:
            if p.port in banner_map:
                p.banner = banner_map[p.port]

        return ports

    def _print_result(self, result: NmapScanResult) -> None:
        print(f"\n[Level1] Scan complete in {result.scan_duration_seconds:.2f}s")
        print(f"[Level1] Target: {result.target}")
        print(f"[Level1] Ports found: {len(result.ports_found)}\n")

        if not result.ports_found:
            print("[Level1] No open ports detected.")
            print("[Level1] Raw output:")
            print(result.raw_output)
            return

        for p in result.ports_found:
            version_info = f" | Version: {p.version}" if p.version else ""
            banner_info = f" | Banner: {p.banner}" if p.banner else ""
            print(
                f" {p.port}/{p.protocol:<4} "
                f"[{p.state:<8}] "
                f"{p.service:<12}"
                f"{version_info}"
                f"{banner_info}"
            )

    def measure_baseline(
        self,
        runs: int = 3,
        log_path: str = "logs/baseline_level1.json",
    ) -> dict:
        """
        Runs the scanner multiple times and records the static baseline.
        This is the baseline the adaptive defender will later be compared against.
        """
        os.makedirs("logs", exist_ok=True)

        results: list[dict] = []

        print(f"\n[Level1] Running {runs} baseline scans...\n")
        for i in range(runs):
            print(f"--- Baseline Run {i + 1}/{runs} ---")
            result = self.scan(verbose=True)
            results.append(result.to_dict())

        durations = [r["scan_duration_seconds"] for r in results]
        ports_found = [len(r["ports_found"]) for r in results]

        baseline = {
            "timestamp": datetime.now().isoformat(),
            "policy": "static_honeypot_no_adaptation",
            "attacker_level": 1,
            "attacker_type": "recon_probe",
            "runs": runs,
            "mean_scan_duration_seconds": round(sum(durations) / len(durations), 3),
            "mean_ports_found": round(sum(ports_found) / len(ports_found), 2),
            "raw_results": results,
        }

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)

        print(f"\n[Level1] Baseline saved to: {log_path}")
        print(f"[Level1] Mean scan duration: {baseline['mean_scan_duration_seconds']}s")
        print(f"[Level1] Mean ports found: {baseline['mean_ports_found']}")
        print("\n[!] This is your LEVEL 1 STATIC BASELINE.")
        print("[!] Your trained RL defender will later be compared against this.")

        return baseline