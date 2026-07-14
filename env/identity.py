# env/identity.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemIdentity:
    """
    Single source of truth for all fake system behavior.

    Every service should derive its externally visible behavior from this object.
    One cross-service mismatch is enough to increase attacker suspicion.
    """

    name: str
    os_name: str
    kernel: str
    hostname: str
    ssh_banner: bytes
    http_server_header: str
    http_os_header: str
    redis_version: str
    redis_os: str
    uptime_seconds: int
    filesystem_hint: str
    expected_latency_min: float
    expected_latency_max: float


PROFILES: list[SystemIdentity] = [
    # Profile 0 — Ubuntu web server
    SystemIdentity(
        name="ubuntu_web",
        os_name="Ubuntu 20.04.6 LTS",
        kernel="5.15.0-91-generic",
        hostname="web-prod-01",
        ssh_banner=b"SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.11\r\n",
        http_server_header="Apache/2.4.41 (Ubuntu)",
        http_os_header="Ubuntu",
        redis_version="6.0.16",
        redis_os="Linux 5.15.0-91-generic x86_64",
        uptime_seconds=1_245_678,
        filesystem_hint="/var/www/html",
        expected_latency_min=10.0,
        expected_latency_max=80.0,
    ),
    # Profile 1 — Debian database server
    SystemIdentity(
        name="debian_db",
        os_name="Debian GNU/Linux 11 (bullseye)",
        kernel="5.10.0-28-amd64",
        hostname="db-internal-02",
        ssh_banner=b"SSH-2.0-OpenSSH_8.4p1 Debian-5+deb11u3\r\n",
        http_server_header="nginx/1.18.0",
        http_os_header="Debian",
        redis_version="5.0.14",
        redis_os="Linux 5.10.0-28-amd64 x86_64",
        uptime_seconds=3_891_045,
        filesystem_hint="/srv/db",
        expected_latency_min=60.0,
        expected_latency_max=250.0,
    ),
    # Profile 2 — CentOS API/app server
    SystemIdentity(
        name="centos_api",
        os_name="CentOS Linux 7 (Core)",
        kernel="3.10.0-1160.108.1.el7.x86_64",
        hostname="api-centos-07",
        ssh_banner=b"SSH-2.0-OpenSSH_7.4p1\r\n",
        http_server_header="Apache/2.4.6 (CentOS)",
        http_os_header="CentOS",
        redis_version="3.2.12",
        redis_os="Linux 3.10.0-1160.108.1.el7.x86_64 x86_64",
        uptime_seconds=8_734_512,
        filesystem_hint="/srv/api",
        expected_latency_min=150.0,
        expected_latency_max=400.0,
    ),
    # Profile 3 — Windows IIS server
    SystemIdentity(
        name="windows_iis",
        os_name="Windows Server 2019",
        kernel="10.0.17763",
        hostname="WIN-APPSERV-03",
        ssh_banner=b"SSH-2.0-OpenSSH_for_Windows_8.1\r\n",
        http_server_header="Microsoft-IIS/10.0",
        http_os_header="Windows",
        redis_version="5.0.14",
        redis_os="Windows Version 10.0.17763 (x86_64)",
        uptime_seconds=432_000,
        filesystem_hint=r"C:\inetpub\wwwroot",
        expected_latency_min=30.0,
        expected_latency_max=120.0,
    ),
]


PROFILE_INDEX = {profile.name: i for i, profile in enumerate(PROFILES)}


def get_profile(index: int) -> SystemIdentity:
    return PROFILES[index % len(PROFILES)]


def get_profile_by_name(name: str) -> SystemIdentity:
    if name not in PROFILE_INDEX:
        raise KeyError(f"Unknown profile name: {name}")
    return PROFILES[PROFILE_INDEX[name]]