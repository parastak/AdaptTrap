# env/service_simulator.py

from __future__ import annotations

import asyncio
import random
import time

from env.identity import SystemIdentity, get_profile


class ServiceSimulator:
    """
    Runs three fake async TCP services: SSH, HTTP, Redis.

    All service responses are derived from the active SystemIdentity profile.
    The defender changes the active identity via set_profile() and timing via
    set_latency().
    """

    def __init__(self, profile_index: int = 0, fast_training: bool = False) -> None:
        self.identity: SystemIdentity = get_profile(profile_index)
        self.latency_ms: float = 50.0
        self.fast_training: bool = fast_training
        self.session_count: int = 0
        self.interaction_depth: int = 0
        self._start_time: float = time.time()

    def set_profile(self, index: int) -> None:
        self.identity = get_profile(index)

    def set_latency(self, ms: float) -> None:
        self.latency_ms = max(10.0, ms)

    async def _apply_latency(self) -> float:
        """
        Apply configured latency and return actual sleep time in milliseconds.

        In fast_training mode, preserve relative timing differences while keeping
        absolute delay small enough for RL throughput.
        """
        base = self.latency_ms / 1000.0
        jitter = base * 0.08 * (random.random() * 2 - 1)

        if self.fast_training:
            scaled = max(0.004, base * 0.18 + jitter * 0.18)
            await asyncio.sleep(scaled)
            return scaled * 1000.0

        actual = max(0.005, base + jitter)
        await asyncio.sleep(actual)
        return actual * 1000.0

    # ------------------------------------------------------------------
    # SSH
    # ------------------------------------------------------------------

    async def handle_ssh(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.session_count += 1

        try:
            await self._apply_latency()
            writer.write(self.identity.ssh_banner)
            await writer.drain()

            try:
                data = await asyncio.wait_for(reader.read(256), timeout=10.0)
                if data:
                    self.interaction_depth += 1

                    profile_rejections = {
                        "ubuntu_web": [
                            b"Permission denied, please try again.\r\n",
                            b"Permission denied (publickey,password).\r\n",
                            b"Authentication failed.\r\n",
                            b"Too many authentication failures.\r\n",
                        ],
                        "debian_db": [
                            b"Authentication failure.\r\n",
                            b"Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password).\r\n",
                            b"Permission denied, please try again.\r\n",
                        ],
                        "centos_api": [
                            b"Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password).\r\n",
                            b"Authentication failed.\r\n",
                            b"Permission denied, please try again.\r\n",
                            b"Too many authentication failures.\r\n",
                        ],
                        "centos_legacy": [
                            b"Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password).\r\n",
                            b"Authentication failed.\r\n",
                            b"Permission denied, please try again.\r\n",
                            b"Too many authentication failures.\r\n",
                        ],
                        "windows_iis": [
                            b"Permission denied.\r\n",
                            b"Authentication failure.\r\n",
                            b"Access denied.\r\n",
                            b"Logon failure: unknown user name or bad password.\r\n",
                        ],
                    }

                    options = profile_rejections.get(
                        self.identity.name,
                        [b"Permission denied.\r\n"],
                    )

                    if self.fast_training:
                        await asyncio.sleep(0.01)
                    else:
                        await asyncio.sleep(random.uniform(0.8, 1.8))

                    choice_idx = (self.session_count + self.interaction_depth) % len(options)
                    writer.write(options[choice_idx])
                    await writer.drain()

            except asyncio.TimeoutError:
                pass

        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    async def handle_http(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.session_count += 1

        try:
            await self._apply_latency()

            try:
                raw = await asyncio.wait_for(reader.read(2048), timeout=10.0)
                request = raw.decode(errors="replace")
            except asyncio.TimeoutError:
                writer.close()
                return

            path = "/"
            if request.startswith("GET "):
                parts = request.split(" ")
                if len(parts) >= 2:
                    path = parts[1]

            response = self._build_http_response(path)
            self.interaction_depth += 1

            writer.write(response.encode())
            await writer.drain()

        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    def _build_http_response(self, path: str) -> str:
        server = self.identity.http_server_header
        os_hdr = self.identity.http_os_header

        traversal_indicators = ["../", "%2e%2e", "etc/passwd", "etc/shadow"]
        blocked_paths = [
            "/admin",
            "/administrator",
            "/.env",
            "/config.php",
            "/wp-admin",
            "/phpmyadmin",
            "/.git",
        ]

        lower_path = path.lower()
        is_traversal = any(t in lower_path for t in traversal_indicators)
        is_blocked = any(lower_path.startswith(b) for b in blocked_paths)

        profile_bodies = {
            "ubuntu_web": (
                "<html><body>"
                "<h1>Apache2 Ubuntu Default Page</h1>"
                "<p>It works!</p>"
                f"<p>Server: {server}</p>"
                "</body></html>"
            ),
            "debian_db": (
                "<html><body>"
                "<h1>Welcome to nginx!</h1>"
                "<p>If you see this page, the nginx web server is running.</p>"
                f"<p>Server: {server}</p>"
                "</body></html>"
            ),
            "centos_api": (
                "<html><body>"
                "<h1>Testing 123..</h1>"
                "<p>This page is used to test the proper operation of the Apache HTTP server.</p>"
                f"<p>Server: {server}</p>"
                "</body></html>"
            ),
            "centos_legacy": (
                "<html><body>"
                "<h1>Testing 123..</h1>"
                "<p>This page is used to test the proper operation of the Apache HTTP server.</p>"
                f"<p>Server: {server}</p>"
                "</body></html>"
            ),
            "windows_iis": (
                "<html><body>"
                "<h1>IIS Windows Server</h1>"
                "<p>Welcome to IIS!</p>"
                f"<p>Server: {server}</p>"
                "</body></html>"
            ),
        }

        if is_traversal or is_blocked:
            body = (
                "<html><head><title>403 Forbidden</title></head><body>"
                "<h1>Forbidden</h1>"
                f"<p>You don't have permission to access {path}.</p>"
                f"<hr/><address>{server}</address>"
                "</body></html>"
            )
            status = "403 Forbidden"
        else:
            body = profile_bodies.get(
                self.identity.name,
                "<html><body><h1>Service Running</h1></body></html>",
            )
            status = "200 OK"

        return (
            f"HTTP/1.1 {status}\r\n"
            f"Server: {server}\r\n"
            f"X-Powered-By: {os_hdr}\r\n"
            f"Content-Type: text/html\r\n"
            f"Content-Length: {len(body.encode())}\r\n"
            f"Connection: close\r\n\r\n"
            f"{body}"
        )

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------

    async def handle_redis(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.session_count += 1

        try:
            await self._apply_latency()

            uptime_drift = self.identity.uptime_seconds + int(time.time() - self._start_time)
            uptime_days = uptime_drift // 86400

            tcp_port = 6380 if self.identity.name == "windows_iis" else 6379
            executable = (
                r"C:\Program Files\Redis\redis-server.exe"
                if self.identity.name == "windows_iis"
                else "/usr/bin/redis-server"
            )

            info_response = (
                "# Server\r\n"
                f"redis_version:{self.identity.redis_version}\r\n"
                f"os:{self.identity.redis_os}\r\n"
                f"uptime_in_seconds:{uptime_drift}\r\n"
                f"uptime_in_days:{uptime_days}\r\n"
                f"tcp_port:{tcp_port}\r\n"
                f"executable:{executable}\r\n"
            )

            try:
                raw = await asyncio.wait_for(reader.read(256), timeout=10.0)
                command = raw.decode(errors="replace").strip().upper()
                self.interaction_depth += 1

                if "AUTH" in command:
                    writer.write(
                        b"-WRONGPASS invalid username-password pair or user is disabled.\r\n"
                    )
                elif "PING" in command:
                    writer.write(b"+PONG\r\n")
                elif "INFO" in command:
                    encoded = f"${len(info_response)}\r\n{info_response}\r\n"
                    writer.write(encoded.encode())
                else:
                    writer.write(
                        b"-ERR This instance has access control list enabled. AUTH required.\r\n"
                    )

                await writer.drain()

            except asyncio.TimeoutError:
                pass

        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_state_snapshot(self) -> dict:
        return {
            "profile": self.identity.name,
            "session_count": self.session_count,
            "interaction_depth": self.interaction_depth,
            "latency_ms": self.latency_ms,
        }


async def run_all_services(host: str = "0.0.0.0") -> None:
    sim = ServiceSimulator(profile_index=0)

    ssh_server = await asyncio.start_server(sim.handle_ssh, host, 2222)
    http_server = await asyncio.start_server(sim.handle_http, host, 8080)
    redis_server = await asyncio.start_server(sim.handle_redis, host, 6379)

    print(f"[AdaptTrap] SSH   -> {host}:2222")
    print(f"[AdaptTrap] HTTP  -> {host}:8080")
    print(f"[AdaptTrap] Redis -> {host}:6379")
    print("[AdaptTrap] All services running. Ctrl+C to stop.\n")

    try:
        async with ssh_server, http_server, redis_server:
            await asyncio.gather(
                ssh_server.serve_forever(),
                http_server.serve_forever(),
                redis_server.serve_forever(),
            )
    except asyncio.CancelledError:
        pass
    