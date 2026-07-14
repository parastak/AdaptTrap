# env/honeypot_env.py

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from attackers.level2_scripted import Level2ScriptedAttacker
from attackers.level3_bandit import Level3BanditAttacker
from defender.actor_critic import DETECTION_PENALTY, EVASION_BONUS
from env.service_simulator import ServiceSimulator


ATTACKER_PROFILES = [
    {
        "name": "recon_probe",
        "detect_threshold": 0.50,
        "ssh_aggression": 0.30,
        "patience": 0.60,
        "timing_tolerance": 0.40,
    },
    {
        "name": "scripted_exploit",
        "detect_threshold": 0.40,
        "ssh_aggression": 0.80,
        "patience": 0.90,
        "timing_tolerance": 0.20,
    },
    {
        "name": "ai_probe",
        "detect_threshold": 0.25,
        "ssh_aggression": 0.50,
        "patience": 1.00,
        "timing_tolerance": 0.10,
    },
]

N_ATTACKER_TYPES = len(ATTACKER_PROFILES)

ATTACKER_NAME_TO_INDEX = {
    profile["name"]: idx for idx, profile in enumerate(ATTACKER_PROFILES)
}

ATTACKER_FACTORY = {
    "recon_probe": Level2ScriptedAttacker,
    "scripted_exploit": Level2ScriptedAttacker,
    "ai_probe": Level3BanditAttacker,
}


class HoneypotEnv(gym.Env):
    """
    AdaptTrap RL environment.

    The defender controls:
    - identity profile
    - temporal response profile
    - exposed port configuration

    Each step launches one attacker session against the live local services.
    """

    metadata = {"render_modes": []}

    N_PROFILES = 4
    N_TEMPORAL = 4
    N_PORT_CFGS = 3

    MAX_SESSIONS = 20
    MAX_DEPTH = 14.0

    BASELINE_DUR_FAST = 0.22
    BASELINE_DUR_REAL = 13.282

    TEMPORAL_LATENCY = {
        0: 50.0,   # normal
        1: 200.0,  # slow_server
        2: 20.0,   # fast_cdn
        3: 80.0,   # load_sim
    }

    PORT_CONFIGS = [
        {"ssh": True, "http": True, "redis": True},
        {"ssh": True, "http": True, "redis": False},
        {"ssh": True, "http": False, "redis": False},
    ]

    TRAIN_PORTS = {"ssh": 12222, "http": 18080, "redis": 16379}
    EVAL_PORTS = {"ssh": 2222, "http": 8080, "redis": 6379}

    SERVICE_STARTUP_TIMEOUT_S = 3.0
    SERVICE_JOIN_TIMEOUT_S = 5.0

    def __init__(
        self,
        target: str = "127.0.0.1",
        fast_training: bool = True,
    ):
        super().__init__()

        self.target = target
        self.fast_training = fast_training
        self.ports = dict(self.TRAIN_PORTS if fast_training else self.EVAL_PORTS)
        self._baseline_dur = (
            self.BASELINE_DUR_FAST if fast_training else self.BASELINE_DUR_REAL
        )

        self.action_space = spaces.MultiDiscrete(
            [self.N_PROFILES, self.N_TEMPORAL, self.N_PORT_CFGS]
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )

        self.simulator = ServiceSimulator(
            profile_index=0,
            fast_training=fast_training,
        )

        self._profile_idx = 0
        self._temporal_idx = 0
        self._port_cfg_idx = 0

        self._last_suspicion = 0.0
        self._last_depth = 0.0
        self._last_duration = self._baseline_dur

        self._session_count = 0
        self._consec_detects = 0
        self._episode_reward = 0.0

        self._reward_histories = {
            "recon_probe": deque(maxlen=200),
            "scripted_exploit": deque(maxlen=200),
            "ai_probe": deque(maxlen=200),
        }

        self._active_attacker_profile: dict[str, Any] = ATTACKER_PROFILES[0]

        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._closed = False
        self._startup_error: Exception | None = None
        self._svc_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._start_service_thread()

        print(
            f"[HoneypotEnv] Initialized | fast_training={fast_training} | "
            f"ports={self.ports} | target={target}"
        )

    # ------------------------------------------------------------------
    # Background service thread
    # ------------------------------------------------------------------

    def _start_service_thread(self) -> None:
        self._svc_thread = threading.Thread(
            target=self._run_service,
            name="adapttrap-service-thread",
            daemon=True,
        )
        self._svc_thread.start()

        ready = self._ready_event.wait(timeout=self.SERVICE_STARTUP_TIMEOUT_S)
        if not ready:
            self.close()
            raise RuntimeError(
                "Timed out waiting for HoneypotEnv services to start."
            )

        if self._startup_error is not None:
            err = self._startup_error
            self.close()
            raise RuntimeError(f"Failed to start HoneypotEnv services: {err}") from err

    def _run_service(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._serve())
        except Exception as e:
            if self._startup_error is None:
                print(f"[HoneypotEnv] Service loop exception: {e}")
        finally:
            pending = [t for t in asyncio.all_tasks(self._loop) if not t.done()]
            for task in pending:
                task.cancel()

            if pending:
                try:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                except Exception:
                    pass

            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except Exception:
                pass

            try:
                self._loop.close()
            except Exception:
                pass

    async def _serve(self) -> None:
        servers: list[asyncio.base_events.Server] = []

        try:
            servers.append(
                await asyncio.start_server(
                    self.simulator.handle_ssh,
                    self.target,
                    self.ports["ssh"],
                )
            )
            servers.append(
                await asyncio.start_server(
                    self.simulator.handle_http,
                    self.target,
                    self.ports["http"],
                )
            )
            servers.append(
                await asyncio.start_server(
                    self.simulator.handle_redis,
                    self.target,
                    self.ports["redis"],
                )
            )

            print(
                f"[HoneypotEnv] Services live â†’ "
                f"SSH:{self.ports['ssh']} "
                f"HTTP:{self.ports['http']} "
                f"Redis:{self.ports['redis']}"
            )
            self._ready_event.set()

            while not self._stop_event.is_set():
                await asyncio.sleep(0.05)

        except OSError as e:
            self._startup_error = e
            self._ready_event.set()
            print(f"[HoneypotEnv] !! SERVICE STARTUP FAILED: {e}")
            print(
                f"[HoneypotEnv] !! Check if ports {list(self.ports.values())} are already in use"
            )
            raise

        except Exception as e:
            self._startup_error = e
            self._ready_event.set()
            raise

        finally:
            for srv in servers:
                try:
                    srv.close()
                except Exception:
                    pass

            for srv in servers:
                try:
                    await srv.wait_closed()
                except Exception:
                    pass

            await asyncio.sleep(0)

            print("[HoneypotEnv] Services shut down cleanly.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("HoneypotEnv is closed.")

    def _attacker_index(self) -> int:
        return ATTACKER_NAME_TO_INDEX[self._active_attacker_profile["name"]]

    def _get_obs(self) -> np.ndarray:
        ports = self.PORT_CONFIGS[self._port_cfg_idx]

        attacker_idx = self._attacker_index()
        attacker_onehot = np.zeros(N_ATTACKER_TYPES, dtype=np.float32)
        attacker_onehot[attacker_idx] = 1.0

        base_obs = np.array(
            [
                self._profile_idx / (self.N_PROFILES - 1),
                self._temporal_idx / (self.N_TEMPORAL - 1),
                float(ports["ssh"]),
                float(ports["http"]),
                float(ports["redis"]),
                float(np.clip(self._last_suspicion, 0.0, 1.0)),
                float(min(self._last_depth / self.MAX_DEPTH, 1.0)),
                float(min(self._session_count / self.MAX_SESSIONS, 1.0)),
                float(min(self._consec_detects / 5.0, 1.0)),
            ],
            dtype=np.float32,
        )

        return np.concatenate([base_obs, attacker_onehot]).astype(np.float32)

    def _coerce_action(self, action: np.ndarray | list[int] | tuple[int, ...]) -> np.ndarray:
        arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if arr.shape != (3,):
            raise ValueError(f"Action must have shape (3,), got {arr.shape}")
        return arr

    def _apply_action(self, action: np.ndarray) -> list[str]:
        p_idx = int(action[0])
        t_idx = int(action[1])
        c_idx = int(action[2])

        if not (0 <= p_idx < self.N_PROFILES):
            raise ValueError(f"Invalid profile index: {p_idx}")
        if not (0 <= t_idx < self.N_TEMPORAL):
            raise ValueError(f"Invalid temporal index: {t_idx}")
        if not (0 <= c_idx < self.N_PORT_CFGS):
            raise ValueError(f"Invalid port config index: {c_idx}")

        changed: list[str] = []

        if p_idx != self._profile_idx:
            changed.append(f"profile:{self._profile_idx}->{p_idx}")
        if t_idx != self._temporal_idx:
            changed.append(f"temporal:{self._temporal_idx}->{t_idx}")
        if c_idx != self._port_cfg_idx:
            changed.append(f"ports:{self._port_cfg_idx}->{c_idx}")

        self._profile_idx = p_idx
        self._temporal_idx = t_idx
        self._port_cfg_idx = c_idx

        self.simulator.set_profile(p_idx)
        self.simulator.set_latency(self.TEMPORAL_LATENCY[t_idx])

        return changed

    def _build_attacker(self):
        attacker_name = self._active_attacker_profile["name"]
        attacker_cls = ATTACKER_FACTORY[attacker_name]
        active_ports = self.PORT_CONFIGS[self._port_cfg_idx]

        return attacker_cls(
            target=self.target,
            ports=self.ports,
            fast_training=self.fast_training,
            simulated_latency_ms=self.simulator.latency_ms,
            active_profile_name=self.simulator.identity.name,
            active_port_config=active_ports,
            attacker_profile=self._active_attacker_profile,
        )

    def _compute_reward(
        self,
        suspicion: float,
        depth: float,
        duration: float,
        flagged: bool,
    ) -> float:
        attacker_idx = self._attacker_index()

        if flagged:
            return float(DETECTION_PENALTY[attacker_idx])

        legitimacy = 1.0 - float(np.clip(suspicion, 0.0, 1.0))
        depth_score = float(min(depth / self.MAX_DEPTH, 1.0))

        dur_ratio = duration / max(self._baseline_dur, 1e-6)
        dur_score = float(min(dur_ratio, 1.5) / 1.5)

        raw = (
            0.45 * legitimacy
            + 0.35 * depth_score
            + 0.20 * dur_score
        )

        reward = raw * EVASION_BONUS[attacker_idx]
        return float(np.clip(reward, -4.0, 4.0))

    def _normalize_reward(self, raw_reward: float) -> float:
        
        attacker_name = self._active_attacker_profile["name"]
        history = self._reward_histories[attacker_name]
        history.append(raw_reward)

        arr  = np.array(history, dtype=np.float32)

        if len(arr) < 30:
            # Bayesian-lite warm-start: blend buffer stats with a weak prior
            prior_mean, prior_std = 0.0, 2.0
            w = len(arr) / 30.0 
            mean = w * float(np.mean(arr)) + (1.0 - w) * prior_mean
            std = max(w * float(np.std(arr)) + (1.0 - w) * prior_std, 0.5)
            normalised = (raw_reward - mean) / std
            return float(np.clip(normalised, -3.0, 3.0))

        mean = float(np.mean(arr))
        std = max(float(np.std(arr)), 0.5)
        normalised = (raw_reward - mean) / std
        return float(np.clip(normalised, -3.0, 3.0))

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None, attacker_name: str | None = None):
        self._ensure_open()
        super().reset(seed=seed)

        if attacker_name is not None:
            matches = [p for p in ATTACKER_PROFILES if p["name"] == attacker_name]
            if not matches:
                raise ValueError(f"Unknown attacker_name: {attacker_name}")
            self._active_attacker_profile = matches[0]
        else:
            self._active_attacker_profile = random.choice(ATTACKER_PROFILES)

        self._profile_idx = 0
        self._temporal_idx = 0
        self._port_cfg_idx = 0

        self._last_suspicion = 0.0
        self._last_depth = 0.0
        self._last_duration = self._baseline_dur

        self._session_count = 0
        self._consec_detects = 0
        self._episode_reward = 0.0

        self.simulator.set_profile(0)
        self.simulator.set_latency(self.TEMPORAL_LATENCY[0])

        return self._get_obs(), {
            "attacker": self._active_attacker_profile["name"],
            "attacker_profile": dict(self._active_attacker_profile),
        }

    def step(self, action: np.ndarray):
        self._ensure_open()

        action = self._coerce_action(action)
        changes = self._apply_action(action)
        attacker = self._build_attacker()

        try:
            result = attacker.run_session(verbose=False)
        except Exception as e:
            print(f"[HoneypotEnv] !! ATTACKER EXCEPTION at step {self._session_count}: {e}")
            print(f"[HoneypotEnv] !! Action was: {action.tolist()}")
            print(f"[HoneypotEnv] !! Ports: {self.ports} | Target: {self.target}")
            print(f"[HoneypotEnv] !! Attacker: {self._active_attacker_profile['name']}")
            raise

        suspicion = float(result.suspicion_score)
        depth = float(result.interaction_depth)
        duration = float(result.session_duration_seconds)
        flagged = bool(result.flagged_as_honeypot)

        self._last_suspicion = suspicion
        self._last_depth = depth
        self._last_duration = duration
        self._session_count += 1

        if flagged:
            self._consec_detects += 1
        else:
            self._consec_detects = 0

        raw_reward = self._compute_reward(
            suspicion=suspicion,
            depth=depth,
            duration=duration,
            flagged=flagged,
        )
        reward = self._normalize_reward(raw_reward)
        self._episode_reward += raw_reward

        terminated = self._consec_detects >= 5
        truncated = self._session_count >= self.MAX_SESSIONS

        obs = self._get_obs()
        info = {
            "suspicion": suspicion,
            "depth": depth,
            "duration": duration,
            "flagged": flagged,
            "session": self._session_count,
            "episode_reward": self._episode_reward,
            "raw_reward": raw_reward,
            "normalized_reward": reward,
            "action": action.tolist(),
            "changes": changes,
            "consec_detects": self._consec_detects,
            "attacker_type": self._active_attacker_profile["name"],
            "attacker_decisions": getattr(result, "decisions", []),
            "suspicion_reasons": getattr(result, "suspicion_reasons", []),
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._closed:
            return

        self._closed = True
        self._stop_event.set()

        if self._svc_thread and self._svc_thread.is_alive():
            self._svc_thread.join(timeout=self.SERVICE_JOIN_TIMEOUT_S)

        if self._svc_thread and self._svc_thread.is_alive():
            print("[HoneypotEnv] Warning: service thread did not exit within timeout.")

        print("[HoneypotEnv] Environment closed.")