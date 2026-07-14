# AdaptTrap Final Architecture

## 1. Purpose

AdaptTrap is an adaptive cyber deception benchmark in which a reinforcement-learning defender interacts with simulated attacker sessions inside a controlled honeypot environment.

The system is designed to answer one core question:

> Can a defender learn to adapt identity, timing, and exposed services in order to keep attackers engaged while reducing honeypot detection?

This document freezes the architecture used for the benchmark core before adding later layers such as evaluation reporting, memory, STIX export, and LLM realism.

---

## 2. System Boundary

The benchmark core includes:

- A live local deception environment (`env/honeypot_env.py`)
- A service simulator for SSH, HTTP, and Redis (`env/service_simulator.py`)
- Identity profiles (`env/identity.py`)
- Three attacker tiers
- A custom attacker-conditioned actor-critic defender
- A curriculum-based training loop

The benchmark core does **not** yet include:

- Persistent attacker memory
- STIX 2.1 export
- LLM-backed response generation
- Final dashboard/reporting layer

Those belong to later steps.

---

## 3. Defender Architecture

### 3.1 RL algorithm

The defender uses a **custom Actor-Critic with GAE**.

This is not PPO and not Stable-Baselines3.  
The training implementation is custom and lives in `train.py`.

### 3.2 Network structure

The defender model (`defender/actor_critic.py`) uses:

- One shared trunk
- Three attacker-specific policy heads
- Three attacker-specific value heads

The attacker types are:

1. `nmap_scanner`
2. `scripted_exploit`
3. `ai_probe`

This architecture isolates attacker-specific policy/value learning while still sharing common state features in the trunk.

### 3.3 State / observation space

Observation dimension: **12**

The observation contains:

#### Base state features (9)
1. Current identity profile index (normalized)
2. Current temporal profile index (normalized)
3. SSH exposed flag
4. HTTP exposed flag
5. Redis exposed flag
6. Last suspicion score
7. Last interaction depth (normalized)
8. Session count (normalized)
9. Consecutive detections (normalized)

#### Attacker identity features (3)
10. One-hot: `nmap_scanner`
11. One-hot: `scripted_exploit`
12. One-hot: `ai_probe`

### 3.4 Action space

The defender action space is:

`MultiDiscrete([4, 4, 3])`

Meaning:

- 4 identity profiles
- 4 temporal response profiles
- 3 port configurations

This yields **48 joint configurations**.

The defender does not choose from a flat 17-action list.  
The benchmark uses structured multi-axis action composition.

---

## 4. Environment Architecture

The environment is implemented in `env/honeypot_env.py`.

### 4.1 Defender-controlled dimensions

The defender controls:

- Identity profile
- Temporal response profile
- Exposed service set

### 4.2 Simulated services

The service simulator exposes:

- SSH
- HTTP
- Redis

### 4.3 Episode structure

Each environment step launches one attacker session against the currently active deception configuration.

An episode ends when either:

- 5 consecutive detections occur, or
- 20 sessions are reached

### 4.4 Ports

Training ports:

- SSH: 12222
- HTTP: 18080
- Redis: 16379

Evaluation ports:

- SSH: 2222
- HTTP: 8080
- Redis: 6379

---

## 5. Attacker Tiers

### 5.1 Level 1 — Nmap-like scanner

File: `attackers/level1_nmap.py`

Purpose:
- Lightweight reconnaissance baseline
- Fast and shallow probing behavior

### 5.2 Level 2 — Scripted exploit attacker

File: `attackers/level2_scripted.py`

Purpose:
- Stronger deterministic attacker
- Multi-service scripted probing and exploit-style behavior

### 5.3 Level 3 — Bandit-style adaptive probe learner

File: `attackers/level3_bandit.py`

Purpose:
- Adaptive session-level attacker
- Stronger than Level 2 because it updates internal action values during the session

Core properties:

- Epsilon-greedy probe selection
- Running Q-value updates
- 8 probe actions:
  - `ssh_banner`
  - `ssh_creds`
  - `http_root`
  - `http_traversal`
  - `redis_ping`
  - `redis_info`
  - `latency_probe`
  - `consistency_probe`
- Early stop when suspicion crosses threshold
- Cross-service and timing consistency checks

This attacker is the benchmark’s hardest non-LLM attacker in the core system.

---

## 6. Reward Design

Reward is computed from:

- Legitimacy / low suspicion
- Interaction depth
- Session duration
- Attacker-specific evasion bonus
- Detection penalty when flagged

Reward normalization is attacker-specific and history-aware.

The benchmark optimizes for:

- keeping attacker suspicion low,
- maintaining deeper interaction,
- extending believable engagement,
- avoiding repeated detection collapse.

---

## 7. Training Design

The training loop is implemented in `train.py`.

### 7.1 Algorithm details

- Custom Actor-Critic
- GAE for return estimation
- Gradient clipping
- Entropy regularization per attacker
- Separate optimizer learning-rate groups for:
  - trunk
  - policy heads
  - value heads

### 7.2 Curriculum

Training phases:

1. Phase 1: `nmap_scanner`
2. Phase 2: `nmap_scanner` + `scripted_exploit`
3. Phase 3: `nmap_scanner` + `scripted_exploit` + `ai_probe`

### 7.3 Stability measures

Implemented stabilization choices include:

- Logit clamping before categorical sampling
- Return clipping
- Critic loss soft handling for difficult `ai_probe` phases
- Fixed random seed
- Per-attacker reward normalization
- Per-attacker entropy coefficients

---

## 8. What is frozen by this architecture

The following decisions are now frozen for the benchmark core:

- RL algorithm: custom Actor-Critic + GAE
- Defender architecture: shared trunk + attacker-specific heads
- Observation size: 12
- Action space: `MultiDiscrete([4,4,3])`
- Hardest benchmark attacker: `Level3BanditAttacker`
- Live services: SSH, HTTP, Redis
- Benchmark scope excludes memory/STIX/LLM until later steps

---

## 9. Deferred layers

These are explicitly outside the benchmark core and will be added later:

### Step 3
- Evaluation framework
- Metrics and baselines
- Reporting

### Step 4
- SQLite attacker memory
- STIX 2.1 export

### Step 5
- Ollama / Phi-3 realism layer for eval/demo only

### Step 6
- Productization, tests, dashboard, packaging

---

## 10. Final statement

AdaptTrap’s benchmark core is a **custom multi-head actor-critic deception benchmark** with three attacker tiers, live local services, structured defender actions, and curriculum-based training.

Any future layer must remain compatible with this frozen core unless a new architecture revision is approved.