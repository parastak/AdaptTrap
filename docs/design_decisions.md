# AdaptTrap Design Decisions

## 1. Why this document exists

This file records the major architectural decisions for AdaptTrap so the project stops drifting between pitch language, partial code, and future ideas.

If a future change conflicts with this document, the change must be deliberate and explicit.

---

## 2. Decision: RL algorithm

### Chosen
**Custom Actor-Critic with GAE**

### Rejected
- PPO via Stable-Baselines3
- Generic off-the-shelf RL wrappers

### Why

We already implemented attacker-conditioned learning using:

- shared trunk
- attacker-specific policy heads
- attacker-specific value heads
- attacker-specific entropy coefficients
- custom curriculum
- custom stabilization logic for harder attackers

That is not a trivial SB3 setup.

Migrating to PPO/SB3 now would either:
1. reduce architectural flexibility, or
2. require heavy wrapper logic that recreates the same complexity under a prettier label.

So the honest decision is:
**keep the custom algorithm and update documentation to match reality.**

---

## 3. Decision: defender network structure

### Chosen
**Shared trunk + per-attacker policy heads + per-attacker value heads**

### Rejected
- One single shared policy/value head for all attackers
- One fully separate network per attacker

### Why

A single shared head risks interference and forgetting across attacker types.

A fully separate network per attacker throws away shared generalization and increases maintenance burden.

The selected hybrid design preserves shared learning while isolating attacker-specific behavior where it matters.

---

## 4. Decision: Level 3 attacker design

### Chosen
**Bandit-style adaptive probe learner (`level3_bandit.py`)**

### Rejected
- Reusing Level 2 with minor tweaks
- LLM attacker inside benchmark core
- Fake “AI attacker” that is just random probing with a dramatic filename

### Why

The current Level 3 attacker is already meaningful:

- adaptive probe selection
- session-local Q-value updates
- timing checks
- cross-service consistency checks
- early stopping on suspicion threshold

This gives a stronger benchmark attacker without adding LLM latency, cost, or reproducibility problems.

LLM attackers remain a future evaluation/demo layer, not part of the benchmark core.

---

## 5. Decision: action space

### Chosen
**Structured action space: `MultiDiscrete([4,4,3])`**

### Rejected
- Flat discrete action catalog
- “17 actions” pitch-style simplification

### Why

The defender really makes three coordinated decisions:

1. which identity to present,
2. which timing mode to use,
3. which ports to expose.

That is naturally modeled as a structured multi-axis action.

A flat action list would make the design less interpretable and harder to extend cleanly.

---

## 6. Decision: observation space

### Chosen
**12-dimensional observation**

### Includes
- current deception configuration
- recent suspicion/depth/session history
- attacker one-hot identity

### Rejected
- oversized early observation vectors
- hidden attacker type
- memory-based features before Step 4

### Why

The current state is enough for the benchmark core and keeps training manageable.

Adding memory/history-rich cross-session features before Step 4 would mix benchmark core with persistence logic too early.

---

## 7. Decision: benchmark attacker tiers

### Chosen
Three tiers:

1. `nmap_scanner`
2. `scripted_exploit`
3. `ai_probe`

### Why

This creates a curriculum from simple to adaptive:

- reconnaissance
- scripted exploitation
- adaptive probing

This structure is sufficient for a defensible benchmark core.

---

## 8. Decision: service scope

### Chosen
- SSH
- HTTP
- Redis

### Rejected
- adding many services early
- fake breadth over real consistency

### Why

Three services are enough to support:
- identity shaping,
- cross-service consistency checks,
- port exposure control,
- believable deception dynamics.

More services right now would increase maintenance and instability faster than they increase research value.

---

## 9. Decision: what is NOT in the benchmark core

The following are intentionally deferred:

- SQLite attacker memory
- STIX 2.1 bundle generation
- LLM response realism
- final dashboard/product layer

### Why

These are useful system layers, but they are not needed to prove the core RL benchmark works.

We finish proof first, then platform features.

---

## 10. Decision: training before evaluation

### Chosen
Freeze architecture first, then run official training, then build evaluation.

### Why

Evaluation on an unstable or undefined architecture is garbage with charts.

A benchmark only matters if the system being benchmarked is clearly defined.

---

## 11. Final decision summary

AdaptTrap is officially defined as:

- a custom Actor-Critic + GAE defender,
- operating in a live local deception environment,
- against three attacker tiers,
- with structured defender actions,
- using a Level 3 bandit attacker as the hardest benchmark core attacker,
- and deferring memory/STIX/LLM layers to later steps.

Any conflicting older pitch language should be updated to match this reality.