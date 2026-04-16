# Emergent Behaviour in Predator-Prey MARL

Multi-agent reinforcement learning experiments in the [marl-aquarium](https://github.com/michaelkoelle/marl-aquarium) `aquarium_v0` environment.

## Project Structure

```
Dylan/
├── Core_Scripts/
│   ├── env.py          # ENV_CONFIG + environment factory + space helpers
│   ├── buffer.py       # ReplayBuffer (off-policy) and RolloutBuffer (on-policy)
│   └── logger.py       # CSV + optional TensorBoard logging
│
├── Agents/
│   ├── random_policy.py  # Random baseline agent
│   ├── ppo_agent.py      # IPPO (Independent PPO with parameter sharing)
│   └── mappo_agent.py    # MAPPO (Multi-Agent PPO, centralized critic)
│
├── run_experiment.py     # Master training/baseline script
├── run_tournament.py     # Cross-evaluate saved policies              [TODO]
├── visual.py             # Visualise an episode with trained weights
└── README.md
```

AQUARIUM BASICS

### State Space (Observations)

Each agent receives a **local, partial observation** — a continuous float vector in `Box(0, 1)`. All values are normalised to [0, 1].

**Self observation** (5 values):

| Index | Value | Description |
|---|---|---|
| 0 | entity_type | 0 = predator, 1 = prey |
| 1 | x | Normalised x position in world |
| 2 | y | Normalised y position in world |
| 3 | direction | Normalised orientation angle (-180..180 → 0..1) |
| 4 | speed | Normalised current speed (0..max_speed → 0..1) |

**Per observed agent** (6 values each, repeated for `predator_observe_count` predators + `prey_observe_count` prey):

| Index | Value | Description |
|---|---|---|
| 0 | entity_type | 0 = predator, 1 = prey |
| 1 | x | Normalised x position of the observed agent |
| 2 | y | Normalised y position of the observed agent |
| 3 | distance | Normalised distance from observer (0..view_distance → 0..1) |
| 4 | direction | Normalised relative bearing (-180..180 → 0..1) |
| 5 | speed | Normalised speed of the observed agent |

If fewer agents are visible (e.g. outside FOV cone), the remaining slots are **zero-filled**.

**Total observation dimension** = 5 + `predator_observe_count` x 6 + `prey_observe_count` x 6

With our defaults (`predator_observe_count=1`, `prey_observe_count=3`):
- **Predator obs dim** = 5 + 1x6 + 3x6 = **29**
- **Prey obs dim** = 5 + 1x6 + 3x6 = **29**

When `fov_enabled=True`, agents only observe others within their view cone (angle + distance). When `fov_enabled=False`, they always see the N nearest regardless of direction.

### Action Space

**Discrete(16)** — 16 evenly spaced heading directions around 360 degrees.

| Action | Heading |
|---|---|
| 0 | 0° (right) |
| 1 | 22.5° |
| 2 | 45° |
| ... | ... |
| 15 | 337.5° |

The chosen action sets a **desired velocity** (unit vector in the chosen direction, scaled to `max_velocity`). This is not teleportation — it's a target the physics engine steers toward.

### Transition Dynamics

The world is an **800x800 toroidal grid** (agents wrap around edges).

Each step, for every agent:
1. **Steering force** = desired_velocity - current_velocity, clamped to `max_steer_force`
2. **Acceleration** = steering force, normalised and scaled to `max_acceleration`
3. **Velocity** += acceleration, clamped to `max_velocity`
4. **Position** += velocity (with toroidal wrapping)
5. **Collision**: agents bounce off same-type entities they overlap with
6. **Catch check**: if a prey is within `catch_radius` of a predator, the prey dies (and respawns if `keep_prey_count_constant=True`)

| Parameter | Predator | Prey |
|---|---|---|
| `max_velocity` | 5.0 | 4.0 |
| `max_acceleration` | 0.6 | 1.0 |
| `max_steer_force` | 0.6 | 0.6 |
| `radius` (hitbox) | 30 | 20 |

Predators are faster but turn/accelerate slower. Prey are slower but more agile.

### Reward Structure

**Base rewards from marl_aquarium:**

| Event | Agent | Reward |
|---|---|---|
| Each time step | Prey | +`prey_reward` (default 1) |
| Prey caught (dies) | Prey | -`prey_punishment` (default 1000) |
| Prey caught | Predator | +`predator_reward` (default 10, **broken** when `keep_prey_count_constant=True`) |

**Custom reward shaping??**

| Shaping | Description | Config key |
|---|---|---|
| Catch bonus | Detect prey death (reward <= -500), give predator +bonus | `predator_catch_bonus` (default 10.0) |
| Proximity reward | Per-step bonus for predator approaching prey | `predator_proximity_reward` (default 0.0) |
| Proximity penalty | Per-step penalty for prey near predator | `prey_proximity_penalty` (default 0.0) |

### Episode Termination

Episodes end when `max_time_steps` is reached. With `keep_prey_count_constant=True` and `predator_max_age` set high, no early termination occurs — every episode runs for exactly `max_time_steps` steps.

## Algorithms

### IPPO (ppo_agent.py)

Independent PPO with parameter sharing. All agents of the same role share one policy (actor + critic). Both actor and critic use only local observations.

### MAPPO (mappo_agent.py)

Multi-Agent PPO (CTDE). Same actor as IPPO (local obs → action), but the **critic is centralized**: it takes the concatenation of ALL agents' observations as input. The centralized critic is only used during training — at test time, execution is fully decentralized.

| | IPPO | MAPPO |
|---|---|---|
| Actor input | Local obs (29-dim) | Local obs (29-dim) |
| Critic input | Local obs (29-dim) | Global obs (all agents concatenated) |
| Parameter sharing | Yes (per role) | Yes (per role) |
| Execution | Decentralized | Decentralized |




Possible training loop.......

```bash
# Phase A: PPO predator learns basic hunting
python run_experiment.py --predator_agent ppo --prey_agent random \
    --episodes 5000 --save_weights --log_dir results/phaseA

# Phase B: PPO prey learns basic evasion
python run_experiment.py --predator_agent random --prey_agent ppo \
    --episodes 5000 --save_weights --log_dir results/phaseB

# Phase C: Co-evolution — load both, continue training
python run_experiment.py --predator_agent ppo --prey_agent ppo \
    --load_predator_weights results/phaseA/.../weights/predator.pt \
    --load_prey_weights results/phaseB/.../weights/prey.pt \
    --episodes 10000 --save_weights --log_dir results/phaseC

# Phase D: MAPPO predator vs trained PPO prey
python run_experiment.py --predator_agent mappo --prey_agent ppo \
    --load_prey_weights results/phaseB/.../weights/prey.pt \
    --episodes 10000 --save_weights --log_dir results/phaseD
```

## CLI Flags

| Flag | Description |
|---|---|
| `--episodes N` | Number of training episodes (default 500) |
| `--predator_agent TYPE` | `random`, `ppo`, `mappo` |
| `--prey_agent TYPE` | `random`, `ppo`, `mappo` |
| `--save_weights` | Save agent weights after training |
| `--load_predator_weights PATH` | Load predator weights from file |
| `--load_prey_weights PATH` | Load prey weights from file |
| `--max_steps N` | Override episode length |
| `--catch_radius N` | Override catch radius |
| `--no_fov` | Disable FOV cone (agents see N nearest always) |
| `--predator_proximity_reward F` | Dense shaping: per-step bonus for approaching prey |
| `--prey_proximity_penalty F` | Dense shaping: per-step penalty for being near predator |
| `--render` | Show Pygame window |
| `--tensorboard` | Enable TensorBoard logging |

## aquarium_v0 Environment Parameters

All defaults are configurable via `ENV_CONFIG` in `Core_Scripts/env.py` or via CLI overrides.

Reference from [marl-aquarium](https://github.com/michaelkoelle/marl-aquarium):

| Parameter | Description | Default |
|---|---|---|
| `render_mode` | Rendering mode: `"human"` or `"rgb_array"` | `"human"` |
| `observable_walls` | Number of observable walls for agents | 2 |
| `width` | Environment window width (pixels) | 800 |
| `height` | Environment window height (pixels) | 800 |
| `caption` | Window caption | `"Aquarium"` |
| `fps` | Frames per second | 60 |
| `max_time_steps` | Maximum time steps per episode | 3000 |
| `action_count` | Number of discrete actions (heading directions) | 16 |
| `predator_count` | Number of predators | 1 |
| `prey_count` | Number of prey | 16 |
| `predator_observe_count` | Number of predators observable by each agent | 1 |
| `prey_observe_count` | Number of prey observable by each agent | 3 |
| `draw_force_vectors` | Draw force vectors (debug) | False |
| `draw_action_vectors` | Draw action vectors (debug) | False |
| `draw_view_cones` | Draw view cones (debug) | False |
| `draw_hit_boxes` | Draw hit boxes (debug) | False |
| `draw_death_circles` | Draw death circles (debug) | False |
| `fov_enabled` | Enable field-of-view cone for agents | True |
| `keep_prey_count_constant` | Respawn prey on death (keep count constant) | True |
| `prey_radius` | Prey hitbox radius | 20 |
| `prey_max_acceleration` | Maximum prey acceleration | 1.0 |
| `prey_max_velocity` | Maximum prey velocity | 4.0 |
| `prey_view_distance` | Prey view distance (pixels) | 100 |
| `prey_replication_age` | Age at which prey replicate | 200 |
| `prey_max_steer_force` | Maximum prey steering force | 0.6 |
| `prey_fov` | Prey field-of-view angle (degrees) | 120 |
| `prey_reward` | Per-step survival reward for prey | 1 |
| `prey_punishment` | Death penalty for prey | 1000 |
| `max_prey_count` | Maximum prey in environment | 20 |
| `predator_max_acceleration` | Maximum predator acceleration | 0.6 |
| `predator_radius` | Predator hitbox radius | 30 |
| `predator_max_velocity` | Maximum predator velocity | 5.0 |
| `predator_view_distance` | Predator view distance (pixels) | 200 |
| `predator_max_steer_force` | Maximum predator steering force | 0.6 |
| `predator_max_age` | Maximum predator age (steps before death) | 3000 |
| `predator_fov` | Predator field-of-view angle (degrees) | 150 |
| `predator_reward` | Reward for predator catching prey | 10 |
| `catch_radius` | Radius within which predators catch prey | 100 |
| `procreate` | Whether entities can procreate | False |
