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

## MDP Formulation

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
| 1 | x |