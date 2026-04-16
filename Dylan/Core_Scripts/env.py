"""
Single entry-point for creating and inspecting the aquarium_v0 environment.

All tuneable parameters are grouped in ENV_CONFIG so you have one place
to adjust the experiment.

Usage:
    from Core_Scripts.env import make_env, get_space_dims, ENV_CONFIG

    env = make_env()
    dims = get_space_dims(env)
    obs, infos = env.reset(seed=42)
"""

import os
if not os.environ.get("_AQUARIUM_RENDER"):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from marl_aquarium import aquarium_v0
from marl_aquarium.env.aquarium import raw_env as _RawEnv


# ═════════════════════════════════════════════════════════════════════════════
# Monkey-patches for marl_aquarium observation bugs (surface with 2+ predators)
#
# Bugs fixed:
#   1. predator_nearby_shark_observations: non-FOV loop runs unconditionally
#      (should be else), FOV check uses prey params instead of predator params.
#   2. prey_nearby_sharks_observations: FOV branch doesn't cap at n_nearest.
#   3. predator_nearby_fish_observations: FOV branch doesn't cap at
#      prey_observe_count, uses prey params instead of predator params.
# ═════════════════════════════════════════════════════════════════════════════

def _fixed_predator_nearby_shark_observations(self, observer):
    observations = []
    limit = self.predator_observe_count * self.obs_size
    if self.fov_enabled:
        for shark in self.predators:
            if (
                shark is not observer
                and self.torus.check_if_entity_is_in_view_in_torus(
                    observer, shark, self.predator_view_distance, self.predator_fov
                )
                and len(observations) < limit
            ):
                observations += self.nearby_animal_observation(observer, shark)
    else:
        dists = sorted(
            ((self.torus.get_distance_in_torus(observer.position, s.position), s)
             for s in self.predators if s is not observer),
            key=lambda x: x[0],
        )
        for _, shark in dists[: self.predator_observe_count]:
            observations += self.nearby_animal_observation(observer, shark)
    if len(observations) < limit:
        observations += [0] * (limit - len(observations))
    return observations


def _fixed_prey_nearby_sharks_observations(self, observer, all_sharks, n_nearest_shark):
    observations = []
    limit = n_nearest_shark * self.obs_size
    if self.fov_enabled:
        for shark in all_sharks:
            if (
                self.torus.check_if_entity_is_in_view_in_torus(
                    observer, shark, self.prey_view_distance, self.prey_fov
                )
                and len(observations) < limit
            ):
                observations += self.nearby_animal_observation(observer, shark)
    else:
        closest = self.prey_get_n_closest_animals(
            observer, all_sharks, self.predator_observe_count
        )
        for shark in closest:
            observations += self.nearby_animal_observation(observer, shark)
    if len(observations) < limit:
        observations += [0] * (limit - len(observations))
    assert len(observations) == limit
    return observations


def _fixed_predator_nearby_fish_observations(self, observer):
    observations = []
    limit = self.prey_observe_count * self.obs_size
    if self.fov_enabled:
        for fish in self.prey:
            if (
                self.torus.check_if_entity_is_in_view_in_torus(
                    observer, fish, self.predator_view_distance, self.predator_fov
                )
                and len(observations) < limit
            ):
                observations += self.nearby_animal_observation(observer, fish)
    else:
        closest = self.predator_get_n_closest_fish(observer)
        for fish in closest:
            if fish is not observer:
                observations += self.nearby_animal_observation(observer, fish)
    if len(observations) < limit:
        observations += [0] * (limit - len(observations))
    return observations


_RawEnv.predator_nearby_shark_observations = _fixed_predator_nearby_shark_observations
_RawEnv.prey_nearby_sharks_observations = _fixed_prey_nearby_sharks_observations
_RawEnv.predator_nearby_fish_observations = _fixed_predator_nearby_fish_observations


# ═════════════════════════════════════════════════════════════════════════════
# Global experiment configuration — edit THIS to change the experiment
# ═════════════════════════════════════════════════════════════════════════════

ENV_CONFIG = {

    # ── World ─────────────────────────────────────────────────────────────
    "width": 800,
    "height": 800,
    "max_time_steps": 500,           # episode length (shorter = more updates/hr)

    # ── Agents ────────────────────────────────────────────────────────────
    "predator_count": 2,
    "prey_count": 3,

    # ── Actions ───────────────────────────────────────────────────────────
    "action_count": 16,              # discrete heading directions (360/16 = 22.5°)

    # ── Observations ──────────────────────────────────────────────────────
    "fov_enabled": True,             # False = always see N nearest (easier to learn)
    "predator_observe_count": 1,     # how many nearby predators each agent sees
    "prey_observe_count": 3,         # how many nearby prey each agent sees
    "predator_fov": 150,             # predator cone angle (degrees)
    "prey_fov": 120,                 # prey cone angle (degrees)
    "predator_view_distance": 300,   # how far predators can see (pixels)
    "prey_view_distance": 100,       # how far prey can see (pixels)
    "observable_walls": 2,

    # ── Predator physics ──────────────────────────────────────────────────
    "predator_max_velocity": 5,
    "predator_max_acceleration": 0.6,
    "predator_max_steer_force": 0.6,
    "predator_radius": 30,          # hitbox size

    # ── Prey physics ──────────────────────────────────────────────────────
    "prey_max_velocity": 4,
    "prey_max_acceleration": 1.0,
    "prey_max_steer_force": 0.6,
    "prey_radius": 20,              # hitbox size

    # ── Catching / rewards ────────────────────────────────────────────────
    "catch_radius": 120,             # proximity for catch event
    "predator_reward": 10,           # base env reward per catch (broken w/ constant prey)
    "prey_reward": 0,                # per-step survival reward per prey
    "prey_punishment": 1,         # death penalty for prey

    # ── Spawning / lifecycle ──────────────────────────────────────────────
    "keep_prey_count_constant": True,  # True = prey respawn (must be True, env bug)
    "predator_max_age": 10000,         # must exceed max_time_steps (env bug)
    "procreate": False,
    "prey_replication_age": 200,
    "max_prey_count": 20,

    # ── Reward shaping (applied in training loop, NOT in marl_aquarium) ──
    "predator_catch_bonus": 10.0,       # reward given to predator when prey dies
    "predator_proximity_reward": 0.0,   # per-step reward scaled by closeness to prey
    "prey_proximity_penalty": 0.0,      # per-step penalty for prey being near predator
    "predator_step_penalty": 0.0,       # per-step cost scaled by speed (movement tax)
    "predator_no_catch_step_penalty": 0.0,  # subtract each step when no prey caught this step
}


# ═════════════════════════════════════════════════════════════════════════════
# Environment factory
# ═════════════════════════════════════════════════════════════════════════════

# Keys that are OUR custom additions (not passed to marl_aquarium)
_CUSTOM_KEYS = {
    "predator_catch_bonus",
    "predator_proximity_reward",
    "prey_proximity_penalty",
    "predator_step_penalty",
    "predator_no_catch_step_penalty",
}


def make_env(render=False, **overrides):
    """
    Create and return a *parallel* aquarium_v0 environment.

    Parameters
    ----------
    render : bool
        If False the environment runs headless (much faster for training).
    **overrides
        Any key from ENV_CONFIG to override for this run.
    """
    cfg = {**ENV_CONFIG, **overrides}
    if not render:
        cfg["render_mode"] = "rgb_array"
    else:
        cfg["render_mode"] = "human"

    env_kwargs = {k: v for k, v in cfg.items() if k not in _CUSTOM_KEYS}
    return aquarium_v0.parallel_env(**env_kwargs)


def get_space_dims(env):
    """
    Return observation and action dimensions for each role.

    Returns
    -------
    dict  {"predator": (obs_dim, act_dim), "prey": (obs_dim, act_dim)}
    """
    dims = {}
    for agent_name in env.possible_agents:
        role = "predator" if agent_name.startswith("predator") else "prey"
        if role not in dims:
            obs_dim = env.observation_space(agent_name).shape[0]
            act_dim = env.action_space(agent_name).n
            dims[role] = (obs_dim, act_dim)
    return dims


def get_global_obs_dim(env, role="predator"):
    """Observation dimension across same-team agents (for MAPPO centralized critic)."""
    return sum(env.observation_space(a).shape[0]
               for a in env.possible_agents if get_role(a) == role)


def get_role(agent_name):
    """Return 'predator' or 'prey' from an agent id string."""
    return "predator" if agent_name.startswith("predator") else "prey"


def get_agents_by_role(env):
    """
    Split env.possible_agents into role groups.

    Returns
    -------
    dict  {"predator": ["predator_0", ...], "prey": ["prey_0", ...]}
    """
    groups = {"predator": [], "prey": []}
    for name in env.possible_agents:
        groups[get_role(name)].append(name)
    return groups
