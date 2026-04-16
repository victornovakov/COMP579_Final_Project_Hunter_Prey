"""
Visualize a single episode using trained weights.

Usage:
    # MAPPO predator (shared weights) vs random prey
    python visual.py --predator_agent mappo \
        --load_predator_weights results/comparison/mappo_vs_random_.../weights/predator.pt

    # PPO predator vs random prey
    python visual.py --predator_agent ppo --load_predator_weights path/to/predator.pt

    # Random vs random (no weights needed)
    python visual.py
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Tell env module we want a real display (must be set before import).
os.environ["_AQUARIUM_RENDER"] = "1"

# Import env module FIRST so monkey-patches are applied before any env is created.
from Core_Scripts.env import ENV_CONFIG, get_space_dims, get_global_obs_dim

# Ensure dummy drivers are removed so pygame opens a real window.
os.environ.pop("SDL_VIDEODRIVER", None)
os.environ.pop("SDL_AUDIODRIVER", None)

from marl_aquarium import aquarium_v0
import numpy as np


def get_role(name):
    return "predator" if name.startswith("predator") else "prey"


def run(args):
    cfg = ENV_CONFIG.copy()
    fov_enabled = not args.no_fov

    predator_view_distance = cfg["predator_view_distance"]
    prey_view_distance = cfg["prey_view_distance"]
    if not fov_enabled:
        # In marl_aquarium, entity distances are scaled by observer.view_distance with an assertion.
        # When FOV is disabled, agents can observe the N-nearest anywhere in the tank, so we must
        # ensure view_distance covers the full world to avoid assertion errors.
        world_diag = int((args.width ** 2 + args.height ** 2) ** 0.5) + 50
        predator_view_distance = max(predator_view_distance, world_diag)
        prey_view_distance = max(prey_view_distance, world_diag)

    predator_max_velocity = cfg["predator_max_velocity"] * args.predator_speed_mult

    env = aquarium_v0.env(
        render_mode="human",
        width=args.width,
        height=args.height,
        prey_count=args.prey_count,
        predator_count=args.predator_count,
        catch_radius=args.catch_radius,
        predator_view_distance=predator_view_distance,
        prey_view_distance=prey_view_distance,
        predator_fov=cfg["predator_fov"],
        prey_fov=cfg["prey_fov"],
        fov_enabled=fov_enabled,
        max_time_steps=args.max_steps,
        action_count=cfg["action_count"],
        predator_observe_count=cfg["predator_observe_count"],
        prey_observe_count=cfg["prey_observe_count"],
        predator_max_velocity=predator_max_velocity,
        keep_prey_count_constant=True,
        predator_max_age=cfg["predator_max_age"],
        draw_force_vectors=args.draw_forces,
        draw_view_cones=args.draw_cones,
        draw_hit_boxes=args.draw_hitboxes,
    )

    # Create a headless parallel env just to read obs/act dimensions.
    temp_env = aquarium_v0.parallel_env(
        width=args.width,
        height=args.height,
        prey_count=args.prey_count,
        predator_count=args.predator_count,
        action_count=cfg["action_count"],
        predator_observe_count=cfg["predator_observe_count"],
        prey_observe_count=cfg["prey_observe_count"],
        predator_view_distance=predator_view_distance,
        prey_view_distance=prey_view_distance,
        predator_fov=cfg["predator_fov"],
        prey_fov=cfg["prey_fov"],
        fov_enabled=fov_enabled,
        catch_radius=args.catch_radius,
        predator_max_velocity=predator_max_velocity,
        render_mode="rgb_array",
    )
    dims = get_space_dims(temp_env)
    global_obs_dim = get_global_obs_dim(temp_env)
    try:
        temp_env.close()
    except SystemExit:
        pass

    agents = {}
    for role in ("predator", "prey"):
        agent_type = getattr(args, f"{role}_agent")
        weight_path = getattr(args, f"load_{role}_weights")
        obs_dim, act_dim = dims[role]

        if agent_type == "ppo":
            from Agents.ppo_agent import PPOAgent
            agent = PPOAgent(obs_dim, act_dim)
            if weight_path:
                agent.load(weight_path)
                print(f"Loaded {role} PPO weights from {weight_path}")
            else:
                print(f"WARNING: {role} is PPO but no weights loaded — using random init")
            agents[role] = agent
        elif agent_type == "mappo":
            from Agents.mappo_agent import MAPPOAgent
            agent = MAPPOAgent(obs_dim, act_dim, global_obs_dim=global_obs_dim)
            if weight_path:
                agent.load(weight_path)
                print(f"Loaded {role} MAPPO weights from {weight_path}")
            else:
                print(f"WARNING: {role} is MAPPO but no weights loaded — using random init")
            agents[role] = agent
        else:
            agents[role] = None

    env.reset(seed=args.seed)
    step = 0
    last_agent = env.possible_agents[-1]

    try:
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                role = get_role(agent_name)
                if agents[role] is not None:
                    action = agents[role].get_action(observation, explore=False)
                else:
                    action = env.action_space(agent_name).sample()

                if agent_name == env.possible_agents[0]:
                    step += 1

            env.step(action)

            if agent_name == last_agent:
                env.render()
    except SystemExit:
        pass

    print(f"\nEpisode finished after {step} steps.", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Visualize a trained agent episode")
    p.add_argument("--predator_agent", type=str, default="random", choices=["random", "ppo", "mappo"])
    p.add_argument("--prey_agent", type=str, default="random", choices=["random", "ppo", "mappo"])
    p.add_argument("--load_predator_weights", type=str, default=None)
    p.add_argument("--load_prey_weights", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--width", type=int, default=ENV_CONFIG["width"])
    p.add_argument("--height", type=int, default=ENV_CONFIG["height"])
    p.add_argument("--prey_count", type=int, default=3)
    p.add_argument("--predator_count", type=int, default=2)
    p.add_argument("--catch_radius", type=int, default=ENV_CONFIG["catch_radius"])
    p.add_argument("--no_fov", action="store_true", help="disable FOV cone (see all N nearest)")
    p.add_argument("--predator_speed_mult", type=float, default=1.0,
                   help="multiplier on predator_max_velocity (e.g. 0.5 halves speed)")
    p.add_argument("--draw_forces", action="store_true")
    p.add_argument("--draw_cones", action="store_true")
    p.add_argument("--draw_hitboxes", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
