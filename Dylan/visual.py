"""
Visualize a single episode using trained weights.

Usage:
    # PPO predator vs random prey
    python visual.py --predator_agent ppo --load_predator_weights results/test_ppo3/.../weights/predator.pt

    # PPO pred vs PPO prey
    python visual.py --predator_agent ppo --prey_agent ppo \
        --load_predator_weights path/to/predator.pt \
        --load_prey_weights path/to/prey.pt

    # Random vs random (no weights needed)
    python visual.py
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from marl_aquarium import aquarium_v0
import numpy as np


def get_role(name):
    return "predator" if name.startswith("predator") else "prey"


def run(args):
    env = aquarium_v0.env(
        prey_count=args.prey_count,
        predator_count=args.predator_count,
        catch_radius=120,
        predator_view_distance=300,
        max_time_steps=args.max_steps,
        action_count=16,
        predator_observe_count=1,
        prey_observe_count=3,
        keep_prey_count_constant=True,
        predator_max_age=10000,
        draw_force_vectors=args.draw_forces,
        draw_view_cones=args.draw_cones,
        draw_hit_boxes=args.draw_hitboxes,
    )

    from Core_Scripts.env import get_space_dims as _get_dims

    temp_env = aquarium_v0.parallel_env(
        prey_count=args.prey_count,
        predator_count=args.predator_count,
        action_count=16,
        predator_observe_count=1,
        prey_observe_count=3,
        render_mode="rgb_array",
    )
    dims = _get_dims(temp_env)
    global_obs_dim = sum(
        temp_env.observation_space(a).shape[0] for a in temp_env.possible_agents
    )
    temp_env.close()

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
        env.render()

    print(f"\nEpisode finished after {step