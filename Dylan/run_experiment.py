
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from Core_Scripts.env import make_env, get_space_dims, get_role, get_agents_by_role, get_global_obs_dim, ENV_CONFIG
from Core_Scripts.logger import ExperimentLogger
from Agents.random_policy import RandomAgent


def _get_ppo_class():
    from Agents.ppo_agent import PPOAgent
    return PPOAgent

def _get_mappo_class():
    from Agents.mappo_agent import MAPPOAgent
    return MAPPOAgent

AGENT_REGISTRY = {
    "random": RandomAgent,
    "ppo": _get_ppo_class,
    "mappo": _get_mappo_class,
}


def build_agent(agent_type, obs_dim, act_dim, global_obs_dim=None, entropy_coef=None):
    """Instantiate an agent by its registry name."""
    entry = AGENT_REGISTRY[agent_type]
    if agent_type == "random":
        return entry(obs_dim, act_dim)
    print(f"  Loading {agent_type} class...", flush=True)
    cls = entry()
    print(f"  Class loaded. Creating instance...", flush=True)
    kwargs = {}
    if entropy_coef is not None:
        kwargs["entropy_coef"] = entropy_coef
    if agent_type == "mappo":
        agent = cls(obs_dim, act_dim, global_obs_dim=global_obs_dim, **kwargs)
    else:
        agent = cls(obs_dim, act_dim, **kwargs)
    print(f"  Instance created.", flush=True)
    return agent


# ── Training loop ─────────────────────────────────────────────────────────

def run(args):
    print("Initializing environment...", flush=True)
    overrides = _parse_env_overrides(args)
    env = make_env(render=args.render, **overrides)
    print("Environment created.", flush=True)

    cfg = {**ENV_CONFIG, **overrides}
    dims = get_space_dims(env)

    pred_obs, pred_act = dims["predator"]
    prey_obs, prey_act = dims["prey"]
    global_obs_dim = get_global_obs_dim(env)
    sharing = not args.no_sharing

    # ── Build agent_map: agent_name → agent instance ──────────────────
    agent_map = {}

    ent_coef = getattr(args, "entropy_coef", None)

    if sharing:
        print(f"Building shared predator agent ({args.predator_agent})...", flush=True)
        pred_agent = build_agent(args.predator_agent, pred_obs, pred_act,
                                 global_obs_dim=global_obs_dim, entropy_coef=ent_coef)
        print(f"Building shared prey agent ({args.prey_agent})...", flush=True)
        prey_agent = build_agent(args.prey_agent, prey_obs, prey_act,
                                 global_obs_dim=global_obs_dim, entropy_coef=ent_coef)
        for name in env.possible_agents:
            agent_map[name] = pred_agent if get_role(name) == "predator" else prey_agent
    else:
        for name in env.possible_agents:
            role = get_role(name)
            atype = args.predator_agent if role == "predator" else args.prey_agent
            obs_d, act_d = dims[role]
            print(f"Building independent agent for {name} ({atype})...", flush=True)
            agent_map[name] = build_agent(atype, obs_d, act_d,
                                          global_obs_dim=global_obs_dim, entropy_coef=ent_coef)

    print("Agents ready.", flush=True)

    # ── Load weights ──────────────────────────────────────────────────
    if args.load_predator_weights:
        print(f"Loading predator weights: {args.load_predator_weights}", flush=True)
        for name in env.possible_agents:
            if get_role(name) == "predator":
                agent_map[name].load(args.load_predator_weights)
                if sharing:
                    break
    if args.load_prey_weights:
        print(f"Loading prey weights: {args.load_prey_weights}", flush=True)
        for name in env.possible_agents:
            if get_role(name) == "prey":
                agent_map[name].load(args.load_prey_weights)
                if sharing:
                    break

    # Deduplicated set of unique agent instances (for updates / step_update)
    unique_agents = {}
    for name, ag in agent_map.items():
        if id(ag) not in unique_agents:
            unique_agents[id(ag)] = (name, ag)

    run_name = f"{args.predator_agent}_vs_{args.prey_agent}_{int(time.time())}"
    logger = ExperimentLogger(
        log_dir=os.path.join(args.log_dir, run_name),
        use_tensorboard=args.tensorboard,
    )

    print("=" * 60, flush=True)
    print(f"  EXPERIMENT STARTED", flush=True)
    print(f"  Run name       : {run_name}", flush=True)
    print(f"  Predator agent : {args.predator_agent}  (obs={pred_obs}, act={pred_act})", flush=True)
    print(f"  Prey agent     : {args.prey_agent}  (obs={prey_obs}, act={prey_act})", flush=True)
    print(f"  Param sharing  : {sharing}", flush=True)
    print(f"  Unique agents  : {len(unique_agents)}", flush=True)
    print(f"  Episodes       : {args.episodes}", flush=True)
    print(f"  Agents         : {env.possible_agents}", flush=True)
    print(f"  max_steps      : {cfg['max_time_steps']}", flush=True)
    print(f"  catch_radius   : {cfg['catch_radius']}", flush=True)
    print(f"  fov_enabled    : {cfg['fov_enabled']}", flush=True)
    print(f"  catch_bonus    : {cfg['predator_catch_bonus']}", flush=True)
    print(f"  prox_reward    : {cfg['predator_proximity_reward']}", flush=True)
    print(f"  prox_penalty   : {cfg['prey_proximity_penalty']}", flush=True)
    print(f"  step_penalty   : {cfg['predator_step_penalty']}", flush=True)
    print(f"  no_catch_pen   : {cfg.get('predator_no_catch_step_penalty', 0.0)}", flush=True)
    if ent_coef is not None:
        print(f"  entropy_coef   : {ent_coef}", flush=True)
    if args.load_predator_weights:
        print(f"  Pred weights   : {args.load_predator_weights}", flush=True)
    if args.load_prey_weights:
        print(f"  Prey weights   : {args.load_prey_weights}", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    t_start = time.time()

    self_size = 5
    obs_size = 6
    dist_offset = 2

    def _closest_prey_dist(o):
        """Extract closest prey distance from a predator observation."""
        start = self_size + cfg["predator_observe_count"] * obs_size
        dists = [
            o[start + i * obs_size + dist_offset]
            for i in range(cfg["prey_observe_count"])
            if start + i * obs_size + dist_offset < len(o)
        ]
        nonzero = [d for d in dists if d > 0]
        return min(nonzero) if nonzero else 1.0

    def _closest_pred_dist(o):
        """Extract closest predator distance from a prey observation."""
        start = self_size
        dists = [
            o[start + i * obs_size + dist_offset]
            for i in range(cfg["predator_observe_count"])
            if start + i * obs_size + dist_offset < len(o)
        ]
        nonzero = [d for d in dists if d > 0]
        return min(nonzero) if nonzero else 1.0

    prev_pred_dists = {}

    for episode in range(1, args.episodes + 1):
        obs, infos = env.reset(seed=args.seed)
        team_rewards = {"predator": 0.0, "prey": 0.0}
        episode_steps = 0

        prev_pred_dists.clear()
        for name in env.possible_agents:
            if get_role(name) == "predator" and name in obs:
                prev_pred_dists[name] = _closest_prey_dist(obs[name])

        while env.agents:
            global_obs_flat = np.concatenate(
                [obs[n] for n in env.possible_agents if n in obs and get_role(n) == "predator"]
            )

            actions = {}
            for agent_name in env.agents:
                actions[agent_name] = agent_map[agent_name].get_action(
                    obs[agent_name], global_obs=global_obs_flat,
                )

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            if args.render:
                env.render()

            # ── reward shaping ────────────────────────────────────────
            # Count prey deaths in this step.
            # NOTE: reward magnitudes depend on ENV_CONFIG (e.g. prey_punishment can be 1),
            # so we must NOT use a hard-coded threshold like -500.
            # In this env setup, prey rewards are non-negative except on death.
            prey_deaths = sum(
                1 for name, r in rewards.items()
                if name.startswith("prey") and r < 0
            )
            if prey_deaths > 0:
                for name in rewards:
                    if name.startswith("predator"):
                        rewards[name] += cfg["predator_catch_bonus"] * prey_deaths

            nc_pen = cfg.get("predator_no_catch_step_penalty", 0.0)
            if nc_pen > 0 and prey_deaths == 0:
                for name in rewards:
                    if name.startswith("predator"):
                        rewards[name] -= nc_pen

            step_pen = cfg["predator_step_penalty"]
            if step_pen > 0 and next_obs:
                for name in rewards:
                    if name.startswith("predator"):
                        o = next_obs.get(name)
                        if o is not None:
                            speed = o[4]  # self-state index 4 = normalized speed
                            rewards[name] -= step_pen * speed

            prox_pred = cfg["predator_proximity_reward"]
            prox_prey = cfg["prey_proximity_penalty"]
            if (prox_pred > 0 or prox_prey > 0) and next_obs:
                for name in list(rewards.keys()):
                    o = next_obs.get(name)
                    if o is None:
                        continue
                    role = get_role(name)
                    if role == "predator" and prox_pred > 0:
                        curr_dist = _closest_prey_dist(o)
                        prev_dist = prev_pred_dists.get(name, curr_dist)
                        rewards[name] += prox_pred * (prev_dist - curr_dist)
                        prev_pred_dists[name] = curr_dist
                    elif role == "prey" and prox_prey > 0:
                        curr_dist = _closest_pred_dist(o)
                        rewards[name] -= prox_prey * (1.0 - curr_dist)

            for agent_name in actions:
                role = get_role(agent_name)
                done = terminations.get(agent_name, True) or truncations.get(agent_name, True)
                next_o = next_obs.get(agent_name, np.zeros(dims[role][0]))
                agent_map[agent_name].store_transition(
                    obs[agent_name], actions[agent_name],
                    rewards.get(agent_name, 0.0), next_o, done,
                    global_obs=global_obs_flat,
                    buffer_id=agent_name,
                )
                team_rewards[role] += rewards.get(agent_name, 0.0)

            for _, ag in unique_agents.values():
                ag.step_update()

            obs = next_obs
            episode_steps += 1

        # Per-episode updates — call once per unique agent instance
        update_info = {}
        for label, ag in unique_agents.values():
            info = ag.episode_update()
            for k, v in info.items():
                update_info[f"{label}_{k}"] = v

        metrics = {
            "episode_length": episode_steps,
            "predator_reward": team_rewards["predator"],
            "prey_reward": team_rewards["prey"],
            **update_info,
        }
        logger.log_episode(episode, metrics)

        if episode == 1:
            print(f"[Ep {episode:>5d}]  steps={episode_steps:<5d}  "
                  f"pred_r={team_rewards['predator']:>8.1f}  "
                  f"prey_r={team_rewards['prey']:>8.1f}  "
                  f"(first episode complete)", flush=True)
        elif episode % args.print_every == 0:
            elapsed = time.time() - t_start
            eps_per_sec = episode / elapsed
            eta = (args.episodes - episode) / eps_per_sec if eps_per_sec > 0 else 0
            loss_str = ""
            if "predator_pg_loss" in update_info:
                loss_str += f"  pred_loss={update_info['predator_pg_loss']:.4f}"
            elif "predator_0_pg_loss" in update_info:
                loss_str += f"  pred0_loss={update_info['predator_0_pg_loss']:.4f}"
            if "prey_pg_loss" in update_info:
                loss_str += f"  prey_loss={update_info['prey_pg_loss']:.4f}"
            elif "prey_0_pg_loss" in update_info:
                loss_str += f"  prey0_loss={update_info['prey_0_pg_loss']:.4f}"
            print(f"[Ep {episode:>5d}/{args.episodes}]  steps={episode_steps:<5d}  "
                  f"pred_r={team_rewards['predator']:>8.1f}  "
                  f"prey_r={team_rewards['prey']:>8.1f}"
                  f"{loss_str}  "
                  f"({eps_per_sec:.1f} ep/s, ETA {eta:.0f}s)", flush=True)

    # ── Save weights ──────────────────────────────────────────────────
    if args.save_weights:
        weight_dir = os.path.join(args.log_dir, run_name, "weights")
        os.makedirs(weight_dir, exist_ok=True)
        if sharing:
            for name in env.possible_agents:
                role = get_role(name)
                path = os.path.join(weight_dir, f"{role}.pt")
                if not os.path.exists(path):
                    agent_map[name].save(path)
        else:
            for name in env.possible_agents:
                agent_map[name].save(os.path.join(weight_dir, f"{name}.pt"))
        print(f"\nWeights saved to {weight_dir}", flush=True)

    logger.close()
    elapsed = time.time() - t_start
    print(flush=True)
    print("=" * 60, flush=True)
    print(f"  EXPERIMENT COMPLETE — {args.episodes} episodes in {elapsed:.1f}s", flush=True)
    print("=" * 60, flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_env_overrides(args):
    """Map CLI flags to ENV_CONFIG overrides."""
    overrides = {}
    mapping = {
        "width": "width",
        "height": "height",
        "prey_count": "prey_count",
        "predator_count": "predator_count",
        "max_steps": "max_time_steps",
        "catch_radius": "catch_radius",
        "fov_enabled": "fov_enabled",
        "predator_view_distance": "predator_view_distance",
        "prey_view_distance": "prey_view_distance",
        "predator_max_velocity": "predator_max_velocity",
        "predator_catch_bonus": "predator_catch_bonus",
        "predator_proximity_reward": "predator_proximity_reward",
        "prey_proximity_penalty": "prey_proximity_penalty",
        "predator_step_penalty": "predator_step_penalty",
        "predator_no_catch_step_penalty": "predator_no_catch_step_penalty",
    }
    for cli_key, cfg_key in mapping.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            overrides[cfg_key] = val
    if args.no_fov:
        overrides["fov_enabled"] = False
        world_diag = int((ENV_CONFIG["width"] ** 2 + ENV_CONFIG["height"] ** 2) ** 0.5) + 50
        overrides.setdefault("predator_view_distance", world_diag)
        overrides.setdefault("prey_view_distance", world_diag)
    return overrides


def parse_args():
    p = argparse.ArgumentParser(description="Aquarium emergent-behaviour experiment runner")

    # ── training ──
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--predator_agent", type=str, default="random",
                   choices=list(AGENT_REGISTRY.keys()))
    p.add_argument("--prey_agent", type=str, default="random",
                   choices=list(AGENT_REGISTRY.keys()))
    p.add_argument("--render", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log_dir", type=str, default="results")
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--save_weights", action="store_true")
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--no_sharing", action="store_true",
                   help="independent policy per agent (no parameter sharing)")
    p.add_argument("--entropy_coef", type=float, default=None,
                   help="entropy bonus coefficient (default: agent default, 0.01)")

    # ── weight loading ──
    p.add_argument("--load_predator_weights", type=str, default=None)
    p.add_argument("--load_prey_weights", type=str, default=None)

    # ── environment overrides ──
    p.add_argument("--width", type=int, default=None, help="tank width (pixels)")
    p.add_argument("--height", type=int, default=None, help="tank height (pixels)")
    p.add_argument("--prey_count", type=int, default=None)
    p.add_argument("--predator_count", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--catch_radius", type=int, default=None)
    p.add_argument("--no_fov", action="store_true", help="disable FOV cone (see all N nearest)")
    p.add_argument("--predator_view_distance", type=int, default=None)
    p.add_argument("--prey_view_distance", type=int, default=None)
    p.add_argument("--predator_max_velocity", type=float, default=None,
                   help="override predator max speed (default from ENV_CONFIG)")

    # ── reward shaping overrides ──
    p.add_argument("--predator_catch_bonus", type=float, default=None)
    p.add_argument("--predator_proximity_reward", type=float, default=None,
                   help="per-step bonus for predator approaching prey (0 = off)")
    p.add_argument("--prey_proximity_penalty", type=float, default=None,
                   help="per-step penalty for prey being near predator (0 = off)")
    p.add_argument("--predator_step_penalty", type=float, default=None,
                   help="per-step cost for predators, multiplied by normalized speed (movement tax)")
    p.add_argument("--predator_no_catch_step_penalty", type=float, default=None,
                   help="subtract this much from predator reward each step with no prey death (catch)")

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
