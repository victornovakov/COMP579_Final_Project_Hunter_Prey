"""
MAPPO (Multi-Agent PPO) agent for discrete action spaces.

Centralized Training, Decentralized Execution (CTDE):
  - Actor  : local observation  → action  (same as PPO)
  - Critic : ALL agents' observations concatenated → value  (centralized)

The only difference from PPO is the critic input.  At test time the critic
is not used, so execution is fully decentralized.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from collections import deque

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Core_Scripts.buffer import RolloutBuffer


# ── Network helpers ───────────────────────────────────────────────────────

def layer_init(layer, gain=np.sqrt(2), bias_const=0.0):
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """Decentralized policy: local obs → action logits."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), gain=0.01),
        )

    def forward(self, x):
        return self.net(x)


class CentralizedCritic(nn.Module):
    """Value network that takes the concatenation of ALL agents' observations."""

    def __init__(self, global_obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(global_obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), gain=1.0),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── MAPPO Agent ───────────────────────────────────────────────────────────

class MAPPOAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        global_obs_dim,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=4,
        mini_batch_size=64,
        device=None,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.global_obs_dim = global_obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.device = device or "cpu"

        self.actor = Actor(obs_dim, act_dim).to(self.device)
        self.critic = CentralizedCritic(global_obs_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr, eps=1e-5,
        )

        self.buffers = {}
        self._cache = deque()

    # ── Action selection ──────────────────────────────────────────────

    def get_action(self, observation, explore=True, **kwargs):
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.actor(obs_t)

            global_obs = kwargs.get("global_obs")
            if global_obs is not None:
                global_t = torch.as_tensor(global_obs, dtype=torch.float32, device=self.device)
                value = self.critic(global_t).cpu().item()
            else:
                value = 0.0

        dist = Categorical(logits=logits)
        if explore:
            action = dist.sample()
        else:
            action = logits.argmax()

        self._cache.append((
            dist.log_prob(action).cpu().item(),
            value,
            global_obs,
        ))
        return action.cpu().item()

    # ── Transition storage ────────────────────────────────────────────

    def store_transition(self, obs, action, reward, next_obs, done, **kwargs):
        log_prob, value, global_obs_flat = self._cache.popleft()
        bid = kwargs.get("buffer_id", "default")
        if bid not in self.buffers:
            self.buffers[bid] = RolloutBuffer()
        self.buffers[bid].push(obs, action, log_prob, reward, done, value,
                               global_obs=global_obs_flat)

    # ── Updates ───────────────────────────────────────────────────────

    def step_update(self):
        return {}

    def episode_update(self):
        active = [b for b in self.buffers.values() if len(b) > 0]
        if not active:
            return {}

        batches = []
        for buf in active:
            buf.compute_returns(last_value=0.0, gamma=self.gamma, lam=self.gae_lambda)
            batches.append(buf.get_batch())

        obs = torch.as_tensor(np.concatenate([b["obs"] for b in batches]), device=self.device)
        actions = torch.as_tensor(np.concatenate([b["actions"] for b in batches]), device=self.device)
        old_log_probs = torch.as_tensor(np.concatenate([b["log_probs"] for b in batches]), device=self.device)
        returns = torch.as_tensor(np.concatenate([b["returns"] for b in batches]), device=self.device)
        advantages = torch.as_tensor(np.concatenate([b["advantages"] for b in batches]), device=self.device)
        global_obs = torch.as_tensor(np.concatenate([b["global_obs"] for b in batches]), device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_global_obs = global_obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                # Actor uses LOCAL obs only
                logits = self.actor(mb_obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - mb_old_lp).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps,
                                    1.0 + self.clip_eps) * mb_advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                # Critic uses GLOBAL obs (centralized)
                new_values = self.critic(mb_global_obs)
                v_loss = 0.5 * (new_values - mb_returns).pow(2).mean()

                loss = pg_loss + self.value_coef * v_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        return {
            "pg_loss": total_pg_loss / max(num_updates, 1),
            "v_loss": total_v_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, filepath):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.