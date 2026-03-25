"""
PPO (Proximal Policy Optimization) agent for discrete action spaces.
From-scratch PyTorch implementation, CleanRL-style.

Supports parameter sharing: multiple environment agents (e.g. all prey)
can share a single PPOAgent instance.
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
    """Xavier uniform init (avoids SVD which conflicts with pygame on macOS)."""
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
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


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), gain=1.0),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── PPO Agent ─────────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
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
        self.critic = Critic(obs_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr, eps=1e-5,
        )

        self.buffer = RolloutBuffer()
        self._cache = deque()

    # ── Action selection ──────────────────────────────────────────────

    def get_action(self, observation, explore=True, **kwargs):
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.actor(obs_t)
            value = self.critic(obs_t)

        dist = Categorical(logits=logits)
        if explore:
            action = dist.sample()
        else:
            action = logits.argmax()

        self._cache.append((
            dist.log_prob(action).cpu().item(),
            value.cpu().item(),
        ))
        return action.cpu().item()

    # ── Transition storage ────────────────────────────────────────────

    def store_transition(self, obs, action, reward, next_obs, done, **kwargs):
        log_prob, value = self._cache.popleft()
        self.buffer.push(obs, action, log_prob, reward, done, value)

    # ── Updates ───────────────────────────────────────────────────────

    def step_update(self):
        return {}

    def episode_update(self):
        if len(self.buffer) == 0:
            return {}

        self.buffer.compute_returns(
            last_value=0.0,
            gamma=self.gamma,
            lam=self.gae_lambda,
        )
        batch = self.buffer.get_batch()

        obs = torch.as_tensor(batch["obs"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], device=self.device)
        returns = torch.as_tensor(batch["returns"], device=self.device)
        advantages = torch.as_tensor(batch["advantages"], device=self.device)

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
                mb_actions = actions[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                logits = self.actor(mb_obs)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - mb_old_lp).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                new_values = self.critic(mb_obs)
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
            "optimizer": self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        ckpt = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
