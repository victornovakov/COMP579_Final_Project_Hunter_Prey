"""
Shared experience-storage classes.

ReplayBuffer  – off-policy (MADDPG)
RolloutBuffer – on-policy  (PPO)
"""

import numpy as np
from collections import deque
import random as _random


# ═════════════════════════════════════════════════════════════════════════════
# Off-policy replay buffer (MADDPG)
# ═════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Fixed-size ring buffer that stores (obs, action, reward, next_obs, done)
    tuples and returns uniformly sampled mini-batches as numpy arrays.
    """

    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.int64),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
        ))

    def sample(self, batch_size):
        batch = _random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return {
            "obs": np.stack(obs),
            "actions": np.stack(actions),
            "rewards": np.stack(rewards),
            "next_obs": np.stack(next_obs),
            "dones": np.stack(dones),
        }

    def __len__(self):
        return len(self.buffer)


# ═════════════════════════════════════════════════════════════════════════════
# On-policy rollout buffer (PPO)
# ═════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """
    Collects a full rollout (one epoch of environment steps), then yields it
    once for PPO updates before being cleared.

    Stores: obs, action, log_prob, reward, done, value.
    After the rollout call `compute_returns()` to fill in discounted returns
    and advantages (GAE-lambda).
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.global_obs = []
        self.returns = None
        self.advantages = None

    def push(self, obs, action, log_prob, reward, done, value, global_obs=None):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.int64))
        self.log_probs.append(np.asarray(log_prob, dtype=np.float32))
        self.rewards.append(np.asarray(reward, dtype=np.float32))
        self.dones.append(np.asarray(done, dtype=np.float32))
        self.values.append(np.asarray(value, dtype=np.float32))
        if global_obs is not None:
            self.global_obs.append(np.asarray(global_obs, dtype=np.float32))

    def compute_returns(self, last_value=0.0, gamma=0.99, lam=0.95):
        """Compute GAE-lambda advantages and discounted returns in-place."""
        n = len(self.rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        values = np.array(self.values, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            next_non_terminal = 1.0 - (dones[t] if t == n - 1 else dones[t])
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + values

    def get_batch(self):
        """Return the full rollout as a dict of numpy arrays, then clear."""
        assert self.returns is not None, "Call compute_returns() first"
        batch = {
            "obs": np.stack(self.obs),
            "actions": np.stack(self.actions),
            "log_probs": np.stack(self.log_probs),
            "returns": self.returns,
            "advantages": self.advantages,
            "values": np.array(self.values, dtype=np.float32),
        }
        if self.global_obs:
            batch["global_obs"] = np.stack(self.global_obs)
        self.clear()
        return batch

    def __len__(self):
        return len(self.rewards)
