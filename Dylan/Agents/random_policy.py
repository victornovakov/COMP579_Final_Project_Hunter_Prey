import numpy as np


class RandomAgent:
    """
    A baseline agent that takes completely random discrete actions.
    Matches the interface of PPOAgent and MADDPGAgent.
    """

    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def get_action(self, observation, explore=True, **kwargs):
        return np.random.randint(0, self.act_dim)

    #doesnt do anything because random agent doesnt need to store transitions, included so that main loop doesnt crash!!!

    def store_transition(self, obs, action, reward, next_obs, done, **kwargs):
        pass

    def step_update(self):
        return {}

    def episode_update(self):
        return {}

    def save(self, filepath):
        pass

    def load(self, filepath):
        pass
