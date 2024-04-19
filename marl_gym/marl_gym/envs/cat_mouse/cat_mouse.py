import gym
from gym import spaces
from utils import *
import numpy as np

class CatMouse(gym.Env):

    metadata = {'render_modes': ['human']}
    
    def __init__(self, area_size=(1,1), n_agents=2, n_mice=4, ma=False, obs_radius=..., render_mode=None):
        
        self.area_size = area_size
        self.n_agents = n_agents
        self.n_mice = n_mice
        
        # need to figure out spaces
        self.observation_space = MultiAgentObservationSpace([spaces.Box(low=0, high=1, shape=(1,1), dtype=np.float32) for _ in range(n_agents + n_mice)])
        # MAOS is just wrapper for list so we can use sample
        self.action_space = MultiAgentActionSpace([spaces.Box(low=0, high=1, dtype=np.float32) for _ in range(n_agents)])

        self.render_mode = render_mode

        self.agents = ...

    # turn env state into observation state
    def _get_obs(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass
    
    # close rendering
    def close(self):
        pass
