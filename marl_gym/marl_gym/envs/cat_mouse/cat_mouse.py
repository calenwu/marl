import gym
from gym import spaces
from utils import *
import numpy as np
import math

class CatMouse(gym.Env):

    metadata = {'render_modes': ['human']}
    
    def __init__(self, area_size=(1,1), n_agents=2, n_mice=4, step_size=0.05, entity_size=0.05, render_mode=None):
        
        self.area_size = area_size
        self.n_agents = n_agents
        self.n_mice = n_mice
        self.step_size = step_size
        self.entity_size = entity_size
        
        # need to figure out spaces
        self.observation_space = MultiAgentObservationSpace([spaces.Box(low=0, high=1, shape=(1,1), dtype=np.float32) for _ in range(n_agents + n_mice)])
        # MAOS is just wrapper for list so we can use sample
        self.action_space = MultiAgentActionSpace([spaces.Box(low=0, high=1, dtype=np.float32) for _ in range(n_agents)])

        self.render_mode = render_mode

        self.reset()

    # turn env state into observation state
    def _get_obs(self):
        pass

    def reset(self):
        self.agents = [(np.random.uniform(),np.random.uniform()) for _ in range(self.n_agents)]
        self.mice = [(np.random.uniform(),np.random.uniform(), 0) for _ in range(self.n_mice)]

    # action here should be 
    def step(self, action):
        assert len(action) == self.n_agents, "action length should be number of agents"

        next_state = None
        reward = 0
        terminated = False
        info = {}

        # move each agent according to action
        for i,a in enumerate(action):
            direction = 2* np.pi * a
            move_x = self.step_size * math.cos(direction)
            move_y = self.step_size * math.sin(direction)
            cur_x, cur_y = self.agents[i]
            self.agents[i] = (min(max(0,cur_x + move_x),1), max(min(1, cur_y + move_y),0)) # clip between 0 and 1

        # assume uniform random movement of mouse rn, move each mouse
        for i in range(self.n_mice):
            
            cur_x, cur_y, caught = self.mice[i]
            if caught:
                continue

            direction = 2 * np.pi * np.random.uniform()
            move_x = self.step_size * math.cos(direction)
            move_y = self.step_size * math.sin(direction)
            
            self.mice[i] = (min(max(0,cur_x + move_x),1), max(min(1, cur_y + move_y), caught))

            # check for current mouse if caught
            for j in range(self.n_agents):
                if self.in_range(self.agents[j], self.mice[i]):
                    self.mice[i][2] = True

        next_state = (self.agents, self.mice)

        terminated = all([caught for _,_,caught in self.mice])

        return next_state, reward, terminated, info

    def in_range(self, agent, mouse):
        agent_x, agent_y = agent
        mouse_x, mouse_y, _ = mouse
        return math.sqrt((agent_x-mouse_x)**2 + (agent_y-mouse_y)**2) < 2 * self.entity_size


    def render(self):
        pass
    
    # close rendering
    def close(self):
        pass
