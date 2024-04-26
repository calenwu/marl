import gym
from gym import spaces
import numpy as np
import math
import pygame

class CatMouse(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 4}
    
    def __init__(self, area_size=(1,1), n_agents=2, n_mice=4, step_size=0.05, entity_size=0.05, step_cost = -0.1, render_mode=None, window_size = 250):
        
        self.area_size = area_size
        self.n_agents = n_agents
        self.n_mice = n_mice
        self.step_size = step_size
        self.entity_size = entity_size
        self.step_cost = step_cost
        self.window = None
        self.clock = None
        self.window_size = window_size
        
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=1, shape=(self.n_agents,2), dtype=np.float32),spaces.Box(low=0, high=1, shape=(self.n_mice,3), dtype=np.float32)))
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_agents,), dtype=np.float32)

        self.render_mode = render_mode

        self.reset()

    # turn env state into observation state
    def _get_obs(self):
        return (self.agents, self.mice)

    def reset(self):
        self.agents = [(np.random.uniform(),np.random.uniform()) for _ in range(self.n_agents)]
        self.mice = [(np.random.uniform(),np.random.uniform(), 0) for _ in range(self.n_mice)]
        return self._get_obs()

    # action here should be 
    def step(self, action):
        assert len(action) == self.n_agents, "action length should be number of agents"

        next_state = None
        reward = 0
        terminated = False
        info = {}

        # move each agent according to action
        self._move_agents(action)
        # assume uniform random movement of mouse rn, move each mouse
        self._move_mice()

        self._check_caught()

        next_state = self._get_obs()

        reward = self._calc_reward()

        terminated = all([caught for _,_,caught in self.mice])

        return next_state, reward, terminated, info

    def _calc_dists(self, list1, list2):
        t1 = np.array(list1)[:,:2]
        t2 = np.array(list2)[:,:2]
        t1 = t1[:,np.newaxis,:]
        t2 = t2[np.newaxis,:]

        x = t1 - t2
        x = x**2
        x = x[:,:,0] + x[:,:,1]
        x = np.sqrt(x)
        return x
    
    def _check_caught(self):
        self.agent_mice_dists =  self._calc_dists(self.agents,self.mice)
        for i in range(self.n_mice):
            if not self.mice[i][2]:
                for j in range(self.n_agents):
                    if self.agent_mice_dists[j][i] < 2 * self.entity_size:
                        self.mice[i] = (self.mice[i][0],self.mice[i][1],1)
                        break
    
    def _calc_reward(self):
        reward = self.step_cost
        for i in range(self.n_agents):
            min_dist = 2
            for j in range(self.n_mice):
                if not self.mice[j][2]:
                    min_dist = min(min_dist, self.agent_mice_dists[i][j])
            reward -= min_dist
        return reward

    def _move_agents(self, action):
        # move each agent according to action
        for i,a in enumerate(action):
            direction = 2* np.pi * a
            move_x = self.step_size * math.cos(direction)
            move_y = self.step_size * math.sin(direction)
            cur_x, cur_y = self.agents[i]
            self.agents[i] = (min(max(0,cur_x + move_x),1), max(min(1, cur_y + move_y),0)) # clip between 0 and 1
    
    def _move_mice(self):
        # assume uniform random movement of mouse rn, move each mouse
        for i in range(self.n_mice):
            cur_x, cur_y, caught = self.mice[i]
            if caught:
                continue

            direction = 2 * np.pi * np.random.uniform()
            move_x = self.step_size * math.cos(direction)
            move_y = self.step_size * math.sin(direction)
            
            self.mice[i] = (min(max(0,cur_x + move_x),1), max(min(1, cur_y + move_y), 0), caught)

    def render(self):
        self._render_frame()
        
    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))

        canvas.fill((255, 255, 255))

        for a in self.agents:
            x,y,*_ = a
            x *= self.window_size
            y *= self.window_size
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (x,y),
                self.entity_size*self.window_size,
            )
        
        for m in self.mice:
            if m[2]:
                continue
            x,y,*_ = m
            x *= self.window_size
            y *= self.window_size
            pygame.draw.circle(
                canvas,
                (255, 0, 0),
                (x,y),
                self.entity_size*self.window_size,
            )
            
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])
    
    # close rendering
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
