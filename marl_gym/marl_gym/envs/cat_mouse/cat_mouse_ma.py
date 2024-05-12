import gym
from gym import spaces
from marl_gym.marl_gym.envs.utils.utils import * # maybe need to change if made to package
import numpy as np
import math
import pygame

# multi-agent version
class CatMouseMA(gym.Env):

    metadata = {'render_modes': ['human'], "render_fps": 4}
    
    def __init__(self, area_size=(1,1), n_agents=2, n_mice=4, step_size=0.05, 
                 entity_size=0.05, observation_radius=0.2, communication_radius=0.2, 
                 step_cost = -0.1, render_mode='human', window_size=250):
        
        self.area_size = area_size
        self.n_agents = n_agents
        self.n_mice = n_mice
        self.step_size = step_size
        self.entity_size = entity_size
        self.observation_radius = observation_radius
        self.communication_radius = communication_radius
        self.step_cost = step_cost
        self.window = None
        self.clock = None
        self.window_size = window_size
        
        # need to figure out spaces
        self.observation_space = MultiAgentObservationSpace([spaces.Tuple((spaces.Box(0,1,shape=(self.n_agents,3)),spaces.Box(0,1,shape=(self.n_mice,3)))) for _ in range(self.n_agents + self.n_mice)])
        # MAOS is just wrapper for list so we can use sample
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_agents,), dtype=np.float32)

        self.render_mode = render_mode
        
        # for each agent, which mice agent knows are caught already
        self.agent_mice_list = [[] for _ in range(self.n_agents)]

        self.reset()

    def _get_global_obs(self):
        obs_agents = []
        obs_mice = []
        for i in range(self.n_agents):
            obs_agents.append((self.agents[i][0],self.agents[i][1]))

        for i in range(self.n_mice):
            mouse_i_caught = 0
            for j in range(self.n_agents):
                obs_caught = i in self.agent_mice_list[j]
                if obs_caught == 1:
                    mouse_i_caught = 1
                    break
            obs_mice.append((self.mice[i][0],self.mice[i][1], mouse_i_caught))
        return (obs_agents, obs_mice)
    # turn env state into observation state
    # observation state for agent i: (agents, mice), where agents/mice are arrays of 3-tuples, first 2 are position, 3rd is flag
    # for agent flag whether it's the current agent, for mice whether current agent saw it getting caught
    def _get_obs(self):

        agent_obs = []
        communication = []

        for i in range(self.n_agents):
            # parse agent local observation from global state, if entity not seen then pos is (-1,-1)
            cur_agent_obs = []
            cur_in_comm_range = []
            for j in range(self.n_agents):
                if i == j:
                    cur_agent_obs.append((self.agents[j][0],self.agents[j][1], 1)) # set to one to let agent know its own position
                elif self.agent_agent_obs_matrix[i][j]:
                    cur_agent_obs.append((self.agents[j][0],self.agents[j][1], 0))
                else:
                    cur_agent_obs.append((-1, -1, 0)) 
                if self.agent_agent_comm_matrix[i][j]:
                    cur_in_comm_range.append(j)

            cur_mice_obs = []
            for j in range(self.n_mice):
                obs_caught = j in self.agent_mice_list[i] # set mouse caught flag to 1 to indicate it observed mouse getting caught
                if self.agent_mouse_obs_matrix[i][j]:
                    cur_mice_obs.append((self.mice[j][0], self.mice[j][1], obs_caught))
                else:
                    cur_mice_obs.append((-1, -1, obs_caught))
        
            agent_obs.append((cur_agent_obs,cur_mice_obs))
            communication.append(cur_in_comm_range)

        return agent_obs, communication

    def reset(self):
        self.agents = [(np.random.uniform(),np.random.uniform()) for _ in range(self.n_agents)]
        self.mice = [(np.random.uniform(),np.random.uniform(), 0) for _ in range(self.n_mice)]
        self.agent_agent_obs_matrix = self._calc_in_range_matrix(self.agents, self.agents, self.observation_radius)
        self.agent_mouse_obs_matrix = self._calc_in_range_matrix(self.agents, self.mice, self.observation_radius)
        self.agent_agent_comm_matrix = self._calc_in_range_matrix(self.agents, self.agents, self.communication_radius)
        return self._get_obs()

    # action here should be 
    def step(self, action):
        assert len(action) == self.n_agents, "action length should be number of agents"

        next_state = None
        reward = []
        terminated = False
        info = {}

        self._move_agents(action)

        self._move_mice()

        # calculate boolean matrices indicating whether in catch/observation range
        self.catch_matrix = self._calc_in_range_matrix(self.agents, self.mice, 2*self.entity_size)
        self.agent_agent_obs_matrix = self._calc_in_range_matrix(self.agents, self.agents, self.observation_radius)
        self.agent_mouse_obs_matrix = self._calc_in_range_matrix(self.agents, self.mice, self.observation_radius)
        self.agent_agent_comm_matrix = self._calc_in_range_matrix(self.agents, self.agents, self.communication_radius)

        self._check_caught()

        next_state, communication = self._get_obs()

        reward = self._calc_reward()
        
        terminated = all([caught for _,_,caught in self.mice])

        return next_state, communication, reward, terminated, info
    
    # entity list (agent or mice)
    # returns matrix of shape (len(e_list1), len(e_list2))
    # where entry [i,j] tells us whether entry i and entry j 
    # of their respective arrays are within given range or not 
    def _calc_in_range_matrix(self, list1, list2, range):
        x = self._calc_dists(list1,list2)
        x = x < range
        return x
    
    def _calc_dists(self, list1,list2):
        t1 = np.array(list1)[:,:2]
        t2 = np.array(list2)[:,:2]
        t1 = t1[:,np.newaxis,:]
        t2 = t2[np.newaxis,:]

        x = t1 - t2
        x = x**2
        x = x[:,:,0] + x[:,:,1]
        x = np.sqrt(x)
        return x
    
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

    def _check_caught(self):
        # check if mouse caught
        for i in range(self.n_mice):

            _, _, caught = self.mice[i]
            if caught:
                continue

            for j in range(self.n_agents):
                if self.catch_matrix[j][i]:
                    self.mice[i] = (self.mice[i][0],self.mice[i][1],1)
                    # if mouse caught, check obs radius to mark all agents that saw it getting caught
                    for k in range(self.n_agents):
                        if self.agent_mouse_obs_matrix[k][i]:
                            self.agent_mice_list[k].append(i)
                    break

    def _calc_reward(self):
        reward = [self.step_cost for _ in range(self.n_agents)]

        dists = self._calc_dists(self.agents, self.mice)
        
        for i in range(self.n_agents):
            min_dist = self.observation_radius
            for j in range(self.n_mice):
                if not self.mice[j][2]:
                    if self.agent_mouse_obs_matrix[i][j]:
                        min_dist = min(min_dist, dists[i][j])
            reward[i] -= min_dist

        return reward

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

        # for checking if range matrices work
        # pygame.font.init()
        # my_font = pygame.font.SysFont('Comic Sans MS', 30)

        for i,a in enumerate(self.agents):
            x,y,*_ = a
            x *= self.window_size
            y *= self.window_size
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (x,y),
                self.entity_size*self.window_size,
            )
            pygame.draw.circle(
                canvas,
                (0,0,0),
                (x,y),
                self.observation_radius*self.window_size,
                width=1
            )
            
            # text_surface = my_font.render("{}".format(self.agent_mouse_obs_matrix[i].sum()), False, (0,0,0))
            # canvas.blit(text_surface, (x,y))
        
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
