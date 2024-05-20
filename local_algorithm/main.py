import local_agent
import sys
import torch
import matplotlib.pyplot as plt
import os
import random
sys.path.append("../")
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import *
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse import *

def train_cat_mouse(num_episodes = 25):
    n_agents = 1
    n_prey = 2
    env = CatMouseMA(n_agents=n_agents, n_prey=n_prey) #render_mode='human'_agents+3*n_mice, action_dim=1, num_agents=num_agents, env=env)
    agents = []
    for i in range(n_agents):
        agents.append(local_agent.Agent(n_agents, n_prey, i)) 
    
    state_approximation = [[], []]
    for ep in range(num_episodes):
        observations, communication = env._get_obs()
        global_obs = env.get_global_obs()
        for i in range(n_agents):
            loc_obs = observations[i]
            agents[i].state_distr.update_estimation_local_observation(loc_obs)
            log_prob_state = agents[i].state_distr.log_prob(global_obs)
            state_approximation[i].append(log_prob_state)
        next_state, communication, reward, terminated, info = env.step([random.random()])
        if terminated:
            break
        env.render()
    plt.plot(state_approximation[0])
    plt.plot(state_approximation[1])
    plt.savefig('Plots/state_approx_.png')
    

def __main__():
    train_cat_mouse(100)

__main__()