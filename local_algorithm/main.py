import time
import local_agent
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from pettingzoo.mpe import simple_spread_v3
import random
sys.path.append("../")
# from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import *
# from marl_gym.marl_gym.envs.cat_mouse.cat_mouse import *
from agent import Agent
# from cat_mouse_state_distribution import Cat_Mouse_State_Distribution
from coop_nav_state_distribution import Coop_Nav_State_Distribution

# def train_cat_mouse(num_episodes = 5000):
#     n_agents = 1
#     n_prey = 2
#     env = CatMouseMA(n_agents=n_agents, n_prey=n_prey) #render_mode='human'_agents+3*n_mice, action_dim=1, num_agents=num_agents, env=env)
#     state_dim = 4*n_agents+5*n_prey 
#     agent = Agent(5, (state_dim,), alpha=0.00003, n_epochs=4, batch_size=16)
#     state_distr = Cat_Mouse_State_Distribution(n_agents, n_prey, 0)
#     eps_rewards = []
#     for ep in range(num_episodes):
#         env.reset()
#         belief_state = state_distr.get_belief_state()
#         observations, communication = env._get_obs()
#         state_distr.update_estimation_local_observation(observations[0])
#         ep_rew = 0
#         discount = 1
#         for i in range(100):
#             action, probs, value, _ = agent.choose_action(belief_state)
#             state_n, reward, done, _, _ = env.step([action/5])
#             ep_rew += discount*reward[0]
#             discount *= 0.99
#             eps_rewards.append(ep_rew)
#             observations_n, communication_n = env._get_obs()
#             state_distr.update_estimation_local_observation(observations_n[0])
#             belief_state_n = state_distr.get_belief_state()
#             agent.remember(belief_state_n, action, probs, value, reward, done)
#             if np.all(done):
#                 break
#         print(ep_rew)
#         eps_rewards.append(ep_rew)
#         agent.learn()


d = []
for x in range(5):
	d.append([x])
     
def train_cooperative_nav(num_episodes):
    n_agents = 1
    n_targets = 1
    env = simple_spread_v3.parallel_env(N=1, max_cycles=25,local_ratio=0.5, continuous_actions=False, render_mode='human')#render_mode=render_mode, 
    state_dim = 4
    agent = Agent(5, (state_dim,), alpha=0.0003, n_epochs=4, batch_size=16)
    state_distr = Coop_Nav_State_Distribution(n_agents, n_targets, 0)
    eps_rewards = []
    agent.load_models()
    n_steps = 0

    for ep in range(num_episodes):
        state, _ = env.reset()
        state_distr.update_estimation_local_observation(state)
        belief_state = state_distr.get_belief_state()
        ep_rew = 0
        discount = 1
        i = 0
        done = False
        while not done and i < 25:
            action, probs, value, _ = agent.choose_action(belief_state)
            actions_dict = {}
            for agent_id, ag in enumerate(env.agents):
                actions_dict[ag] = d[action][agent_id]
            state_n, reward, done, _, _ = env.step(actions_dict)
            reward = sum(list(reward.values()))
            done = all(list(done.values()))
            # print(belief_state)
            time.sleep(0.25)
            env.render()
            ep_rew += discount * reward
            discount *= 0.99
            state_distr.update_estimation_local_observation(state_n)
            belief_state_n = state_distr.get_belief_state()
            agent.remember(belief_state, action, probs, value, reward, done)
            # belief_state = belief_state_n
            # print(state_n['agent_0'][2].item() + state_n['agent_0'][4].item(), state_n['agent_0'][3].item() + state_n['agent_0'][5].item())
            # print(belief_state_n)
            # time.sleep(0.25)
            n_steps += 1
            if n_steps % 20 == 0:
                agent.learn()
            state = state_n
            belief_state = belief_state_n
            i += 1

        eps_rewards.append(ep_rew)
        avg_score = np.mean(eps_rewards[-100:])
        print('episode', ep, 'score %.1f' % eps_rewards[-1], 'avg score %.1f' % avg_score, 'time_steps', n_steps, )

    agent.save_models()
def __main__():
    train_cooperative_nav(12000)

__main__()