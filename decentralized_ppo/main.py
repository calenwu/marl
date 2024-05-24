import gym
import torch as T
import numpy as np
import matplotlib.pyplot as plt
import time
from agent import Agent


def state_to_array_lumber(state):
        state_list = []
        for agent in state[0]:
            state_list.append(agent[0])
            state_list.append(agent[1])
        for tree in state[1]:
            state_list.append(tree[0][0])
            state_list.append(tree[0][1])
            state_list.append(tree[1])
        return state_list


def __main__():
    n_trees = 4
    n_agents = 1
    agent = Agent(5**n_agents, (22*n_agents,), alpha=0.00003, n_epochs=4, batch_size=16)
    env = gym.make('ma_gym:Lumberjacks-v1', grid_shape=(5, 5), n_trees=n_trees, n_agents=n_agents)
    scores = []
    for ep in range(5000):
        if ep % 100 == 0:
             print(f"Epp: {ep}")
             print(np.mean(np.array(scores)))
             scores = []
        state = env.reset()
        score = 0
        steps = 0
        for i in range(100):
            action, probs, value, _ = agent.choose_action(state)
            state_n, reward, done, info = env.step([action])
            agent.remember(state_n, action, probs, value, reward, done)
            score = score + reward
            if np.all(done):
                break
            scores.append(score)
            steps += 1
        agent.learn()
    agent.save_models()
    """state = env.reset()
    for i in range(25):
        action, prob, val, _ = agent.choose_action(state)
        observation_, reward, done, info = env.step([action])
        if np.all(done):
            break
        env.render()
        time.sleep(0.25)
    env.close()"""
            


            
      
__main__()