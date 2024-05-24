import time
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from normalization import Normalization
from buffer import Buffer
from agent import Agent
from pettingzoo.mpe import simple_spread_v3
# from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import CatMouseMA
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_discrete import CatMouseMAD


def trans_obs(obs):
	ret = []
	for agent_obs in obs:
		temp = []
		temp.append(agent_obs['agents']['cur_agent'])
		for agent_pos in agent_obs['agents']['position']:
			temp.append(agent_pos)
		for prey_pos in agent_obs['prey']['position']:
			temp.append(prey_pos)
		temp.append(agent_obs['prey']['caught'])
		ret.append(np.concatenate(temp))
	return np.array(ret)

def trans_state(state):
	ret = []
	for agent_pos in state['agents']['position']:
		ret += agent_pos.tolist()
	for i, prey_pos in enumerate(state['prey']['position']):
		ret += prey_pos.tolist()
		ret.append(state['prey']['caught'][i])
	return np.array(ret)

def get_action(action):
	action_dict = {
		0: 0,
		1: 0.25,
		2: 0.5,
		3: 0.75
	}
	return action_dict[action]


class SimpleSpreadV3:
	def __init__(self):
		self.env = simple_spread_v3.parallel_env(N=3, max_cycles=25, local_ratio=0.5,
			render_mode='human',
			continuous_actions=False)
		self.env.reset(seed=42)
		self.n_agents = self.env.num_agents
		self.obs_dim = [self.env.observation_spaces[agent].shape[0] for agent in self.env.agents][0]
		self.state_dim = self.obs_dim * self.n_agents
		self.action_dim = 5
		# env.action_dim_n = [env.action_spaces[agent].n for agent in env.agents][0]

	def reset(self):
		obs_n, info = self.env.reset()
		obs_n = np.array([obs_n[agent] for agent in obs_n.keys()])
		return obs_n, info

	def step(self, a_n):
		actions = {}
		for i, agent in enumerate(self.env.agents):
			actions[agent] = a_n[i]
		obs_next_n, r_n, done_n, trunc, info = self.env.step(actions)
		obs_next_n = np.array([obs_next_n[agent] for agent in obs_next_n.keys()])
		done_n = np.array([val for val in done_n.values()])
		r_n = list(r_n.values())
		return obs_next_n, r_n, done_n, trunc, info

	def render(self):
		self.env.render()

	def close(self):
		self.env.close()


class CatMouse:
	def __init__(self):
		self.env = CatMouseMAD(observation_radius=10, n_agents=2, n_prey=4)
		self.state_dim = self.env.n_agents * 2 + self.env.n_prey * 3
		self.obs_dim = self.env.n_agents * 3 + self.env.n_prey * 3
		# self.state_dim = 2
		# self.obs_dim = 2
		self.n_agents = self.env.n_agents
		self.action_dim = 4
		self.env.reset()

	def reset(self):
		obs_n, info = self.env.reset()
		obs_n = np.array(trans_obs(obs_n))
		return obs_n, info

	def step(self, a_n):
		obs_next_n, r_n, done_n, trunc, info = self.env.step([get_action(a) for a in a_n])
		obs_next_n = trans_obs(obs_next_n)
		done_n = [done_n]
		return obs_next_n, r_n, done_n, trunc, info
	def get_global_obs(self):
		return self.env.get_global_obs()
	
	def render(self):
		self.env.render()

	def close(self):
		self.env.close()


eps = []
rewards = []


class Runner_MAPPO:
	def __init__(self, env_name, number, seed):
		self.env_name = env_name
		self.number = number

		self.seed = seed
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)

		self.episode_limit = 25
		self.max_train_steps = 2000000
		self.evaluate_freq = 200

		# self.env = make_env(self.episode_limit, render_mode=None)
		# self.env = SimpleSpreadV3()
		self.env = CatMouse()
		self.n_agents = self.env.n_agents
		self.obs_dim = self.env.obs_dim
		self.action_dim = self.env.action_dim
		self.state_dim = self.env.state_dim
		self.batch_size = 64

		self.agent_n = Agent(
			continuous=False,
			n_agents=self.n_agents, obs_dim=self.obs_dim, action_dim=self.action_dim, state_dim=self.state_dim,
			episode_limit=self.episode_limit, batch_size=self.batch_size, mini_batch_size=8,
			max_train_steps=self.max_train_steps
		)
		self.buffer = Buffer(n_agents=self.n_agents, obs_dim=self.obs_dim,
				state_dim=self.state_dim, episode_limit=self.episode_limit, batch_size=self.batch_size)

		self.evaluate_rewards = []
		self.evaluate_rewards_timestep = []
		self.total_steps = 0

		self.reward_norm = Normalization(shape=self.n_agents)

	def run_episode(self, evaluate=False):
		episode_reward = 0
		obs_n, info = self.env.reset()

		for episode_step in range(self.episode_limit):
			a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents

			# cat_mouse
			s = trans_state(self.env.get_global_obs())

			# simple_spread
			# stay, right, left, top, bottom
			# a_n = [np.array([0,0,0.5,1,0.5]) for _ in range(self.n_agents)]
			# s = obs_n.flatten()

			v_n = self.agent_n.get_value(s)  # Get the state values (V(s)) of N agents

			obs_next_n, r_n, done_n, _, _ = self.env.step(a_n)
			# episode_reward += sum(r_n)
			episode_reward += r_n[0]

			if evaluate:
				time.sleep(0.1)
				# print(a_n)
				self.env.render()

			if not evaluate:
				r_n = self.reward_norm([r_n])
				self.buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

			obs_n = np.array(obs_next_n)
			if all(done_n):
				break

		if not evaluate:
			# Store v_n in the last step
			# cat_mouse
			s = trans_state(self.env.get_global_obs())

			#simple_spread
			# s = np.array(obs_n).flatten()

			v_n = self.agent_n.get_value(s)
			self.buffer.store_last_value(episode_step + 1, v_n)

		return episode_reward, episode_step + 1

	def train(self):
		total_episodes = 0
		while self.total_steps < self.max_train_steps:
			if total_episodes % self.evaluate_freq == 0:
				self.evaluate_policy()

			episode_reward, episode_steps = self.run_episode()
			total_episodes += 1
			self.total_steps += episode_steps

			if self.buffer.episode_num == self.batch_size:
				self.agent_n.train(self.buffer, self.total_steps)
				self.buffer.reset_buffer()

		self.evaluate_policy()
		self.env.close()

	def evaluate_policy(self):
		evaluate_reward = 0
		for _ in range(5):
			episode_reward, _ = self.run_episode(evaluate=True)
			evaluate_reward += episode_reward

		evaluate_reward /= 1
		self.evaluate_rewards.append(evaluate_reward)
		self.evaluate_rewards_timestep.append(self.total_steps)

		print(f'total_steps:{self.total_steps} \t evaluate_reward:{evaluate_reward}')



if __name__ == '__main__':
	runner = Runner_MAPPO(env_name='simple_spread_v3', number=3, seed=0)
	# runner.train()
	# runner.agent_n.save_model()

	# plt.figure(figsize=(10, 5))
	# plt.plot(runner.evaluate_rewards_timestep, runner.evaluate_rewards)
	# plt.xlabel('Episodes')
	# plt.ylabel('Reward')
	# plt.title('Reward vs Episodes')
	# plt.grid(True)
	# plt.savefig('reward_vs_episodes.png')
	# data = {'Episodes': runner.evaluate_rewards_timestep, 'Reward': runner.evaluate_rewards}
	# df = pd.DataFrame(data)
	# df.to_csv('reward_vs_episodes.csv', index=False)

	runner.agent_n.load_model()
	runner.evaluate_policy()
