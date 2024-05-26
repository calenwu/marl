import copy
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import cat_mouse_state_distribution
import coop_nav_state_distribution
import torch as T
import os
from torch.distributions import Categorical
import matplotlib.pyplot as plt
device = T.device('cpu')

from torch.distributions import Normal

class LocalPPOMemory:

	def __init__(self, n_agents, agent_id, batch_size = 16, max_size = 100):
		self.agent_id = agent_id
		self.n_agents = n_agents
		self.max_size = max_size
		self.batch_size = batch_size

		self.trans_ind = []				# a list of the indices of transitions stored in this buffer
		self.belief_states = []			# the current belief about the state in the transition
		self.comm = [] 					# list of lists containing all communicated with agents
		
		
		self.probs = []					
		self.vals = []					# sum of the currently observed values
		self.actions = []				# list of all the actions executed in the transitions (-1 for unknown)
		self.rewards = []				# sum of the rewards obtained in the transitions
		self.dones = []					# list of all the dones executed in the transitions (-1 for unknown)
		
	def get_size(self):
		return len(self.comm)

	def delete_memory(self, num_rem):
		self.trans_ind = self.trans_ind[num_rem:]
		self.comm = self.comm[num_rem:]
		self.belief_states = self.belief_states[num_rem:]
		self.vals = self.vals[num_rem:]
		self.probs = self.probs[num_rem:]
		self.actions = self.actions[num_rem:]
		self.rewards = self.rewards[num_rem:]
		self.dones = self.dones[num_rem:]

	def generate_training_batches(self):
		all_com = np.array([len(com) for com in self.comm])
		all_com = np.array(all_com == self.n_agents)
		np.array(self.belief_states)[all_com]
		n_states = np.array(self.belief_states)[all_com].shape[0]
		batch_start = np.arange(0, n_states, self.batch_size)
		indices = np.arange(n_states, dtype=np.int64)
		np.random.shuffle(indices)
		batches = [indices[i:i+self.batch_size] for i in batch_start]
		return np.array(self.belief_states)[all_com], np.array(self.actions)[all_com], np.array(self.probs)[all_com], np.array(self.vals)[all_com], np.array(self.rewards)[all_com], np.array(self.dones)[all_com], batches

	def store_memory(self, belief_state, action, probs, vals, reward, done, trans_ind):
		if self.get_size() >= self.max_size:
			self.delete_memory(int(self.max_size/5))
		
		self.trans_ind.append(trans_ind)
		self.comm.append([self.agent_id])
		
		self.belief_states.append(belief_state)
		all_actions = [-1 for _ in range(self.n_agents)]
		all_actions[self.agent_id] = action
		self.actions.append(all_actions)
		all_probs = [-1 for _ in range(self.n_agents)]
		all_probs[self.agent_id] = probs
		self.probs.append(all_probs) # What to do with probs ????
		all_vals = [-1 for _ in range(self.n_agents)]
		all_vals[self.agent_id] = vals 
		self.vals.append(all_vals)
		all_rewards = [-1 for _ in range(self.n_agents)]
		all_rewards[self.agent_id] = reward
		self.rewards.append(all_rewards)
		all_dones = [-1 for _ in range(self.n_agents)]
		all_dones[self.agent_id] = done
		self.dones.append(all_dones)

	# Assumes all transition indices are equall at the corresponding positions
	@staticmethod
	def merge_memories(agent_list, n_agents, distr):
		agent_ids = []
		
		for ag in agent_list:
			agent_ids.append(ag.agent_id)

		for i in range(agent_list[0].memory.get_size()):
			if len(agent_list[0].memory.comm[i]) == n_agents:
				continue
			all_com = []
			all_actions = [-1 for _ in range(n_agents)]
			all_probs = [-1 for _ in range(n_agents)]
			all_vals = [-1 for _ in range(n_agents)]
			all_rewards = [-1 for _ in range(n_agents)]
			all_dones = [-1 for _ in range(n_agents)]
			all_belief_states = []
			for ag in agent_list:
				ag.memory.comm[i] = list(set(agent_ids+ag.memory.comm[i]))
				all_probs[ag.agent_id] = ag.memory.probs[i][ag.agent_id]
				all_actions[ag.agent_id] = ag.memory.actions[i][ag.agent_id]
				all_vals[ag.agent_id] = ag.memory.vals[i][ag.agent_id]
				all_rewards[ag.agent_id] = ag.memory.rewards[i][ag.agent_id]
				all_dones[ag.agent_id] = ag.memory.dones[i][ag.agent_id]
				all_belief_states.append(ag.memory.belief_states[i])
			final_belief_state = distr.update_belief_state(all_belief_states,agent_ids, n_agents, n_agents)
			for ag in agent_list:
				ag.memory.belief_states[i] = final_belief_state
				ag.memory.probs[i] = all_probs
				ag.memory.actions[i] = all_actions
				ag.memory.vals[i] = all_vals
				ag.memory.rewards[i] = all_rewards
				ag.memory.dones[i] = all_dones
				ag.memory.belief_states[i] = final_belief_state
		return agent_list

class ActorNetwork(nn.Module):
	def __init__(self, n_actions, input_dims, alpha, agent_id, fc1_dims=32, fc2_dims=32, chkpt_dir='tmp/ppo'):
		super(ActorNetwork, self).__init__()
		self.checkpoint_file = os.path.join(chkpt_dir, f'actor_torch_ppo{agent_id}')
		self.actor = nn.Sequential(
			nn.Linear(*input_dims, fc1_dims),
			nn.Tanh(),
			nn.Linear(fc1_dims, fc2_dims),
			nn.Tanh(),
			nn.Linear(fc2_dims, n_actions),
			nn.Softmax(dim=-1)
		)
		self.optimizer = optim.Adam(self.parameters(), lr=alpha)
		self.device = device
		self.to(self.device)

	def forward(self, state):
		dist = self.actor(state)
		dist = Categorical(dist)
		return dist

	def save_checkpoint(self):
		T.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		T.load(self.state_dict(), self.checkpoint_file)

class CriticNetwork(nn.Module):
	def __init__(self, input_dims, alpha, agent_id, fc1_dims=32, fc2_dims=32, chkpt_dir='tmp/ppo'):
		super(CriticNetwork, self).__init__()
		self.checkpoint_file = os.path.join(chkpt_dir, f'critic_torch_ppo_{agent_id}')
		self.critic = nn.Sequential(
			nn.Linear(*input_dims, fc1_dims),
			nn.Tanh(),
			nn.Linear(fc1_dims, fc2_dims),
			nn.Tanh(),
			nn.Linear(fc2_dims, 1)
		)
		self.optimizer = optim.Adam(self.parameters(), lr=alpha)
		self.device = device
		self.to(self.device)
	
	def forward(self, state):
		value = self.critic(state)
		return value
	
	def save_checkpoint(self):
		T.save(self.state_dict(), self.checkpoint_file)

	def load_checkpoint(self):
		self.load_state_dict(T.load(self.checkpoint_file))


class Local_Agent:
	
	def __init__(self, input_dims, num_agents, n_actions, agent_id, state_distribution, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
			policy_clip=0.2, batch_size=64, n_epochs=10, layer_size = 64):
		self.agent_id = agent_id
		self.state_distr = state_distribution
		self.num_agents = num_agents
		self.n_actions = n_actions
		self.gamma = gamma
		self.policy_clip = policy_clip
		self.n_epochs = n_epochs
		self.gae_lambda = gae_lambda
	
		self.memory = LocalPPOMemory(num_agents, agent_id)
		self.actor = ActorNetwork(n_actions, input_dims, alpha, agent_id, fc1_dims= layer_size, fc2_dims= layer_size)
		self.critic = CriticNetwork(input_dims, alpha, agent_id, fc1_dims= layer_size, fc2_dims= layer_size)

	def observe(self, action, probs, vals, reward, done, trans_ind):
		belief_state = self.state_distr.get_belief_state()
		self.memory.store_memory(belief_state, action, probs, vals, reward, done, trans_ind)

	def choose_action(self, observation):
		self.state_distr.update_estimation_local_observation(observation)
		belief_state = self.state_distr.get_belief_state()
		state = T.tensor([belief_state], dtype=T.float32).to(self.actor.device)
		dist = self.actor(state)
		value = self.critic(state)
		action = dist.sample()
		probs = T.squeeze(dist.log_prob(action)).item()
		action = T.squeeze(action).item()
		value = T.squeeze(value).item()
		return action, probs, value, T.softmax(dist.logits, dim=-1)
		#return random.randint(0, self.n_actions-1), 1, 1, np.ones(self.n_actions)/self.n_actions

	@staticmethod
	def communicate(agent_list, n_agents, distr):
		"""
		Updates all relevant parameters of the agents that are able to communicate
		"""
		agent_list = LocalPPOMemory.merge_memories(agent_list, n_agents, distr)
		return agent_list
	
	def learn(self):
		for _ in range(self.n_epochs):
			state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_training_batches()
			action_arr = action_arr.T[self.agent_id]
			reward_arr = np.sum(reward_arr, axis=1)
			old_prob_arr = np.prod(old_prob_arr, axis=1)
			vals_arr = np.sum(vals_arr, axis = 1)
			dones_arr = np.all(dones_arr, axis=1)
			values = vals_arr
			advantages = np.zeros(len(reward_arr), dtype=np.float32)

			for t in range(len(reward_arr) - 1):
				discount = 1
				a_t = 0
				for k in range(t, len(reward_arr) - 1):
					a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
					discount *= self.gamma * self.gae_lambda
				advantages[t] = a_t
			advantages = T.tensor(advantages, dtype=T.float32).to(self.actor.device)
			values = T.tensor(values, dtype=T.float32).to(self.actor.device)
			for batch in batches:
				states = T.tensor(state_arr[batch], dtype=T.float32).to(self.actor.device)
				old_probs = T.tensor(old_prob_arr[batch], dtype=T.float32).to(self.actor.device)
				actions = T.tensor(action_arr[batch], dtype=T.float32).to(self.actor.device)

				dist = self.actor(states)
				critic_values = self.critic(states)
				critic_values = T.squeeze(critic_values)

				new_probs = dist.log_prob(actions)
				prob_ratio = new_probs.exp() / old_probs.exp()

				weighted_probs = advantages[batch] * prob_ratio
				weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages[batch]
				actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
				returns = advantages[batch] + values[batch]
				critic_loss = (returns - critic_values) ** 2
				critic_loss = critic_loss.mean()
				total_loss = actor_loss + 0.5 * critic_loss
				self.actor.optimizer.zero_grad()
				self.critic.optimizer.zero_grad()
				total_loss.mean().backward()
				self.actor.optimizer.step()
				self.critic.optimizer.step()
		#self.memory.clear_memory()

	def save_models(self):
		print('... saving models ...')
		self.actor.save_checkpoint()
		self.critic.save_checkpoint()

	def load_models(self):
		print('... loading models ...')
		self.actor.load_checkpoint()
		self.critic.load_checkpoint()