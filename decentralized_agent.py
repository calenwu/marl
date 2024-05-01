import copy
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
#from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torch.distributions import Normal
#from utils import ReplayBuffer, get_env, run_episode
import actor_critic

class Agent:
	def __init__(self, state_dim, action_dim, num_agents, agent_id, mu_0, beta):
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.num_agents = num_agents
		self.agent_id = agent_id
		self.mu = mu_0
		self.beta = beta

		self.discount = 0.98
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print("Using device: {}".format(self.device))
		#self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
		self.setup_agent()

	def setup_agent(self):
		self.actor = actor_critic.Actor(self.state_dim)
		self.critic = actor_critic.Critic(self.state_dim, self.action_dim*self.num_agents).to(self.device)

	def choose_action(self, s):
		with torch.no_grad():
			action, log_prob = self.actor(s.to(self.device), False)
		return action, log_prob

	def get_action(self, s, train = True) -> np.ndarray:
		#return random.uniform(0,1)
		with torch.no_grad():
			state_tensor = torch.Tensor(np.array(s)).to(self.device)
			action = self.actor(state_tensor, train).cpu().numpy().squeeze(0)
		return np.atleast_1d(action)

	@staticmethod
	def run_gradient_update_step(object, loss: torch.Tensor):
		object.optimizer.zero_grad()
		loss.mean().backward()
		object.optimizer.step()

	def update_mu(self, t, r):
		beta = self.beta(t)
		self.mu = (1-beta) * self.mu + beta * r

	def update_critic(self, s_t, a_t, s_tn, a_tn, r):
		delta = r - self.mu + self.critic(s_tn, a_tn)-self.critic(s_t, a_t)
		self.run_gradient_update_step(self.critic, delta)

	def get_subset_actions(self) -> List:
		subset_actions = []
		for i in range(10):
			subset_actions.append(random.uniform(0,1))
		return subset_actions

	def update_actor(self, s_t, a_t):
		A = self.critic(s_t + a_t)
		for a_i in self.get_subset_actions():
			a = a_t.clone()
			a[self.agent_id] = a_i
			# This has to be changed to pi(a_i|s)
			prob = self.actor.get_prob(s_t, a_i)
			A -= prob * self.critic(s_t + a_t) #self.actor.get_prob(s_t, a_i)
		self.run_gradient_update_step(self.actor, A)
		A_grad = torch.autograd.grad(A, self.actor.parameters(), retain_graph=True)
		for param, grad in zip(self.actor.parameters(), A_grad):
			param.data.add_(self.beta * grad)  # update the actor parameters

	def get_omega(self):
		return self.critic.parameters()
	
	def set_omega(self, omega):
		self.critic.set_parameters(omega)