import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torch.distributions import Normal
from utils import ReplayBuffer, get_env, run_episode
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
		self.memory = ReplayBuffer(self.min_buffer_size, self.max_buffer_size, self.device)
		self.setup_agent()

	def setup_agent(self):
		self.actor = actor_critic.Actor(256, 2, 1e-4, self.state_dim, self.action_dim, self.device)
		self.critic = actor_critic.Critic(256, 2, 1e-4, self.state_dim, self.action_dim*self.num_agents).to(self.device)

	def choose_action(self, s):
		with torch.no_grad():
			action, log_prob = self.actor(s.to(self.device), False)
		return action, log_prob

	def get_action(self, s: np.ndarray, train: bool) -> np.ndarray:
		with torch.no_grad():
			action, _ = self.actor(torch.Tensor(s).to(self.device), train)
			action = action.cpu().numpy().squeeze(0)
		assert action.shape == (1,), 'Incorrect action shape.'
		assert isinstance(action, np.ndarray ), 'Action dtype must be np.ndarray'
		return np.atleast_1d(action)

	@staticmethod
	def run_gradient_update_step(object, loss: torch.Tensor):
		object.optimizer.zero_grad()
		loss.mean().backward()
		object.optimizer.step()

	def update_mu(self, t, r):
		beta = self.beta(t)
		self.mu = (1-beta) * self.mu + beta * (r)

	def update_critic(self, s, a, r):
		delta = r - self.mu + self.critic(s, a)
		

	def update_actor(self, s, a, r):
		pass

	def get_omega(self):
		return self.critic.parameters()
	
	def set_omega(self, omega):
		self.critic.set_parameters(omega)