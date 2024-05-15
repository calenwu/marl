import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import *
import numpy as np
import numpy.typing as npt


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
	for name, param in layer.named_parameters():
		if 'bias' in name:
			nn.init.constant_(param, 0)
		elif 'weight' in name:
			nn.init.orthogonal_(param, gain=gain)


class Actor_MLP(nn.Module):
	def __init__(self, obs_dim: int, action_dim: int, hidden_dim=256, continuous=False):
		super(Actor_MLP, self).__init__()
		activation_func = nn.Tanh() if continuous else nn.ReLU()
		self.fc1 = nn.Linear(obs_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc4 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, 2) if continuous else nn.Linear(hidden_dim, action_dim)
		orthogonal_init(self.fc1)
		orthogonal_init(self.fc2)
		orthogonal_init(self.fc4)
		orthogonal_init(self.fc3, gain=0.01)
		if continuous:
			self.actor = nn.Sequential(
				self.fc1,
				activation_func,
				self.fc2,
				activation_func,
				self.fc4,
				activation_func,
				self.fc3,
				nn.Tanh(),
			)
		else:
			self.actor = nn.Sequential(
				self.fc1,
				activation_func,
				self.fc2,
				activation_func,
				self.fc3,
				nn.Softmax(dim=-1)
			)

	def forward(self, observation):
		return self.actor(observation)

	def save_checkpoint(self, filename='./checkpoints/mapppo_actor.pth'):
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		torch.save(self.state_dict(), filename)

	def load_checkpoint(self, filename='./checkpoints/mapppo_actor.pth'):
		self.load_state_dict(torch.load(filename))


class Critic_MLP(nn.Module):
	def __init__(self, global_state_dim: int, hidden_dim=256):
		super(Critic_MLP, self).__init__()
		self.fc1 = nn.Linear(global_state_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc4 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, 1)
		orthogonal_init(self.fc1)
		orthogonal_init(self.fc2)
		orthogonal_init(self.fc4)
		orthogonal_init(self.fc3)
		self.critic = nn.Sequential(
			self.fc1,
			nn.ReLU(),
			self.fc2,
			nn.ReLU(),
			self.fc4,
			nn.ReLU(),
			self.fc3
		)

	def forward(self, global_state: npt.NDArray):
		return self.critic(global_state)

	def save_checkpoint(self, filename='./checkpoints/mapppo_critic.pth'):
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		torch.save(self.state_dict(), filename)

	def load_checkpoint(self, filename='./checkpoints/mapppo_critic.pth'):
		self.load_state_dict(torch.load(filename))


class Agent:
	def __init__(self, continuous: bool, n_agents: int, state_dim: int, obs_dim: int, action_dim: int,
			episode_limit=25, batch_size=32, mini_batch_size=8, max_train_steps=int(3e6),
			lr=5e-4, gamma=0.99, lambda_=0.95, epsilon=0.2, K_epochs=15, entropy_coef=0.01,):
		self.continuous = continuous

		self.N = n_agents
		self.action_dim = state_dim
		self.obs_dim = obs_dim
		self.state_dim = state_dim

		self.episode_limit = episode_limit
		self.batch_size = batch_size
		self.mini_batch_size = mini_batch_size
		self.max_train_steps = max_train_steps

		self.lr = lr
		self.gamma = gamma
		self.lamda = lambda_
		self.epsilon = epsilon
		self.K_epochs = K_epochs
		self.entropy_coef = entropy_coef

		self.actor = Actor_MLP(obs_dim, action_dim, continuous=continuous)
		self.critic = Critic_MLP(state_dim)

		self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
		self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)

	def choose_action(self, obs_n: npt.NDArray, evaluate: bool):
		with torch.no_grad():
			actor_inputs = []
			obs_n = torch.tensor(obs_n, dtype=torch.float32)
			actor_inputs.append(obs_n)

			actor_inputs = torch.cat(actor_inputs, dim=-1)
			if self.continuous:
				temp = self.actor(actor_inputs)
				tensor1, tensor2 = torch.split(temp, split_size_or_sections=1, dim=1)
				mu = tensor1.squeeze(dim=1)
				sig = tensor2.squeeze(dim=1)
				if evaluate:
					return mu.numpy(), None
				dist = Normal(mu, sig)
				a_n = dist.sample()
				a_logprob_n = dist.log_prob(a_n)
				return a_n.numpy(), a_logprob_n.numpy()
			prob = self.actor(actor_inputs)
			# if evaluate:
			# 	a_n = prob.argmax(dim=-1)
			# 	return a_n.numpy(), None
			dist = Categorical(probs=prob)
			a_n = dist.sample()
			a_logprob_n = dist.log_prob(a_n)
			# obs_n = obs_n.numpy()[0]
			# if obs_n[1] - obs_n[3] >= 0:
			# 	# go left
			# 	x = [2]
			# else:
			# 	# go right
			# 	x = [0]
			# if obs_n[2] - obs_n[4] >= 0:
			# 	# go up
			# 	y = [3]
			# else:
			# 	# go down
			# 	y = [1]
			# if abs(obs_n[1] - obs_n[3]) > abs(obs_n[2] - obs_n[4]):
			# 	a_n = torch.tensor(x)
			# else:
			# 	a_n = torch.tensor(y)
			return a_n.numpy(), a_logprob_n.numpy()

	def get_value(self, s):
		with torch.no_grad():
			critic_inputs = []
			# Each agent has same global state
			s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)
			critic_inputs.append(s)
			critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
			v_n = self.critic(critic_inputs)
			return v_n.numpy().flatten()

	def train(self, buffer, total_steps):
		batch = buffer.get_training_data()
		adv = []
		gae = 0
		with torch.no_grad():
			deltas = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1]
			for t in reversed(range(self.episode_limit)):
				gae = deltas[:, t] + self.gamma * self.lamda * gae
				adv.insert(0, gae)
			adv = torch.stack(adv, dim=1)
			v_target = adv + batch['v_n'][:, :-1]
			adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

		actor_inputs, critic_inputs = self.get_inputs(batch)

		# Optimize policy for K epochs:
		for _ in range(self.K_epochs):
			for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
				values_now = self.critic(critic_inputs[index]).squeeze(-1)
				if self.continuous:
					mu, sig = self.actor(actor_inputs[index])
					dist_now = Normal(mu, sig)
					dist_entropy = dist_now.entropy()
				else:
					probs_now = self.actor(actor_inputs[index])
					dist_now = Categorical(probs_now)
					dist_entropy = dist_now.entropy()
				a_logprob_n_now = dist_now.log_prob(batch['a_n'][index])
				# a/b=exp(log(a)-log(b))
				ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())
				surr1 = ratios * adv[index]
				surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
				actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

				critic_loss = (values_now - v_target[index]) ** 2

				self.ac_optimizer.zero_grad()
				ac_loss = actor_loss.mean() + critic_loss.mean()
				ac_loss.backward()
				# Clip Gradient
				torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
				self.ac_optimizer.step()

		# learning ratge decay
		lr_now = self.lr * (1 - total_steps / self.max_train_steps)
		for p in self.ac_optimizer.param_groups:
			p['lr'] = lr_now

	def get_inputs(self, batch):
		actor_inputs, critic_inputs = [], []
		actor_inputs.append(batch['obs_n'])
		critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))

		actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_inputs.shape=(batch_size, episode_limit, N, actor_input_dim)
		critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_inputs.shape=(batch_size, episode_limit, N, critic_input_dim)
		return actor_inputs, critic_inputs

	def save_model(self):
		self.actor.save_checkpoint()
		self.critic.save_checkpoint()

	def load_model(self):
		self.actor.load_checkpoint()
		self.critic.load_checkpoint()
