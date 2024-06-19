import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import *


def orthogonal_init(layer, gain=1.0):
	for name, param in layer.named_parameters():
		if 'bias' in name:
			nn.init.constant_(param, 0)
		elif 'weight' in name:
			nn.init.orthogonal_(param, gain=gain)


class Actor_MLP(nn.Module):
	def __init__(self, obs_dim, action_dim, hidden_dim=64):
		super(Actor_MLP, self).__init__()
		self.fc1 = nn.Linear(obs_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, action_dim)
		self.activate_func = nn.Tanh()

		orthogonal_init(self.fc1)
		orthogonal_init(self.fc2)
		orthogonal_init(self.fc3, gain=0.01)

	def forward(self, actor_input):
		x = self.activate_func(self.fc1(actor_input))
		x = self.activate_func(self.fc2(x))
		prob = torch.softmax(self.fc3(x), dim=-1)
		return prob


class Critic_MLP(nn.Module):
	def __init__(self, critic_input_dim: int, hidden_dim: int=64):
		super(Critic_MLP, self).__init__()
		self.fc1 = nn.Linear(critic_input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, 1)
		self.activate_func = nn.Tanh()
		orthogonal_init(self.fc1)
		orthogonal_init(self.fc2)
		orthogonal_init(self.fc3)

	def forward(self, critic_input):
		x = self.activate_func(self.fc1(critic_input))
		x = self.activate_func(self.fc2(x))
		value = self.fc3(x)
		return value


class MAPPO_MPE:
	def __init__(self, args):
		self.N = args.N
		self.action_dim = args.action_dim
		self.obs_dim = args.obs_dim
		self.state_dim = args.state_dim
		self.episode_limit = args.episode_limit
		self.rnn_hidden_dim = args.rnn_hidden_dim

		self.batch_size = args.batch_size
		self.mini_batch_size = args.mini_batch_size
		self.max_train_steps = args.max_train_steps
		self.lr = args.lr
		self.gamma = args.gamma
		self.lamda = args.lamda
		self.epsilon = args.epsilon
		self.K_epochs = args.K_epochs
		self.entropy_coef = args.entropy_coef
		self.set_adam_eps = args.set_adam_eps
		self.use_grad_clip = args.use_grad_clip
		self.use_lr_decay = args.use_lr_decay
		self.use_adv_norm = args.use_adv_norm
		self.use_rnn = args.use_rnn
		self.add_agent_id = args.add_agent_id
		self.use_value_clip = args.use_value_clip

		# get the input dimension of actor and critic
		self.actor_input_dim = args.obs_dim
		self.critic_input_dim = args.state_dim

		self.actor = Actor_MLP(self.actor_input_dim, self.action_dim)
		self.critic = Critic_MLP(self.critic_input_dim)

		self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
		self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)

	def choose_action(self, obs_n, evaluate):
		with torch.no_grad():
			actor_inputs = []
			obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)
			actor_inputs.append(obs_n)
			actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # actor_input.shape=(N, actor_input_dim)
			prob = self.actor(actor_inputs)  # prob.shape=(N,action_dim)
			if evaluate:  # When evaluating the policy, we select the action with the highest probability
				a_n = prob.argmax(dim=-1)
				return a_n.numpy(), None
			else:
				dist = Categorical(probs=prob)
				a_n = dist.sample()
				a_logprob_n = dist.log_prob(a_n)
				return a_n.numpy(), a_logprob_n.numpy()

	def get_value(self, s):
		with torch.no_grad():
			critic_inputs = []
			# Because each agent has the same global state, we need to repeat the global state 'N' times.
			s = torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)  # (state_dim,)-->(N,state_dim)
			critic_inputs.append(s)
			if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
				critic_inputs.append(torch.eye(self.N))
			critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # critic_input.shape=(N, critic_input_dim)
			v_n = self.critic(critic_inputs)  # v_n.shape(N,1)
			return v_n.numpy().flatten()

	def train(self, bufer, total_steps):
		batch = bufer.get_training_data()  # get training data

		# Calculate the advantage using GAE
		adv = []
		gae = 0
		with torch.no_grad():  # adv and td_target have no gradient
			deltas = batch['r_n'] + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1]  # deltas.shape=(batch_size,episode_limit,N)
			for t in reversed(range(self.episode_limit)):
				gae = deltas[:, t] + self.gamma * self.lamda * gae
				adv.insert(0, gae)
			adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,episode_limit,N)
			v_target = adv + batch['v_n'][:, :-1]  # v_target.shape(batch_size,episode_limit,N)
			if self.use_adv_norm:  # Trick 1: advantage normalization
				adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

		actor_inputs, critic_inputs = self.get_inputs(batch)

		# Optimize policy for K epochs:
		for _ in range(self.K_epochs):
			for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
				probs_now = self.actor(actor_inputs[index])
				values_now = self.critic(critic_inputs[index]).squeeze(-1)

				dist_now = Categorical(probs_now)
				dist_entropy = dist_now.entropy()
				a_logprob_n_now = dist_now.log_prob(batch['a_n'][index])
				ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())
				surr1 = ratios * adv[index]
				surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
				actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

				critic_loss = (values_now - v_target[index]) ** 2

				self.ac_optimizer.zero_grad()
				ac_loss = actor_loss.mean() + critic_loss.mean()
				ac_loss.backward()
				torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
				self.ac_optimizer.step()

		lr_now = self.lr * (1 - total_steps / self.max_train_steps)
		for p in self.ac_optimizer.param_groups:
			p['lr'] = lr_now

	def get_inputs(self, batch):
		actor_inputs, critic_inputs = [], []
		actor_inputs.append(batch['obs_n'])
		critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
		if self.add_agent_id:
			agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.episode_limit, 1, 1)
			actor_inputs.append(agent_id_one_hot)
			critic_inputs.append(agent_id_one_hot)

		actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)
		critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
		return actor_inputs, critic_inputs

	def save_model(self, env_name, number, seed, total_steps):
		torch.save(self.actor.state_dict(), "./model{}/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(number, env_name, number, seed, int(total_steps / 1000)))

	def load_model(self, env_name, number, seed, step):
		self.actor.load_state_dict(torch.load("./model{}/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(number, env_name, number, seed, step)))

