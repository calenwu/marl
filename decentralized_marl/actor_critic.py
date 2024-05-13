import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
#from gym.wrappers.monitoring.video_recorder import VideoRecorder
from torch.distributions import Normal
#from utils import ReplayBuffer, get_env, run_episode


class NeuralNetwork(nn.Module):
	'''
	This class implements a neural network with a variable number of hidden layers and hidden units.
	You may use this function to parametrize your policy and critic networks.
	'''
	def __init__(self, input_dim: int, output_dim: int, hidden_size: int, hidden_layers: int, activation: str):
		super(NeuralNetwork, self).__init__()
		layers = [nn.Linear(input_dim, hidden_size)]
		layers += [nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layers)]
		layers += [nn.Linear(hidden_size, output_dim)]

		self.layers = nn.ModuleList(layers)
		self.activation = activation

	def forward(self, s: torch.Tensor) -> torch.Tensor:
		# TODO: Implement the forward pass for the neural network you have defined.
		for layer in self.layers[:-1]:
			if self.activation == 'softmax':
				s = F.softmax(layer(s))
			else:
				s = F.leaky_relu(layer(s))
		return self.layers[-1](s)


class Actor(nn.Module):
	"""
	Policy
	"""
	def __init__(self, state_dim, action_dim, hidden_size = 10, hidden_layers = 1, actor_lr = 1e-4, device: torch.device = torch.device('cpu')):
		super(Actor, self).__init__()
		self.hidden_size = hidden_size
		self.hidden_layers = hidden_layers
		self.actor_lr = actor_lr
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = device
		#self.LOG_STD_MIN = -20
		#self.LOG_STD_MAX = 2
		self.setup_actor()

	def setup_actor(self):
		'''
		This function sets up the actor network in the Actor class.
		'''
		# TODO: Implement this function which sets up the actor network.
		# Take a look at the NeuralNetwork class in utils.py.
		self.network = NeuralNetwork(
			self.state_dim,
			self.action_dim,
			self.hidden_size,
			self.hidden_layers,
			'relu').to(self.device)
		self.optimizer = optim.Adam(self.network.parameters(), lr=self.actor_lr, amsgrad=True)

	def clamp_log_std(self, log_std: torch.Tensor) -> torch.Tensor:
		'''
		:param log_std: torch.Tensor, log_std of the policy.
		Returns:
		:param log_std: torch.Tensor, log_std of the policy clamped between LOG_STD_MIN and LOG_STD_MAX.
		'''
		return torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
	
	def forward(self, state: torch.Tensor) -> (torch.Tensor, torch.Tensor):
		'''
		:param state: torch.Tensor, state of the agent
		:param deterministic: boolean, if true return a deterministic action otherwise sample from the policy distribution.
		Returns:
		:param action: torch.Tensor, action the policy returns for the state.
		:param log_prob: log_probability of the the action.
		'''
		# TODO: Implement this function which returns an action and its log probability.
		# If working with stochastic policies, make sure that its log_std are clamped 
		# using the clamp_log_std function.
		output = self.network(state)
		action_probs = F.softmax(output)
		return action_probs
	
	def get_prob(self, s_t, a_t):
		action_probs = self.forward(torch.Tensor(np.array(s_t)))
		prob = action_probs[a_t]
		return prob


class Critic(nn.Module):
	"""
	Q-function(s)
	"""
	def __init__(self, state_dim, action_dim, hidden_size = 5, hidden_layers = 1, critic_lr = 1e-10, device: torch.device = torch.device('cpu')):
		super(Critic, self).__init__()
		self.hidden_size = hidden_size
		self.hidden_layers = hidden_layers
		self.critic_lr = critic_lr
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.device = device
		self.network_copy = None
		self.setup_critic()

	def setup_critic(self):
		self.network = NeuralNetwork(self.state_dim + self.action_dim, 1, self.hidden_size, self.hidden_layers, 'relu')
		self.optimizer = optim.Adam(self.parameters(), lr=self.critic_lr)

	def forward(self, s_a):
		s_a_tensor = torch.Tensor(np.array(s_a))
		return self.network.forward(s_a_tensor)


class TrainableParameter:
	'''
	This class could be used to define a trainable parameter in your method. You could find it 
	useful if you try to implement the entropy temerature parameter for SAC algorithm.
	'''
	def __init__(self, init_param: float, lr_param: float, 
				 train_param: bool, device: torch.device = torch.device('cpu')):
		self.log_param = torch.tensor(np.log(init_param), requires_grad=train_param, device=device)
		self.optimizer = optim.Adam([self.log_param], lr=lr_param)

	def get_param(self) -> torch.Tensor:
		return torch.exp(self.log_param)

	def get_log_param(self) -> torch.Tensor:
		return self.log_param
