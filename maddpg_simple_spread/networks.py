import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
# if T.backends.mps.is_available():
#	 device = T.device("mps")

class CriticNetwork(nn.Module):
	def __init__(self, beta: float, input_dims: int, fc1_dims: int, fc2_dims: int, name: str):
		super(CriticNetwork, self).__init__()

		self.name = name

		self.fc1 = nn.Linear(input_dims, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		self.q = nn.Linear(fc2_dims, 1)

		self.optimizer = optim.Adam(self.parameters(), lr=beta)
		self.device = device
		self.to(self.device)

	def forward(self, state, action):
		x = F.relu(self.fc1(T.cat([state, action], dim=1)))
		x = F.relu(self.fc2(x))
		q = self.q(x)

		return q

	def save_checkpoint(self, t: str):
		checkpoint_temp = os.path.join('checkpoints', self.name) + t
		os.makedirs(os.path.dirname(checkpoint_temp), exist_ok=True)
		checkpoint_path = checkpoint_temp + '.pt'
		T.save(self.state_dict(), checkpoint_path)


	def load_checkpoint(self, t: str):
		checkpoint_temp = self.chkpt_file + t
		checkpoint_path = checkpoint_temp + '.pt'
		self.load_state_dict(T.load(checkpoint_path))


# Actor Network:
# used to approximate the policy function for each agent
class ActorNetwork(nn.Module):
	def __init__(self, alpha: float, input_dims: int, fc1_dims: int, fc2_dims: int, n_actions: int, name: str):
		super(ActorNetwork, self).__init__()

		self.name = name

		self.fc1 = nn.Linear(input_dims, fc1_dims)
		self.fc2 = nn.Linear(fc1_dims, fc2_dims)
		self.pi = nn.Linear(fc2_dims, n_actions)

		self.optimizer = optim.Adam(self.parameters(), lr=alpha)
		self.device = device

		self.to(self.device)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		pi = T.softmax(self.pi(x), dim=1)

		return pi

	def save_checkpoint(self, t: str):
		checkpoint_temp = os.path.join('checkpoints', self.name) + t
		os.makedirs(os.path.dirname(checkpoint_temp), exist_ok=True)
		checkpoint_path = checkpoint_temp + '.pt' 
		T.save(self.state_dict(), checkpoint_path)

	def load_checkpoint(self, t: str):
		checkpoint_temp = os.path.join('checkpoints', self.name) + t
		checkpoint_path = checkpoint_temp + '.pt'
		self.load_state_dict(T.load(checkpoint_path))
