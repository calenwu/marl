import copy
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import cat_mouse_state_distribution

from torch.distributions import Normal

class Agent:
	def __init__(self, num_agents, num_mice, agent_id):
		self.state_distr = cat_mouse_state_distribution.Cat_Mouse_State_Distribution(num_agents, num_mice, agent_id)
