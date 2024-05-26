import sys
import torch
from torch.distributions import Normal, Bernoulli, MultivariateNormal

class Lumberjacks_State_Distribution(torch.distributions.distribution.Distribution):

    def __init__(self, n_agents, n_trees, agent_id):
        pass