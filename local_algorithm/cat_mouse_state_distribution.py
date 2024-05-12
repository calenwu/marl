import sys
import torch
from torch.distributions import Normal, Bernoulli, MultivariateNormal

class Cat_Mouse_State_Distribution(torch.distributions.distribution.Distribution):

    def __init__(self, num_agents, num_mice, agent_id):
        super().__init__()
        self.num_agents = num_agents
        self.num_mice = num_mice
        self.agent_id = agent_id
        self.agent_pos_distribution = []
        self.mouse_pos_distribution = []
        self.mouse_found_distribution = []
        for i in range(num_agents):
            position_mean = torch.Tensor([0.5, 0.5])
            position_var = torch.Tensor([[10, 0],[0, 10]])
            agent_pos_i_distribution = MultivariateNormal(position_mean, position_var)
            self.agent_pos_distribution.append(agent_pos_i_distribution)
        
        for i in range(num_mice):
            position_mean = torch.Tensor([0.5, 0.5])
            position_var = torch.Tensor([[10, 0],[0, 10]])
            mouse_pos_i_distribution = MultivariateNormal(position_mean, position_var)
            self.mouse_pos_distribution.append(mouse_pos_i_distribution)

        for i in range(num_mice):
            mouse_found_i_distribution = Bernoulli(0)
            self.mouse_found_distribution.append(mouse_found_i_distribution)

    def log_prob(self, glob_state):
        lp = 0
        for i in range(self.num_agents):
            pos_agent_i = torch.Tensor(glob_state[0][i])
            lp += self.agent_pos_distribution[i].log_prob(pos_agent_i)
        for i in range(self.num_mice):
            if glob_state[1][i][2] == 1:
                continue
            pos_mouse_i = torch.Tensor([glob_state[1][i][0], glob_state[1][i][1]])
            lp += self.mouse_pos_distribution[i].log_prob(pos_mouse_i)
        for i in range(self.num_mice):
            found_mouse_i = torch.Tensor([glob_state[1][i][2]])
            lp +=  self.mouse_found_distribution[i].log_prob(found_mouse_i)[0] 
        return lp
    
    def update_estimation_local_observation(self, loc_obs):
        loc_agent_obs = loc_obs[0]
        for i in range(self.num_agents):
            if loc_agent_obs[i][0] != -1:
                self.agent_pos_distribution[i] = MultivariateNormal(torch.Tensor([loc_agent_obs[i][0], loc_agent_obs[i][1]]), torch.Tensor([[0.01, 0], [0, 0.01]]))
            else:
                self.agent_pos_distribution[i] = MultivariateNormal(self.agent_pos_distribution[i].mean, self.agent_pos_distribution[i].covariance_matrix + torch.Tensor([[1, 0], [0, 1]]))
        loc_mice_obs = loc_obs[1]
        for i in range(self.num_mice):
            if loc_mice_obs[i][0] != -1:
                self.mouse_pos_distribution[i] = MultivariateNormal(torch.Tensor([loc_mice_obs[i][0], loc_mice_obs[i][1]]), torch.Tensor([[0.01, 0], [0, 0.01]]))
            else:
                self.agent_pos_distribution[i] = MultivariateNormal(self.mouse_pos_distribution[i].mean, self.mouse_pos_distribution[i].covariance_matrix + torch.Tensor([[1, 0], [0, 1]]))
        for i in range(self.num_mice):
            if loc_mice_obs[i][2]:
                self.mouse_found_distribution[i] = Bernoulli(1)
            else:
                self.mouse_found_distribution[i] = Bernoulli(self.mouse_found_distribution[i].probs + 0.01)