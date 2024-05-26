import sys
import torch
from torch.distributions import Normal, Bernoulli, MultivariateNormal

class Coop_Nav_State_Distribution(torch.distributions.distribution.Distribution):

    def __init__(self, num_agents, num_targets, agent_id):
        super().__init__()
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.agent_id = agent_id
        self.ini_var = 10
        self.agent_pos_distribution = []
        self.target_pos_distribution = []
        for i in range(self.num_agents):
            position_mean = torch.Tensor([0.5, 0.5])
            position_var = torch.Tensor([[self.ini_var, 0],[0, self.ini_var]])
            agent_pos_i_distribution = MultivariateNormal(position_mean, position_var)
            self.agent_pos_distribution.append(agent_pos_i_distribution)
        
        for i in range(self.num_targets):
            position_mean = torch.Tensor([0.5, 0.5])
            position_var = torch.Tensor([[self.ini_var, 0],[0, self.ini_var]])
            target_pos_i_distribution = MultivariateNormal(position_mean, position_var)
            self.target_pos_distribution.append(target_pos_i_distribution)

    def log_prob(self, glob_state):
        lp = 0
        for i in range(self.num_agents):
            pos_agent_i = torch.Tensor(glob_state['agents']['position'][i])
            lp += self.agent_pos_distribution[i].log_prob(pos_agent_i)
        for i in range(self.num_targets):
            pos_target_i = torch.Tensor(glob_state['prey']['position'][i])
            lp += self.mouse_pos_distribution[i].log_prob(pos_target_i)
        return lp

    def update_estimation_local_observation(self, loc_obs):
        loc_obs = loc_obs["agent_0"]
        for i in range(self.num_agents):
            self.agent_pos_distribution[i] = MultivariateNormal(torch.Tensor([loc_obs[2], loc_obs[3]]), torch.Tensor([[0.01, 0], [0, 0.01]]))

        for i in range(self.num_targets):
            self.target_pos_distribution[i] = MultivariateNormal(torch.Tensor([loc_obs[2] + loc_obs[4], loc_obs[3] + loc_obs[5]]), torch.Tensor([[0.01, 0], [0, 0.01]]))

    @staticmethod
    def update_estimation_communication(self, distributions):
        num_comm = len(distributions)
        for i in range(self.num_agents):
            mean = torch.Tensor([0,0])
            cov_sum = 0
            min_cov = self.ini_var
            for j in range(len(num_comm)):
                cov = distributions[j].agent_pos_distribution[i].covariance_matrix[0][0]
                mean += distributions[j].agent_pos_distribution[i].mean/cov
                cov_sum += 1/cov
                min_cov += min(min_cov, cov)
            for j in range(len(num_comm)):
                distributions[j].agent_pos_distribution[i] = MultivariateNormal(mean/cov_sum, torch.Tensor([[min_cov, 0],[0, min_cov]]))

        for i in range(self.num_targets):
            mean = torch.Tensor([0,0])
            cov_sum = 0
            min_cov = self.ini_var
            for j in range(len(num_comm)):
                cov = distributions[j].target_pos_distribution[i].covariance_matrix[0][0]
                mean += distributions[j].target_pos_distribution[i].mean/cov
                cov_sum += 1/cov
                min_cov += min(min_cov, cov)
            for j in range(len(num_comm)):
                distributions[j].target_pos_distribution[i] = MultivariateNormal(mean/cov_sum, torch.Tensor([[min_cov, 0],[0, min_cov]]))

    def get_belief_state(self):
        state = []
        for i in range(self.num_agents):
            state.append(self.agent_pos_distribution[i].mean[0])
            state.append(self.agent_pos_distribution[i].mean[1])
            # state.append(self.agent_pos_distribution[i].covariance_matrix[0][0])
            # state.append(self.agent_pos_distribution[i].covariance_matrix[1][1])
        for i in range(self.num_targets):
            state.append(self.target_pos_distribution[i].mean[0])
            state.append(self.target_pos_distribution[i].mean[1])
            # state.append(self.target_pos_distribution[i].covariance_matrix[0][0])
            # state.append(self.target_pos_distribution[i].covariance_matrix[1][1])
        return state
        