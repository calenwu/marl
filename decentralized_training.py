import decentralized_agent

class Decentralized:

    def __init__(self, state_dim, action_dim, num_agents, env):
        self.num_agents = num_agents
        self.env = env
        self.agents = []
        for i in range(num_agents):
            agent_i = decentralized_agent.Agent(state_dim=state_dim, action_dim=action_dim, num_agents=num_agents, agent_id=i)
            self.agents.append(agent_i)

    def get_actions(self, s):
        actions = []
        for ag in self.agents:
            a_i = ag.get_action(s)
            actions.append(a_i)
        return actions

    def train(self, num_iterations):
        s_t = self.env.get_s0()
        a_t = self.get_actions(s_t)
        
        for t in range(num_iterations):
            s_tn, rs = self.env.transition(s_t, a_t)

            for i in range(self.num_agents):
                self.agents[i].update_mu(rs[i])

            a_tn = self.get_actions(s_tn)

            for i in range(self.num_agents):
                self.agents[i].update_critic(s_tn, a_tn, rs[i])
                self.agents[i].update_actor(s_tn, a_tn, rs[i])

            con = self.env.get_connections(s_tn)

            for i in range(self.num_agents):
                omega_i = self.agents[i].get_omega()
                for j in range(con[i]):
                    omega_i += self.agents[j].get_omega()
                self.agents[i].set_omega(omega_i/(len(con[i])+1))
            
            s_t = s_tn
            a_t = a_tn