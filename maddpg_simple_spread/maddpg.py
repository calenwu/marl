from typing import Dict
from agent import Agent
from buffer import MultiAgentReplayBuffer


class MADDPG:
	def __init__(self, actor_dims, critic_dims, whole_state_observation_dims, n_agents, n_actions, env):
		self.memory = MultiAgentReplayBuffer(
			1_000_000, whole_state_observation_dims, actor_dims,
			n_actions, n_agents, batch_size=32, agent_names=env.agents)
		self.n_agents = n_agents
		self.n_actions = n_actions
		self.agents: Dict[str, Agent] = {}
		for agent_idx, agent_name in enumerate(env.possible_agents):
			self.agents[agent_name] = Agent(actor_dims[agent_idx],
				critic_dims,
				n_actions[agent_idx],
				agent_name = agent_name,
			)

	def save_checkpoint(self, t: str):
		print('... saving checkpoint ...')
		for _, agent in self.agents.items():
			agent.save_models(t)

	def load_checkpoint(self, t: str):
		print('... loading checkpoint ...')
		for _, agent in self.agents.items():
			agent.load_models(t)

	# def choose_action(self, raw_obs):
	# 	actions = []
	# 	for agent_name, agent in self.agents.items():
	# 		action = agent.choose_action(raw_obs[agent_name])
	# 		actions.append(action)
	# 	print(actions)
	# 	return actions

	def choose_action(self, raw_obs):
		actions = {agent.agent_name: agent.choose_action(raw_obs[agent.agent_name]) for agent in self.agents.values()}
		return actions

	def store_transition(self, obs, state, action, reward, obs_, state_, done):
		self.memory.store_transition(obs, state, action, reward, obs_, state_, done)

	def learn(self):
		for agent in self.agents.values():
			agent.learn(self.memory, self.agents)
