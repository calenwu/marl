import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from buffer import Buffer
from mappo_mpe import MAPPO_MPE
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import CatMouseMA



def make_env(episode_limit, render_mode='None'):
	# env = simple_spread_v3.parallel_env(N=3, max_cycles=episode_limit,
	# 									local_ratio=0.5, render_mode=render_mode, continuous_actions=False)
	# env.reset(seed=42)
	env = CatMouseMA(observation_radius=10)
	env.reset()
	return env


eps = []
rewards = []


class Runner_MAPPO_MPE:
	def __init__(self, args, env_name, number, seed):
		self.args = args
		self.env_name = env_name
		self.number = number

		# Set random seed
		self.seed = seed
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)

		# Create env
		self.env = make_env(self.args.episode_limit, render_mode=args.render_mode)  # Discrete action space
		# self.args.N = self.env.max_num_agents  # The number of agents
		# self.args.obs_dim_n = [self.env.observation_spaces[agent].shape[0] for agent in self.env.agents]  # obs dimensions of N agents
		# self.args.action_dim_n = [self.env.action_spaces[agent].n for agent in self.env.agents]  # actions dimensions of N agents
		self.args.N = self.env.n_agents
		self.args.obs_dim_n = [(self.env.n_agents * 3 + self.env.n_prey * 3) for agent in range(self.args.N)]
		self.args.action_dim_n = [4 for agent in range(self.args.N)]
		# Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
		self.args.obs_dim = self.args.obs_dim_n[0]  # The dimensions of an agent's observation space
		self.args.action_dim = self.args.action_dim_n[0]  # The dimensions of an agent's action space
		self.args.state_dim = 2 * self.env.n_agents + 3 * self.env.n_prey  # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
		print('observation_space=', self.env.observation_space)
		print('obs_dim_n={}'.format(self.args.obs_dim_n))
		print('action_space=', self.env.action_space)
		print('action_dim_n={}'.format(self.args.action_dim_n))

		# Create N agents
		self.agent_n = MAPPO_MPE(self.args)
		self.bufer = Buffer(self.args)

		# Create a tensorboard
		self.writer = SummaryWriter(
			log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

		self.evaluate_rewards = []  # Record the rewards during the evaluating
		self.total_steps = 0
		if self.args.use_reward_norm:
			print('------use reward norm------')
			self.reward_norm = Normalization(shape=self.args.N)
		elif self.args.use_reward_scaling:
			print('------use reward scaling------')
			self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

	def run(self, ):
		while self.total_steps < self.args.max_train_steps:
			if self.total_steps % self.args.evaluate_freq == 0:
				self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps

			_, episode_steps = self.run_episode_mpe()  # Run an episode
			self.total_steps += episode_steps

			if self.bufer.episode_num == self.args.batch_size:
				self.agent_n.train(self.bufer, self.total_steps)  # Training
				self.bufer.reset_buffer()

		self.evaluate_policy()
		self.env.close()

		plt.figure(figsize=(10, 5))
		plt.plot(eps, rewards)
		plt.xlabel('Episodes')
		plt.ylabel('Reward')
		plt.title('Reward vs Episodes')
		plt.grid(True)

		# Save the plot to a file
		plt.savefig('reward_vs_episodes.png')

		# Storing the data in a .csv file
		data = {'Episodes': eps, 'Reward': rewards}
		df = pd.DataFrame(data)
		df.to_csv('reward_vs_episodes.csv', index=False)


	def evaluate_policy(self):
		evaluate_reward = 0
		for _ in range(self.args.evaluate_times):
			episode_reward, _ = self.run_episode_mpe(evaluate=True)
			evaluate_reward += episode_reward

		evaluate_reward = evaluate_reward / self.args.evaluate_times
		self.evaluate_rewards.append(evaluate_reward)

		rewards.append(evaluate_reward)
		eps.append(self.total_steps)

		print('total_steps:{} \t evaluate_reward:{}'.format(self.total_steps, evaluate_reward))
		self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward,
							   global_step=self.total_steps)
		# Save the rewards and models
		np.save('./train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed),
				np.array(self.evaluate_rewards))
		self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

	def trans_obs(self, obs):
		ret = []
		for agent_obs in obs:
			temp = []
			temp.append(agent_obs['agents']['cur_agent'])
			for agent_pos in agent_obs['agents']['position']:
				temp.append(agent_pos)
			for prey_pos in agent_obs['prey']['position']:
				temp.append(prey_pos)
			temp.append(agent_obs['prey']['caught'])
			ret.append(np.concatenate(temp))
		return np.array(ret)

	def trans_state(self, state):
		ret = []
		for agent_pos in state['agents']['position']:
			ret += agent_pos.tolist()
		for i, prey_pos in enumerate(state['prey']['position']):
			ret += prey_pos.tolist()
			ret.append(state['prey']['caught'][i])
		return np.array(ret)
	def run_episode_mpe(self, evaluate=False):
		action_dict = {
			0: 0,
			1: 0.25,
			2: 0.5,
			3: 0.75
		}
		episode_reward = 0
		observations, info = self.env.reset()

		# obs_n = np.array([observations[agent] for agent in observations.keys()])
		obs_n = np.array(self.trans_obs(observations))
		if self.args.use_reward_scaling:
			self.reward_scaling.reset()
		if self.args.use_rnn:  # If you use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
			self.agent_n.actor.rnn_hidden = None
			self.agent_n.critic.rnn_hidden = None
		for episode_step in range(self.args.episode_limit):
			a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
			s = self.trans_state(self.env.get_global_obs())  # In MPE, global state is the concatenation of all agents' local obs.
			v_n = self.agent_n.get_value(s)  # Get the state values (V(s)) of N agents

			# need to transit 'a_n' into dict

			obs_next_n, r_n, done_n, _, _ = self.env.step(np.array([action_dict[x] for x in a_n]))
			obs_next_n = self.trans_obs(obs_next_n)

			episode_reward += r_n[0]

			if not evaluate:
				if self.args.use_reward_norm:
					r_n = self.reward_norm(r_n)
				elif self.args.use_reward_scaling:
					r_n = self.reward_scaling(r_n)

				# Store the transition
				self.bufer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

			obs_n = np.array(obs_next_n)
	 
			if done_n:
				break

		if not evaluate:
			# An episode is over, store v_n in the last step
			s = self.trans_state(self.env.get_global_obs())
			v_n = self.agent_n.get_value(s)
			self.bufer.store_last_value(episode_step + 1, v_n)

		return episode_reward, episode_step + 1


if __name__ == '__main__':
	parser = argparse.ArgumentParser('Hyperparameters Setting for MAPPO in MPE environment')
	parser.add_argument('--max_train_steps', type=int, default=int(1e7), help=' Maximum number of training steps')
	parser.add_argument('--episode_limit', type=int, default=25, help='Maximum number of steps per episode')
	parser.add_argument('--evaluate_freq', type=float, default=int(5000),
						help='Evaluate the policy every "evaluate_freq" steps')
	parser.add_argument('--evaluate_times', type=float, default=3, help='Evaluate times')

	parser.add_argument('--batch_size', type=int, default=32, help='Batch size (the number of episodes)')
	parser.add_argument('--mini_batch_size', type=int, default=8, help='Minibatch size (the number of episodes)')
	parser.add_argument('--rnn_hidden_dim', type=int, default=64,
						help='The number of neurons in hidden layers of the rnn')
	parser.add_argument('--mlp_hidden_dim', type=int, default=64,
						help='The number of neurons in hidden layers of the mlp')
	parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
	parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
	parser.add_argument('--lamda', type=float, default=0.95, help='GAE parameter')
	parser.add_argument('--epsilon', type=float, default=0.2, help='GAE parameter')
	parser.add_argument('--K_epochs', type=int, default=15, help='GAE parameter')
	parser.add_argument('--use_adv_norm', type=bool, default=True, help='Trick 1:advantage normalization')
	parser.add_argument('--use_reward_norm', type=bool, default=True, help='Trick 3:reward normalization')
	parser.add_argument('--use_reward_scaling', type=bool, default=False,
						help='Trick 4:reward scaling. Here, we do not use it.')
	parser.add_argument('--entropy_coef', type=float, default=0.01, help='Trick 5: policy entropy')
	parser.add_argument('--use_lr_decay', type=bool, default=True, help='Trick 6:learning rate Decay')
	parser.add_argument('--use_grad_clip', type=bool, default=True, help='Trick 7: Gradient clip')
	parser.add_argument('--use_orthogonal_init', type=bool, default=True, help='Trick 8: orthogonal initialization')
	parser.add_argument('--set_adam_eps', type=float, default=True, help='Trick 9: set Adam epsilon=1e-5')
	parser.add_argument('--use_relu', type=float, default=False, help='Whether to use relu, if False, we will use tanh')
	parser.add_argument('--use_rnn', type=bool, default=False, help='Whether to use RNN')
	parser.add_argument('--add_agent_id', type=float, default=False,
						help='Whether to add agent_id. Here, we do not use it.')
	parser.add_argument('--use_value_clip', type=float, default=False, help='Whether to use value clip.')
	parser.add_argument('--render_mode', type=str,
						default='None', help='File path to my result')

	args = parser.parse_args()
	runner = Runner_MAPPO_MPE(args, env_name='simple_spread_v3', number=3, seed=0)
	runner.run()
