import os
from argparse import ArgumentParser

import gym
import matplotlib.pyplot as plt
import numpy as np
from agent import Agents
from buffer import ReplayBuffer
from worker import RolloutWorker


class Runner:
	def __init__(self, env, args):
		self.env = env

		# communication agent
		self.agents = Agents(args)
		self.rolloutWorker = RolloutWorker(env, self.agents, args)

		# coma doesnt use replay buffer, its on policy
		if args.alg.find('coma') == -1:
			self.buffer = ReplayBuffer(args)

		self.args = args
		self.save_path = self.args.result_dir + '/' + args.alg
		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)


	def run(self, num):
		plt.figure()
		plt.axis([0, self.args.n_steps, 0, 100])
		win_rates = []
		episode_rewards = []
		train_steps = 0
		time_steps = 0
		evaluate_steps = -1
		# win_rate, episode_reward = self.evaluate(epoch_num=100)
		while time_steps < self.args.n_steps:
			if time_steps // self.args.evaluate_cycle > evaluate_steps:
				print('Run {}, train step {}/{}'.format(num, time_steps, self.args.n_steps))
				win_rate, episode_reward = self.evaluate(epoch_num=time_steps)
				episode_rewards.append(episode_reward)
				win_rates.append(win_rate)

				plt.cla()
				plt.subplot(2, 1, 1)
				plt.plot(range(len(win_rates)), win_rates)
				plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
				plt.ylabel('win_rate')

				plt.subplot(2, 1, 2)
				plt.plot(range(len(episode_rewards)), episode_rewards)
				plt.xlabel('step*{}'.format(self.args.evaluate_cycle))
				plt.ylabel('episode_rewards')

				plt.savefig(self.save_path + '/plt_{}_{}_{}ts.png'.format(num, self.args.env, self.args.n_steps), format='png')
				np.save(self.save_path + '/episode_rewards_{}_{}_{}ts'.format(num, self.args.env, self.args.n_steps), episode_rewards)
				np.save(self.save_path + '/win_rates_{}_{}_{}ts'.format(num, self.args.env, self.args.n_steps), win_rates)

				evaluate_steps += 1

			episodes = []

			for episode_idx in range(self.args.n_episodes):
				episode, _, _, info = self.rolloutWorker.generate_episode(episode_idx)
				episodes.append(episode)
				time_steps += info['steps_taken']

			episode_batch = episodes[0]
			episodes.pop(0)

			# put observations of all the generated epsiodes together
			for episode in episodes:
				for key in episode_batch.keys():
					episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

			self.buffer.store_episode(episode_batch)
			for train_step in range(self.args.train_steps):
				mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
				self.agents.train(mini_batch, train_steps)
				train_steps += 1

		self.agents.policy.save_model(train_steps)

		plt.cla()
		plt.subplot(2, 1, 1)
		plt.plot(range(len(win_rates)), win_rates)
		plt.xlabel('steps*{}'.format(self.args.evaluate_cycle))
		plt.ylabel('win_rate')

		plt.subplot(2, 1, 2)
		plt.plot(range(len(episode_rewards)), episode_rewards)
		plt.xlabel('steps*{}'.format(self.args.evaluate_cycle))
		plt.ylabel('episode_rewards')

		plt.savefig(self.save_path + '/plt_{}_{}_{}ts.png'.format(num, self.args.env, self.args.n_steps), format='png')
		np.save(self.save_path + '/episode_rewards_{}_{}_{}ts'.format(num, self.args.env, self.args.n_steps), episode_rewards)
		np.save(self.save_path + '/win_rates_{}_{}_{}ts'.format(num, self.args.env, self.args.n_steps), win_rates)

	def evaluate(self, epoch_num=None):
		win_counter = 0
		episode_rewards = 0
		steps_avrg = 0
		for epoch in range(self.args.evaluate_epoch):
			_, episode_reward, won, info = self.rolloutWorker.generate_episode(epoch, evaluate=True, epoch_num=epoch_num)
			episode_rewards += episode_reward
			steps_avrg += info['steps_taken']
			if won:  # if env ended in winning state
				win_counter += 1
		return win_counter / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch



def common_args():
	parser = ArgumentParser()
	# general args
	parser.add_argument("--env", "-e", default="Lumberjacks-v0", help="set env name")
	parser.add_argument("--n_steps", "-ns", type=int, default=600000, help="set total time steps to run")
	parser.add_argument("--n_episodes", "-nep", type=int, default=1, help="set n_episodes")
	parser.add_argument("--epsilon", "-eps", default=0.5, help="set epsilon value")
	parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
	parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
	parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
	parser.add_argument('--evaluate_epoch', type=int, default=20, help='the number of the epoch to evaluate the agent')
	parser.add_argument('--alg', type=str, default='qmix', help='the algorithm to train the agent')
	parser.add_argument('--optimizer', type=str, default="RMS", help='the optimizer')
	parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
	parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
	parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
	parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
	parser.add_argument('--evaluate_cycle', type=int, default=5000, help='how often to eval the model')
	parser.add_argument('--target_update_cycle', type=int, default=200, help='how often to update the target network')
	parser.add_argument('--save_cycle', type=int, default=100000, help='how often to save the model')
	parser.add_argument('--cuda', type=bool, default=False, help='whether to use the GPU')

	args = parser.parse_args()

	# buffer/batch sizes
	args.batch_size = 32
	args.buffer_size = int(5e3)

	args.epsilon = 1
	args.min_epsilon = 0.05
	anneal_steps = 50000
	args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
	args.epsilon_anneal_scale = 'step'

	# netowrk args for vdn
	# network
	args.rnn_hidden_dim = 64
	args.qmix_hidden_dim = 32
	args.two_hyper_layers = False
	args.hyper_hidden_dim = 64
	args.qtran_hidden_dim = 64
	args.lr = 5e-4

	# train steps for vdn
	args.train_steps = 1

	# prevent gradient explosion
	args.grad_norm_clip = 10

	# QTRAN lambda
	args.lambda_opt = 1
	args.lambda_nopt = 1

	return args

if __name__ == '__main__':
	args = common_args()
	env = gym.make('ma_gym:Lumberjacks-v0', grid_shape=(8,8), n_agents=4)
	args.n_actions = env.action_space[0].n
	args.n_agents = env.n_agents
	args.state_shape = 22 * args.n_agents
	args.obs_shape = 22
	args.episode_limit = env._max_steps

	runner = Runner(env, args)
	# parameterize run according to the number of independent experiments to run, i.e., independent sets of n_epochs over the model; default is 1
	if args.learn:
		runner.run(1)
