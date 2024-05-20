import time
import gym
import numpy as np
from agent import Agent
import matplotlib.pyplot as plt
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import CatMouseMA
import torch


def trans_obs(obs):
	# ret = []
	# for agent_obs in obs:
	# 	temp = []
	# 	# temp.append(agent_obs['agents']['cur_agent'])
	# 	for agent_pos in agent_obs['agents']['position']:
	# 		temp.append(agent_pos)
	# 	for prey_pos in agent_obs['prey']['position']:
	# 		temp.append(prey_pos)
	# 	# temp.append(agent_obs['prey']['caught'])
	# 	ret.append(np.concatenate(temp))
	temp = []
	temp.append(obs['agents']['position'][0] - obs['prey']['position'][0])
	return np.array(temp)

def trans_state(state):
	ret = []
	for agent_pos in state['agents']['position']:
		ret += agent_pos.tolist()
	for i, prey_pos in enumerate(state['prey']['position']):
		ret += prey_pos.tolist()
		# ret.append(state['prey']['caught'][i])
	# return np.array(ret)
	return (state['agents']['position'] - state['prey']['position'])[0]

def get_action(action):
	action_dict = {
		0: 0,
		1: 0.25,
		2: 0.5,
		3: 0.75
	}
	return action_dict[action]

def make_env(episode_limit, render_mode='None'):
	env = CatMouseMA(observation_radius=1, n_agents=1, n_prey=1)
	env.obs_dim = 2
	env.state_dim = 2
	# env.obs_dim = env.n_agents * 3 + env.n_prey * 3
	# env.state_dim = 2 * env.n_agents + 3 * env.n_prey
	env.action_dim = 4
	env.reset()
	return env


def plot_learning_curve(x, scores, figure_file):
	running_avg = np.zeros(len(scores))
	for i in range(len(running_avg)):
		running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
	plt.plot(x, running_avg)
	plt.title('Running average of previous 100 scores')
	plt.savefig(figure_file)


if __name__ == '__main__':
	# env = gym.make('CartPole-v0')
	env = make_env(1000)
	N = 20
	batch_size = 16
	n_epochs = 4
	alpha = 0.0003
	agent = Agent(n_actions=env.action_dim, input_dims=[env.obs_dim], alpha=alpha, gamma=0.99,
				  n_epochs=n_epochs, batch_size=batch_size)
	n_games = 100000
	# agent.load_models()
	figure_file = 'plots/cartpole.png'
	best_score = -1000
	score_history = []

	learn_iters = 0
	avg_score = 0
	n_steps = 0

	for i in range(n_games):
		done = False
		observation, _ = env.reset()
		# observation = np.array(trans_obs(observation))
		observation = trans_obs(observation[0])
		score = 0
		steps = 0
		while not done and steps < N:
			action, prob, val, _ = agent.choose_action(observation[0])
			observation_, reward, done, _, info = env.step([action])
			# observation_ = np.array(trans_obs(observation_))
			observation_ = trans_obs(observation_[0])
			n_steps += 1
			score += reward
			if steps > 500000 == 0:
				env.render()
			agent.remember(observation, action, prob, val, reward, done)
			if n_steps % N == 0:
				agent.learn()
				learn_iters += 1
			observation = observation_
			steps += 1
		score_history.append(score)
		avg_score = np.mean(score_history[-100:])
		if avg_score > best_score:
			best_score = avg_score
			agent.save_models()
		if i % 500 == 0:
			print('0.5, 0.5: ', agent.critic(torch.tensor([0.5, 0.5], dtype=torch.float32)).tolist())
			print('probs: ', agent.choose_action(np.array([0.5, 0.5]))[3].tolist())
			print('0.5, -0.5: ', agent.critic(torch.tensor([0.5, -0.5], dtype=torch.float32)).tolist())
			print('probs: ', agent.choose_action(np.array([0.5, -0.5]))[3].tolist())
			print('-0.5, 0.5: ', agent.critic(torch.tensor([-0.5, 0.5], dtype=torch.float32)).tolist())
			print('probs: ', agent.choose_action(np.array([-0.5, 0.5]))[3].tolist())
			print('-0.5, -0.5: ', agent.critic(torch.tensor([-0.5, -0.5], dtype=torch.float32)).tolist())
			print('probs: ', agent.choose_action(np.array([-0.5, -0.5]))[3].tolist())
			print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
					'time_steps', n_steps, 'learning_steps', learn_iters)
	x = [i + 1 for i in range(len(score_history))]
	plot_learning_curve(x, score_history, figure_file)


	# observation = env.reset()
	# done = False
	# while not done:
	# 	action, prob, val = agent.choose_action(observation)
	# 	observation_, reward, done, info = env.step(action)
	# 	env.render()
	# 	time.sleep(0.01)
	# 	observation = observation_