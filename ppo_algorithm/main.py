import time
import gym
import numpy as np
from agent import Agent
import matplotlib.pyplot as plt
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import CatMouseMA


d = []
for x in range(5):
	for y in range(5):
		d.append([x, y])

def make_env(episode_limit, render_mode='None'):
	env = CatMouseMA(observation_radius=1, n_agents=1, n_prey=1)
	env.obs_dim = env.n_agents * 3 + env.n_prey * 3
	env.state_dim = 2 * env.n_agents + 3 * env.n_prey
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
	env = gym.make('ma_gym:Lumberjacks-v1', grid_shape=(5, 5), n_agents=2)
	N = 20
	batch_size = 16
	n_epochs = 4
	alpha = 0.0003
	agent = Agent(
		n_actions=25,
		input_dims=44,
		alpha=alpha,
		gamma=0.99,
		n_epochs=n_epochs,
		batch_size=batch_size)
	n_games = 5000
	agent.load_models()
	figure_file = 'plots/cartpole.png'
	best_score = -100
	score_history = []

	learn_iters = 0
	avg_score = 0
	n_steps = 0

	for i in range(n_games):
		done = False
		observation = env.reset()
		observation = np.array(observation[0])
		score = 0
		while not done:
			action, prob, val = agent.choose_action(observation)
			observation_, reward, done, info = env.step(d[action])
			env.render()
			time.sleep(0.25)
			observation_ = np.array(observation_[0])
			reward = sum(reward)
			done = all(done)
			n_steps += 1
			score += reward
			agent.remember(observation, action, prob, val, reward, done)
			if n_steps % N == 0:
				agent.learn()
				learn_iters += 1
			observation = observation_
		score_history.append(score)
		avg_score = np.mean(score_history[-100:])
		if avg_score > best_score:
			best_score = avg_score
			agent.save_models()
		print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
			  'time_steps', n_steps, 'learning_steps', learn_iters)
	x = [i + 1 for i in range(len(score_history))]
	# plot_learning_curve(x, score_history, figure_file)


	# observation = env.reset()
	# done = False
	# while not done:
	# 	action, prob, val = agent.choose_action(observation)
	# 	observation_, reward, done, info = env.step(action)
	# 	env.render()
	# 	time.sleep(0.01)
	# 	observation = observation_