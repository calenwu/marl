import time
import gym
import numpy as np
from agent import Agent
import matplotlib.pyplot as plt
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import CatMouseMA
from pettingzoo.mpe import simple_spread_v3


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
	# env = gym.make('ma_gym:Lumberjacks-v1', grid_shape=(5, 5), n_agents=2)

	env = simple_spread_v3.parallel_env(N=2, max_cycles=25, local_ratio=0.5, continuous_actions=False, render_mode='human')
	N = 20
	batch_size = 16
	n_epochs = 4
	alpha = 0.0003
	agent_1 = Agent(
		n_actions=5,
		input_dims=12,
		alpha=alpha,
		gamma=0.99,
		n_epochs=n_epochs,
		batch_size=batch_size)
	agent_2 = Agent(
		n_actions=5,
		input_dims=12,
		alpha=alpha,
		gamma=0.99,
		n_epochs=n_epochs,
		batch_size=batch_size)
	n_games = 20000
	
	agent_1.load_models(path='agent_1.pth')
	agent_2.load_models(path='agent_2.pth')

	best_score = -100
	score_history = []

	learn_iters = 0
	avg_score = 0
	n_steps = 0

	for i in range(n_games):
		done = [False]
		observation, _ = env.reset()
		observation = list(observation.values())
		observation_1 = observation[0]
		observation_2 = observation[1]
		score = 0
		steps = 0
		while not all(done) and steps < 25:
			action_1, prob_1, val_1 = agent_1.choose_action(observation_1)
			action_2, prob_2, val_2 = agent_2.choose_action(observation_2)

			actions_dict = {}
			actions = [action_1, action_2]
			for agent_id, ag in enumerate(env.agents):
				actions_dict[ag] = actions[agent_id]
			observation_, reward, done, _, info = env.step(actions_dict)
			observation_ = list(observation_.values())
			reward = list(reward.values())
			done = list(done.values())

			observation_1 = observation_[0]
			observation_2 = observation_[1]

			reward_1 = reward[0]
			reward_2 = reward[1]

			done_1 = done[0]
			done_2 = done[1]

			env.render()
			time.sleep(0.25)

			n_steps += 1
			steps += 1

			score += sum(reward)
			agent_1.remember(observation_1, action_1, prob_1, val_1, reward_1, done_1)
			agent_2.remember(observation_2, action_2, prob_2, val_2, reward_2, done_2)
			if n_steps % N == 0:
				agent_1.learn()
				agent_2.learn()
				learn_iters += 1

			observation_1 = observation_[0]
			observation_2 = observation_[1]

		score_history.append(score)
		avg_score = np.mean(score_history[-100:])
		if avg_score > best_score:
			best_score = avg_score
			# agent_1.save_models('agent_1.pth')
			# agent_2.save_models('agent_2.pth')
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