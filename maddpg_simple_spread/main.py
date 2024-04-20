import time
import warnings

import numpy as np
from pettingzoo.mpe import simple_spread_v3
from tqdm import tqdm

from maddpg import MADDPG

LOAD_TYPE = ['Regular', 'Best']  # Regular: save every 10k, Best: save only if avg_score > best_score
PRINT_INTERVAL = 500
SAVE_INTERVAL = 5000
MAX_STEPS = 25
EPISODES = 60_000


def obs_list_to_state_vector(observation):
	"""
	Convert a list of observations to a state vector by concatenating them.
	"""
	state = np.array([])
	for obs in observation:
		state = np.concatenate([state, obs])
	return state


def visualize_agents(agents, env, n_episodes=20, speed=0.1):
	"""
	Visualize the agents' behavior in the environment.
	"""
	# Ensure speed is between 0 and 1
	speed = np.clip(speed, 0, 1)

	for episode in range(n_episodes):
		prev_reward = -np.inf
		obs, _ = env.reset()
		terminal = [False] * env.num_agents

		while not any(terminal):
			actions = agents.choose_action(obs)
			obs, rewards, done, truncation, _ = env.step(actions)

			# Sum rewards
			rewards = sum(rewards.values())

			# Determine direction
			direction = "Right direction" if rewards > prev_reward else "Wrong direction"
			prev_reward = rewards

			# Render as an RGB array
			img = env.render()
			time.sleep(0.05)

			# Clear the current axes and plot the new image
			# plt.clf()
			# plt.imshow(img)

			# Determine the center position for the text
			# center_x = img.shape[1] / 2

			# Add direction text to the figure, centered horizontally
			# plt.text(center_x, 20, direction, fontsize=12, color='white', bbox=dict(facecolor='black', alpha=1), ha='center')

			# Display the updated figure
			# display.clear_output(wait=True)
			# display.display(plt.gcf())
			# plt.pause(0.1 / speed)

			terminal = [d or t for d, t in zip(done.values(), truncation.values())]

		print(f'Episode {episode + 1} completed')


def solve_env_with_subpolicies(env, evaluate: bool=False):
	"""
	Solve the environment using subpolicies.
	"""
	obs = env.reset()

	n_agents = env.num_agents
	actor_dims = [env.observation_spaces[agent_name].shape[0] for agent_name in env.agents]
	n_actions = [env.action_spaces[agent_name].shape[0] for agent_name in env.agents]
	critic_dims = sum(actor_dims) + sum(n_actions)
	# what everyone is seeing
	whole_state_observation_dims = sum(actor_dims)

	maddpg_agents = MADDPG(
		actor_dims, critic_dims, whole_state_observation_dims, n_agents, n_actions, env=env)

	total_steps = 0
	score_history = []
	score_history_100 = []
	best_score = - np.inf  #the first score will always be better than this
	epsiode_mean_agent_rewards = {agent_name: [] for agent_name in env.agents}
	episode_lengths = []

	# maddpg_agents.load_checkpoint(LOAD_TYPE[0])  # load best
	if evaluate:
		maddpg_agents.load_checkpoint(LOAD_TYPE[0])  # load best
		visualize_agents(maddpg_agents, env, n_episodes=20, speed=10)
	else:
		for i in tqdm(range(EPISODES), desc='Training'):
			obs, _ = env.reset()
			score = 0
			done = [False] * n_agents
			episode_step = 0
			episode_length = 0
			agent_rewards = {agent_name: [] for agent_name in env.agents}

			while not any(done):
				actions = maddpg_agents.choose_action(obs)

				obs_, reward, termination, truncation, _ = env.step(actions)
				state = np.concatenate([i for i in obs.values()])
				state_ = np.concatenate([i for i in obs_.values()])

				if episode_step >= MAX_STEPS:
					done = [True] * n_agents

				if any(termination.values()) or any(truncation.values()) or (episode_step >= MAX_STEPS):
					done = [True] * n_agents

				maddpg_agents.store_transition(obs, state, actions, reward, obs_, state_, done)

				if total_steps % 5 == 0:
					maddpg_agents.learn()

				obs = obs_
				for agent_name, r in reward.items():
					agent_rewards[agent_name].append(r)

				score += sum(reward.values())
				total_steps += 1
				episode_step += 1
				episode_length += 1

			score_history.append(score)
			avg_score = np.mean(score_history[-100:])
			score_history_100.append(avg_score)
			episode_lengths.append(episode_length)

			if (avg_score > best_score) and (i > PRINT_INTERVAL):
				print(' avg_score, best_score', avg_score, best_score)
				maddpg_agents.save_checkpoint(LOAD_TYPE[1])
				best_score = avg_score

			if i % SAVE_INTERVAL == 0 and i > 0:
				maddpg_agents.save_checkpoint(LOAD_TYPE[0])

			# Compute mean agent rewards
			for agent_name, rewards in agent_rewards.items():
				mean_agent_reward = sum(rewards)
				epsiode_mean_agent_rewards[agent_name].append(mean_agent_reward)


if __name__ == '__main__':
	warnings.filterwarnings('ignore')

	env = simple_spread_v3.parallel_env(max_cycles=25, continuous_actions=True)
	# env = simple_spread_v3.parallel_env(max_cycles=25, continuous_actions=True, render_mode='human')
	# env = simple_spread_v3.parallel_env(max_cycles=25, continuous_actions=True, render_mode='rgb_array')

	solve_env_with_subpolicies(env, evaluate=False)
