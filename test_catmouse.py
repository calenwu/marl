from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import *
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse import *

env = CatMouseMA()

state = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    # communication_list only returned for CatMouseMA
    next_state, communication_list, reward, terminated, info = env.step(action)
    state = next_state
    env.render()

env.close()
    
    