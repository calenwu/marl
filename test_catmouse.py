from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import *
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse import *

env = CatMouseMA()

state, info = env.reset()
for _ in range(100):
    print(state)
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state
    env.render()

env.close()
    
    