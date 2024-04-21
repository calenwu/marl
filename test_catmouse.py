from marl_gym.marl_gym.envs.cat_mouse.cat_mouse_ma import *
from marl_gym.marl_gym.envs.cat_mouse.cat_mouse import *
import time

env = CatMouseMA()

state = env.reset()
for _ in range(100):
    print(state)
    action = env.action_space.sample()
    next_state, reward, terminated, info = env.step(action)
    state = next_state
    print(reward)
    time.sleep(1)
    
    