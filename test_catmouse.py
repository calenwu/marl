from marl_gym.marl_gym.envs.cat_mouse.cat_mouse import *
import time

env = CatMouse()

state = env.reset()
for _ in range(100):
    print(state)
    action = env.action_space.sample()
    next_state, reward, terminated, info = env.step(action)
    time.sleep(1)
    
    