import underactuated_manipulation_gym
import gymnasium as gym
import time
import numpy as np
env = gym.make("queenie_gym_envs/RandomURDFsSOEnvironment-v0")
env.reset()
j = 0
for i in range(1000000):
    action = env.action_space.sample()
    action = np.array([0.0, 0.0, 0.0, 0.0, 1])
    obs, reward, done, _, _ = env.step(action)
    if done:
        print(f"done {j}")
        env.reset()
        j += 1
    # time.sleep(1)