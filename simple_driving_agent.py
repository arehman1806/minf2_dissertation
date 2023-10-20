import gymnasium as gym
from stable_baselines3 import SAC
import underactuated_manipulation_gym
import time
import numpy as np

env = gym.make("queenie_gym_envs/DifferentialDriveEnv-v0")
env.reset()
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
model.save("sac_queenie")