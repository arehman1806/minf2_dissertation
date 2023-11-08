import gymnasium as gym
from stable_baselines3 import SAC
import underactuated_manipulation_gym
import time
import numpy as np

# IMPORTANT. edit this before every run:
tb_log_name = "first_run"

env = gym.make("queenie_gym_envs/RandomURDFsSOEnvironment-v0")
env.reset()
model = SAC("MultiInputPolicy", env, verbose=1, buffer_size=500000, tensorboard_log="./logs/simple_multi_object_pickup_agent")
model.learn(total_timesteps=500000, log_interval=10, tb_log_name=tb_log_name)
model.save("sac_queenie_multi_object")
model = SAC.load("sac_queenie", env=env)
vec_env = model.get_env()
obs= vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
