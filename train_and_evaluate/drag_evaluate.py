import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
sys.path.append("/home/arehman/dissertation/")
import underactuated_manipulation_gym
import time
import numpy as np
import pybullet as p


from rule_based_policies.drag import DragPolicy

env = gym.make("queenie_gym_envs/DragEnvironment-v0", config_file="./underactuated_manipulation_gym/resources/config/environment_config/drag_environment.yaml")
env = DummyVecEnv([lambda: env])
obs = env.reset()
model = DragPolicy(env)
for i in range(1000):
    action, _state = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        env.reset()
