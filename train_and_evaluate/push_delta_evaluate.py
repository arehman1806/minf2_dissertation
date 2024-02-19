import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
sys.path.append("/home/arehman/dissertation/")
import underactuated_manipulation_gym
import time
import numpy as np
import pybullet as p


from rule_based_policies.push_delta import PushDeltaPolicy

env = gym.make("queenie_gym_envs/PushDeltaEnvironment-v0", config_file="./underactuated_manipulation_gym/resources/config/environment_config/push_delta_environment.yaml")
env = DummyVecEnv([lambda: env])
obs = env.reset()
model = PushDeltaPolicy(env)
for i in range(10000):
    action, _state = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        env.reset()
