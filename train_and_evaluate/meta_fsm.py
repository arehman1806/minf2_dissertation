import gymnasium as gym


import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
# from video_record_callback import VideoRecorderCallback
import time
import numpy as np
import sys
sys.path.append("/home/arehman/dissertation/")
import underactuated_manipulation_gym
from rule_based_policies.meta import FSMMetaPolicy

NUM_TRAILS = 10

env = gym.make("queenie_gym_envs/MetaEnvironment-v0", config_file="./underactuated_manipulation_gym/resources/config/environment_config/meta_environment.yaml")
env = Monitor(env)
env = DummyVecEnv([lambda: env])

obs = env.reset()

policy_kwargs = dict(
    net_arch=dict(pi=[512, 512, 512, 512], vf=[512, 512, 512, 512])
)
model = FSMMetaPolicy(env, policy_kwargs)

for trial in range(NUM_TRAILS):
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            print(f"Trial {trial} done.")