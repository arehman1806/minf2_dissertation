import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import sys
sys.path.append("/home/arehman/dissertation/")
import underactuated_manipulation_gym
# from video_record_callback import VideoRecorderCallback
import time
import numpy as np

# IMPORTANT. edit this before every run:
tb_log_name = "hha_single_object_finetune"
# tb_log_name = "testing_videos"
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=50000,
  save_path=f"./runs/grasp/{tb_log_name}/",
  name_prefix=tb_log_name,
  save_replay_buffer=False,
  save_vecnormalize=False,
)

env = gym.make("queenie_gym_envs/GraspEnvironment-v1", config_file="./underactuated_manipulation_gym/resources/config/environment_config/grasp_environment_1.yaml")
env = Monitor(env)
env = DummyVecEnv([lambda: env])
# env = VecNormalize(env, norm_obs=False, norm_reward=False, norm_obs_keys=["vect_obs"])

env.reset()
# video_recorder = VideoRecorderCallback(env, render_freq=100)
model = SAC.load(f"./runs/grasp/hha_single_object/hha_single_object_160000_steps", env=env)
model.learn(total_timesteps=150000, log_interval=10 ,tb_log_name=tb_log_name, callback=checkpoint_callback, progress_bar=True)
model.save_replay_buffer(f"./runs/grasp/replay_buffer/{tb_log_name}")
model.save(f"./runs/grasp/{tb_log_name}_final")
# env.save(f"{tb_log_name}_vec_normalize")
# model = SAC.load("sac_queenie", env=env)
# vec_env = model.get_env()
# obs= vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs)
#     obs, reward, done, info = vec_env.step(action)
