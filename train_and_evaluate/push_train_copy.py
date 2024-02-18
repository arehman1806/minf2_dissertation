import gymnasium as gym
from stable_baselines3 import SAC, PPO
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
tb_log_name = "single_object_SAC_hha"
# tb_log_name = "testing_videos"
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=40000,
  save_path=f"./runs/push/{tb_log_name}/",
  name_prefix=tb_log_name,
  save_replay_buffer=False,
  save_vecnormalize=False,
)

env = gym.make("queenie_gym_envs/PushEnvironment-v0", config_file="./underactuated_manipulation_gym/resources/config/environment_config/push_environment.yaml")
env = Monitor(env)
env = DummyVecEnv([lambda: env])
# env = VecNormalize(env, norm_obs=True, norm_reward=True, norm_obs_keys=["vect_obs"])

env.reset()
# video_recorder = VideoRecorderCallback(env, render_freq=100)
# model = SAC("MultiInputPolicy", env, verbose=1, buffer_size=200000, tensorboard_log="./logs/push_agent")
policy_kwargs = dict(
    net_arch=dict(pi=[512, 512, 512, 512], vf=[512, 512, 512, 512])
)
model = SAC("MultiInputPolicy", env, buffer_size=200000, verbose=1, tensorboard_log="./logs/push")
model.learn(total_timesteps=200000, log_interval=100, tb_log_name=tb_log_name, callback=checkpoint_callback, progress_bar=True)
model.save(f"./runs/push{tb_log_name}_last")
# env.save(f"./runs/push/vec_normalize/{tb_log_name}")
model.save_replay_buffer(f"./runs/push/replay_buffer/{tb_log_name}")
# model = SAC.load("sac_queenie", env=env)
# vec_env = model.get_env()
# obs= vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs)
#     obs, reward, done, info = vec_env.step(action)
