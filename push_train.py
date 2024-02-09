import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import underactuated_manipulation_gym
# from video_record_callback import VideoRecorderCallback
import time
import numpy as np

# IMPORTANT. edit this before every run:
tb_log_name = "single_object_push_256x3_networks"
# tb_log_name = "testing_videos"
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=50000,
  save_path=f"./{tb_log_name}/",
  name_prefix=tb_log_name,
  save_replay_buffer=False,
  save_vecnormalize=False,
)

env = gym.make("queenie_gym_envs/PushEnvironment-v0", config_file="./underactuated_manipulation_gym/resources/config/environment_config/push_environment.yaml")
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=False, norm_obs_keys=["vect_obs"])

env.reset()
# video_recorder = VideoRecorderCallback(env, render_freq=100)
# model = SAC("MultiInputPolicy", env, verbose=1, buffer_size=200000, tensorboard_log="./logs/push_agent")
policy_kwargs = dict(
    net_arch=dict(pi=[512, 512, 512, 512], vf=[512, 512, 512, 512])
)
model = PPO("MultiInputPolicy", env, verbose=1, batch_size=256, tensorboard_log="./logs/push_agent", policy_kwargs=policy_kwargs)
model.learn(total_timesteps=1000000, log_interval=10, tb_log_name=tb_log_name, callback=checkpoint_callback, progress_bar=True)
model.save(f"{tb_log_name}_model")
env.save(f"{tb_log_name}_vec_normalize")
# model = SAC.load("sac_queenie", env=env)
# vec_env = model.get_env()
# obs= vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs)
#     obs, reward, done, info = vec_env.step(action)