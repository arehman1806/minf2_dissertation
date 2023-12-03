import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import underactuated_manipulation_gym
# from video_record_callback import VideoRecorderCallback
import time
import numpy as np

# IMPORTANT. edit this before every run:
tb_log_name = "100_objects_rgb_multi_object_palm_contact_fixed_gripper_free_joints_2"
# tb_log_name = "testing_videos"
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=50000,
  save_path=f"./{tb_log_name}/",
  name_prefix=tb_log_name,
  save_replay_buffer=False,
  save_vecnormalize=False,
)

env = gym.make("queenie_gym_envs/RandomURDFsSOEnvironment-v0", config_file="./underactuated_manipulation_gym/resources/config/environment_config/simple_manipulation.yaml")
env.reset()
# video_recorder = VideoRecorderCallback(env, render_freq=100)
model = SAC("MultiInputPolicy", env, verbose=1, buffer_size=500000, tensorboard_log="./logs/simple_multi_object_pickup_agent")
model.learn(total_timesteps=500000, log_interval=10, tb_log_name=tb_log_name, callback=checkpoint_callback, progress_bar=True)
model.save(f"{tb_log_name}_final")
# model = SAC.load("sac_queenie", env=env)
# vec_env = model.get_env()
# obs= vec_env.reset()
# for i in range(1000):
#     action, _state = model.predict(obs)
#     obs, reward, done, info = vec_env.step(action)
