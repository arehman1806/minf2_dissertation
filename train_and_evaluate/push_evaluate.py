import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
sys.path.append("/home/arehman/dissertation/")
import underactuated_manipulation_gym
# from video_record_callback import VideoRecorderCallback
import time
import numpy as np
import pybullet as p

exp_name = "single_push_linear_reward"

def convert_obs_from_single_to_vec(obs):
    obs["image_obs"] = np.expand_dims(obs["image_obs"], axis=0)
    obs["vect_obs"] = np.expand_dims(obs["vect_obs"], axis=0)
    return obs

def convert_action_from_vec_to_single(action):
    action = action[0]
    return action

env = gym.make("queenie_gym_envs/PushEnvironment-v0", config_file="./underactuated_manipulation_gym/resources/config/environment_config/push_environment.yaml")
env = DummyVecEnv([lambda: env])
# env = VecNormalize(env, norm_obs=False, norm_reward=False, norm_obs_keys=["vect_obs"])
# env.load(f"./runs/push/vec_normalize/{exp_name}", env)
# robot = env.get_robot()
model = PPO.load(f"./runs/push/{exp_name}/{exp_name}_400000_steps.zip", env=env)
obs = env.reset()
# obs = convert_obs_from_single_to_vec(obs)
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    # action = convert_action_from_vec_to_single(action)
    obs, reward, done, _ = env.step(action)
    # obs = convert_obs_from_single_to_vec(obs)
    if done:
        # action = np.concatenate((np.array([0, 0]), action[2:4], np.array([-1])))
        # for i in range(1000):
        #     robot.apply_action(action, use_gripper=True)
        #     p.stepSimulation(robot.client)
        # action[-3] = 1
        # for i in range(1000):
        #     robot.apply_action(action, use_gripper=True)
        #     p.stepSimulation(robot.client)
        # time.sleep(2)
        # action[-1] = 1
        # robot.apply_action(action, use_gripper=True)
        # p.stepSimulation(robot.client)
        # time.sleep(2)
        env.reset()
