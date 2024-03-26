import gymnasium as gym
import cv2
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
import pybullet as p


def convert_obs_from_single_to_vec(obs):
    obs["image_obs"] = np.expand_dims(obs["image_obs"], axis=0)
    obs["vect_obs"] = np.expand_dims(obs["vect_obs"], axis=0)
    return obs

def convert_action_from_vec_to_single(action):
    action = action[0]
    return action
tb_log_name = "hha_single_object_160000_steps.zip"
env = gym.make("queenie_gym_envs/GraspEnvironment-v1", config_file="./underactuated_manipulation_gym/resources/config/environment_config/grasp_environment_1.yaml")
env = Monitor(env)
env = DummyVecEnv([lambda: env])
# Load the previously normalized environment
# env = VecNormalize.load(f"{tb_log_name}_vec_normalize", env)
env.training = False  # Disable training mode if you're only doing evaluation
env.norm_reward = False  # Since you're not normalizing reward in your original setup
# robot = env.get_robot()
obs = env.reset()
model = SAC.load(f"./runs/grasp/hha_single_object/{tb_log_name}", env=env)

# obs = convert_obs_from_single_to_vec(obs)
# model.policy.set_training_mode(False)
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    # action = convert_action_from_vec_to_single(action)
    obs, reward, done, _ = env.step(action)
    # obs["image_obs"][0][2] = np.zeros((84,84))
    # cv2.imshow("1", obs["image_obs"][0][0])
    # cv2.imshow("2", obs["image_obs"][0][1])
    # cv2.imshow("3", obs["image_obs"][0][2])
    # cv2.waitKey(1)
    # obs["vect_obs"] = np.zeros((1, len(obs["vect_obs"][0])))
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
        # env.reset()
        pass
