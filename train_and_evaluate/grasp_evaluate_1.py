import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
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
tb_log_name = "100obj_rgb_gripper_normvectobs"
env = gym.make("queenie_gym_envs/RandomURDFsSOEnvironment-v1", config_file="./underactuated_manipulation_gym/resources/config/environment_config/simple_manipulation_1.yaml")
env = Monitor(env)
env = DummyVecEnv([lambda: env])
# Load the previously normalized environment
env = VecNormalize.load(f"{tb_log_name}_vec_normalize", env)
env.training = False  # Disable training mode if you're only doing evaluation
env.norm_reward = False  # Since you're not normalizing reward in your original setup
# robot = env.get_robot()
obs = env.reset()
model = SAC.load(f"{tb_log_name}_final", env=env)

# obs = convert_obs_from_single_to_vec(obs)
for i in range(10000):
    action, _state = model.predict(obs)
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
