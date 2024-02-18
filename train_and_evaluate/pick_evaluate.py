import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
sys.path.append("/home/arehman/dissertation/")
import underactuated_manipulation_gym
# from video_record_callback import VideoRecorderCallback
import time
import numpy as np
import pybullet as p


from rule_based_policies.pick import PickPolicy

exp_name = "single_object_push_256x3_networks"

env = gym.make("queenie_gym_envs/PickEnvironment-v0", config_file="./underactuated_manipulation_gym/resources/config/environment_config/pick_environment.yaml")
env = DummyVecEnv([lambda: env])
# robot = env.get_robot()
obs = env.reset()
model = PickPolicy(env)
# obs = convert_obs_from_single_to_vec(obs)
for i in range(1000):
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
