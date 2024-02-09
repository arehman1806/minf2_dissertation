import gymnasium as gym
from stable_baselines3 import SAC, PPO
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import underactuated_manipulation_gym
# from video_record_callback import VideoRecorderCallback
import time
import numpy as np
import pybullet as p


class BasePolicyExecutor:
    def __init__(self, env_name, env_config_file_path, vec_normalze_path, controllers, model_path, model_class) -> None:
        self._env = self._create_env(env_name, env_config_file_path, vec_normalze_path, controllers)
        self._model = self._load_model(model_path, model_class)
    
    def execute_policy(self, steps=-1):
        done = False
        rewards = 0
        if steps == -1:
            steps = 1e6
        
        while not done and steps > 0:
            action, _state = self._model.predict(obs)
            obs, reward, done, _ = self._env.step(action)
            rewards += reward
            steps -= 1
            if done:
                return obs, rewards, done
        
        return obs, rewards, done
    
    def _create_env(self, env_name, env_config_file_path, vec_normalze_path, controllers):
        env = gym.make(env_name, env_config_file_path, controllers)
        env = DummyVecEnv([lambda: env])
        env = VecNormalize(env, norm_obs=False, norm_reward=False, norm_obs_keys=["vect_obs"])
        env.load(vec_normalze_path, env)
        return env
    
    def _load_model(self, model_path, model_class_str):
        model_class = getattr(sb3, model_class_str)
        model = model_class.load(model_path, env=self._env)
        return model
    



# def convert_obs_from_single_to_vec(obs):
#     obs["image_obs"] = np.expand_dims(obs["image_obs"], axis=0)
#     obs["vect_obs"] = np.expand_dims(obs["vect_obs"], axis=0)
#     return obs

# def convert_action_from_vec_to_single(action):
#     action = action[0]
#     return action

# env = gym.make("queenie_gym_envs/PushEnvironment-v0", config_file="./underactuated_manipulation_gym/resources/config/environment_config/push_environment.yaml")
# env = DummyVecEnv([lambda: env])
# env = VecNormalize(env, norm_obs=False, norm_reward=False, norm_obs_keys=["vect_obs"])
# env.load("single_object_push_256x3_networks_vec_normalize", env)
# # robot = env.get_robot()
# model = PPO.load("single_object_push_256x3_networks/single_object_push_256x3_networks_500000_steps.zip", env=env)
# obs = env.reset()
# done = False
# # obs = convert_obs_from_single_to_vec(obs)
# while not done:
#     action, _state = model.predict(obs)
#     # action = convert_action_from_vec_to_single(action)
#     obs, reward, done, _ = env.step(action)
#     # obs = convert_obs_from_single_to_vec(obs)
#     if done:
#         # action = np.concatenate((np.array([0, 0]), action[2:4], np.array([-1])))
#         # for i in range(1000):
#         #     robot.apply_action(action, use_gripper=True)
#         #     p.stepSimulation(robot.client)
#         # action[-3] = 1
#         # for i in range(1000):
#         #     robot.apply_action(action, use_gripper=True)
#         #     p.stepSimulation(robot.client)
#         # time.sleep(2)
#         # action[-1] = 1
#         # robot.apply_action(action, use_gripper=True)
#         # p.stepSimulation(robot.client)
#         # time.sleep(2)
#         pass
