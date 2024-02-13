import gymnasium as gym
from stable_baselines3 import SAC, PPO
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import underactuated_manipulation_gym
# from video_record_callback import VideoRecorderCallback
import time
import numpy as np
import pybullet as p
import yaml


class BasePolicyExecutor:
    def __init__(self, env_config_file_path, controllers) -> None:
        self._env_config = self._parse_config(env_config_file_path)["environment"]
        self._env = self._create_env(env_config_file_path, controllers)
        self._model = self._load_model()
    
    def execute_policy(self, steps=-1):
        done = False
        s = self._env._get_observation()[0]
        if s in self._env.initial_state(s):
            done = True
            obs = s
            rewards = -1
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
    
    def _create_env(self, env_config_file_path, controllers):
        env_name = self._env_config["name"]
        # vec_normalze_path = self._env_config["vec_normalize_path"]
        env = gym.make(f"queenie_gym_envs/{env_name}", config_file=env_config_file_path, controllers=controllers, as_subpolicy=True)
        env = DummyVecEnv([lambda: env])
        # env = VecNormalize(env, norm_obs=False, norm_reward=False, norm_obs_keys=["vect_obs"])
        # env.load(vec_normalze_path, env)
        return env
    
    def _load_model(self):
        model_path = self._env_config["model_path"]
        model_class_str = self._env_config["model_class"]
        model_class = getattr(sb3, model_class_str)
        model = model_class.load(model_path, env=self._env)
        return model
    
    """loads yaml config file and returns a dictionary"""
    def _parse_config(self, config_file):
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
        