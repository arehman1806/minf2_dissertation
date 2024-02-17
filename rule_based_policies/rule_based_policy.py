import numpy as np
from stable_baselines3.common.env_util import unwrap_wrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

class RuleBasedPolicy():
    """Base class for rule-based policies."""

    def __init__(self, env):
        self._env = env
        self._rules = []
        self.proprioception_indices = self._env.envs[0].proprioception_indices

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Predict an action given an observation."""
        raise NotImplementedError
    
    def _unwrap_env(self):
        """Unwrap the environment to get the base environment."""
        env = self._env
        monitor_env = unwrap_wrapper(env, Monitor)